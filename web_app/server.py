from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import hashlib
import ast
import struct
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles


ROOT = Path("/scratch/e1554543/avatar_system_full")
WEB_ROOT = ROOT / "web_app"
STATIC_ROOT = WEB_ROOT / "static"
OUTPUT_ROOT = ROOT / "outputs"
UPLOAD_ROOT = OUTPUT_ROOT / "web_uploads"
CACHE_ROOT = ROOT / "cache" / "pipeline"
SCRIPT = ROOT / "scripts" / "run_agent.sh"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Avatar Web Studio")
app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")

jobs: dict[str, dict[str, Any]] = {}
viewer_exports: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()
exports_lock = threading.Lock()
settings_lock = threading.Lock()
runtime_settings: dict[str, str] = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
    "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    "LLM_MODEL": os.environ.get("LLM_MODEL", "openai/gpt-oss-120b:free"),
}


class RuntimeSettings(BaseModel):
    openai_api_key: str = ""
    openai_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-oss-120b:free"


def _load_avatar_labels() -> dict[str, str]:
    labels: dict[str, str] = {
        "306": "GaussianAvatars demo",
        "1001": "demo1 custom subject",
    }

    map_path = (
        ROOT
        / "GSavatar_runs"
        / "GaussianAvatars"
        / "datasets"
        / "nersemble_preprocessed"
        / "nersemble_avatar_map.tsv"
    )
    if map_path.exists():
        lines = map_path.read_text(encoding="utf-8").splitlines()
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) >= 3:
                subject, _source_dir, avatar_id = parts[:3]
                labels[str(avatar_id)] = f"NeRSemble {subject}"
    return labels


def _list_available_avatars() -> list[dict[str, str]]:
    media_root = ROOT / "GSavatar_runs" / "GaussianAvatars" / "media"
    labels = _load_avatar_labels()
    avatars: list[dict[str, str]] = []
    if not media_root.exists():
        return avatars

    for path in sorted(media_root.iterdir(), key=lambda p: p.name):
        if not path.is_dir():
            continue
        avatar_id = path.name
        if not (path / "point_cloud.ply").exists():
            continue
        label = labels.get(avatar_id, f"Avatar {avatar_id}")
        avatars.append({"id": avatar_id, "label": label})
    return avatars


class ViewerExportRequest(BaseModel):
    camera: dict[str, Any]
    width: int = 550
    height: int = 802
    fps: int = 25
    render_mode: str = "gaussian"


class ViewerFrameRequest(BaseModel):
    camera: dict[str, Any]
    frame: int = 0
    width: int = 550
    height: int = 802


class ViewerSplatMotionRequest(BaseModel):
    frame_stride: int = 1
    max_frames: int = 360


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug[:80] or "audio"


def _tail(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) > max_chars:
        data = data[-max_chars:]
    return data.decode("utf-8", errors="replace")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _is_web_run_dir(path: Path) -> bool:
    if not path.is_dir() or not path.name.startswith("web_"):
        return False
    if path.name == "web_uploads":
        return False
    return (path / "state.json").exists() or (path / "manifest.json").exists()


def _prune_output_runs(keep: int = 5) -> None:
    keep = max(1, int(keep))
    run_dirs = [path for path in OUTPUT_ROOT.iterdir() if _is_web_run_dir(path)]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    kept_names = {path.name for path in run_dirs[:keep]}
    for old_dir in run_dirs[keep:]:
        shutil.rmtree(old_dir, ignore_errors=True)
        upload_wav = UPLOAD_ROOT / f"{old_dir.name}.wav"
        upload_wav.unlink(missing_ok=True)
    for upload_wav in UPLOAD_ROOT.glob("web_*.wav"):
        if upload_wav.stem not in kept_names:
            upload_wav.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key(
    *,
    audio_sha256: str,
    avatar_id: str,
    no_llm: bool,
    no_video_export: bool,
    prepare_only: bool,
    settings: dict[str, str],
) -> str:
    cache_obj = {
        "schema": 1,
        "audio_sha256": audio_sha256,
        "avatar_id": str(avatar_id),
        "no_llm": bool(no_llm),
        "no_video_export": bool(no_video_export),
        "prepare_only": bool(prepare_only),
        "openai_base_url": settings.get("OPENAI_BASE_URL", ""),
        "llm_model": "" if no_llm else settings.get("LLM_MODEL", ""),
    }
    encoded = json.dumps(cache_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _copy_if_exists(src: str | None, dst: Path) -> str | None:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return str(dst)


def _cache_entry_path(cache_key: str) -> Path:
    return CACHE_ROOT / cache_key[:2] / cache_key


def _cache_manifest_path(cache_key: str) -> Path:
    return _cache_entry_path(cache_key) / "manifest.json"


def _restore_cached_run(
    *,
    cache_key: str,
    run_id: str,
    run_dir: Path,
    avatar_id: str,
    input_wav: Path,
    prepare_only: bool,
) -> bool:
    cache_dir = _cache_entry_path(cache_key)
    cached_manifest = _load_json(cache_dir / "manifest.json")
    if not cached_manifest or cached_manifest.get("error"):
        return False

    artifacts_dir = run_dir / "artifacts"
    outputs_dir = run_dir / "outputs"
    logs_dir = run_dir / "logs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cached_artifacts = cache_dir / "artifacts"
    reply_wav = _copy_if_exists(str(cached_artifacts / "reply.wav"), outputs_dir / "reply.wav")
    artifact_reply_wav = _copy_if_exists(str(cached_artifacts / "reply.wav"), artifacts_dir / "reply.wav")
    enhanced_wav = _copy_if_exists(str(cached_artifacts / "reply_enhanced.wav"), artifacts_dir / "reply_enhanced.wav")
    flame_npz = _copy_if_exists(str(cached_artifacts / "flame_motion.npz"), artifacts_dir / "flame_motion.npz")
    output_video = _copy_if_exists(str(cached_artifacts / "final_video.mp4"), artifacts_dir / "final_video.mp4")
    white_video = _copy_if_exists(str(cached_artifacts / "white_model.mp4"), artifacts_dir / "white_model.mp4")
    deeptalk_npy = _copy_if_exists(str(cached_artifacts / "deeptalk.npy"), outputs_dir / "deeptalk.npy")

    state = {
        "input_wav": str(input_wav),
        "avatar_id": str(avatar_id),
        "base_name": run_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "log_dir": str(logs_dir),
        "prepare_only": bool(prepare_only),
        "launch_viewer": False,
        "current_stage": "done",
        "finished_stages": [
            "cache_restore",
            "perception",
            "task1",
            "emotivoice_prepare",
            "emotivoice_tts",
            "deeptalk",
            "flame_merge",
            "viewer",
            "artifact_export",
        ],
        "failed_stage": None,
        "error": None,
        "perception_json": cached_manifest.get("perception_json"),
        "task1_input_json": cached_manifest.get("task1_input_json"),
        "task1_reply_json": cached_manifest.get("task1_reply_json"),
        "reply_text": cached_manifest.get("reply_text"),
        "reply_style": cached_manifest.get("reply_style"),
        "emotivoice_txt": cached_manifest.get("emotivoice_txt"),
        "reply_wav": reply_wav,
        "deeptalk_npy": deeptalk_npy,
        "flame_motion_npz": flame_npz,
        "point_cloud_path": cached_manifest.get("point_cloud_path"),
        "template_npz": cached_manifest.get("template_npz"),
        "viewer_command": cached_manifest.get("viewer_command"),
        "viewer_started": False,
        "viewer_pid": None,
        "artifact_dir": str(artifacts_dir),
        "artifact_reply_wav": artifact_reply_wav,
        "artifact_flame_motion_npz": flame_npz,
        "output_video": output_video,
        "output_white_model_video": white_video,
        "artifact_enhanced_reply_wav": enhanced_wav,
        "video_export_command": cached_manifest.get("video_export_command"),
        "video_export_error": None,
        "extra": {
            "cache_hit": True,
            "cache_key": cache_key,
            "cached_from_run_id": cached_manifest.get("run_id"),
        },
    }

    manifest = {
        "run_id": run_id,
        "input_wav": str(input_wav),
        "avatar_id": str(avatar_id),
        "task1_reply_json": state["task1_reply_json"],
        "reply_text": state["reply_text"],
        "reply_wav": reply_wav,
        "deeptalk_npy": deeptalk_npy,
        "flame_motion_npz": flame_npz,
        "point_cloud_path": state["point_cloud_path"],
        "viewer_command": state["viewer_command"],
        "viewer_started": False,
        "viewer_pid": None,
        "artifact_dir": str(artifacts_dir),
        "artifact_reply_wav": artifact_reply_wav,
        "artifact_enhanced_reply_wav": enhanced_wav,
        "artifact_flame_motion_npz": flame_npz,
        "output_video": output_video,
        "output_white_model_video": white_video,
        "video_export_command": state["video_export_command"],
        "video_export_error": None,
        "finished_stages": state["finished_stages"],
        "failed_stage": None,
        "error": None,
        "run_dir": str(run_dir),
        "log_dir": str(logs_dir),
        "cache_hit": True,
        "cache_key": cache_key,
        "cached_from_run_id": cached_manifest.get("run_id"),
    }

    with (run_dir / "state.json").open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with (artifacts_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with (run_dir / "web_stdout.log").open("w", encoding="utf-8") as f:
        f.write(f"[cache] hit {cache_key}\n")
        f.write(f"[cache] restored from run {cached_manifest.get('run_id')}\n")
        f.write("[cache] skipped pipeline execution\n")
    return True


def _store_cache(cache_key: str, run_id: str) -> None:
    run_dir = OUTPUT_ROOT / run_id
    manifest = _load_json(run_dir / "manifest.json")
    state = _load_json(run_dir / "state.json")
    if not manifest or manifest.get("error") or (state and state.get("error")):
        return

    cache_dir = _cache_entry_path(cache_key)
    artifacts_dir = cache_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    artifact_map = {
        "artifact_reply_wav": "reply.wav",
        "artifact_enhanced_reply_wav": "reply_enhanced.wav",
        "artifact_flame_motion_npz": "flame_motion.npz",
        "output_video": "final_video.mp4",
        "output_white_model_video": "white_model.mp4",
        "deeptalk_npy": "deeptalk.npy",
    }
    for key, filename in artifact_map.items():
        src = manifest.get(key) or (state or {}).get(key)
        _copy_if_exists(src, artifacts_dir / filename)

    cache_manifest = dict(manifest)
    if state:
        for key in [
            "perception_json",
            "task1_input_json",
            "reply_style",
            "emotivoice_txt",
            "template_npz",
        ]:
            if key in state and key not in cache_manifest:
                cache_manifest[key] = state[key]
    cache_manifest["cache_key"] = cache_key
    cache_manifest["cached_from_run_id"] = run_id
    with (cache_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(cache_manifest, f, ensure_ascii=False, indent=2)


def _artifact_urls(manifest: dict[str, Any] | None) -> dict[str, str]:
    if not manifest:
        return {}
    urls: dict[str, str] = {}
    for key in [
        "output_video",
        "output_white_model_video",
        "artifact_reply_wav",
        "artifact_enhanced_reply_wav",
        "artifact_flame_motion_npz",
    ]:
        value = manifest.get(key)
        if value:
            path = Path(value)
            try:
                rel = path.resolve().relative_to(OUTPUT_ROOT.resolve())
            except ValueError:
                continue
            urls[key] = "/outputs/" + str(rel).replace(os.sep, "/")
    return urls


def _outputs_url(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    try:
        rel = path.resolve().relative_to(OUTPUT_ROOT.resolve())
    except ValueError:
        return None
    return "/outputs/" + str(rel).replace(os.sep, "/")


def _npy_shape_from_member(zf: zipfile.ZipFile, member: str) -> tuple[int, ...] | None:
    try:
        with zf.open(member) as f:
            magic = f.read(6)
            if magic != b"\x93NUMPY":
                return None
            major = f.read(1)[0]
            f.read(1)
            if major == 1:
                header_len = struct.unpack("<H", f.read(2))[0]
            else:
                header_len = struct.unpack("<I", f.read(4))[0]
            header = f.read(header_len).decode("latin1").strip()
        info = ast.literal_eval(header)
        shape = info.get("shape")
        if isinstance(shape, tuple):
            return tuple(int(v) for v in shape)
    except Exception:
        return None
    return None


def _npz_frame_count(path_value: str | None) -> int | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    try:
        with zipfile.ZipFile(path) as zf:
            for member in ("expr.npy", "jaw_pose.npy", "dynamic_offset.npy", "timestep_id.npy"):
                if member not in zf.namelist():
                    continue
                shape = _npy_shape_from_member(zf, member)
                if shape and len(shape) > 0:
                    return int(shape[0])
    except zipfile.BadZipFile:
        return None
    return None


def _job_payload(run_id: str) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / run_id
    manifest = _load_json(run_dir / "manifest.json")
    state = _load_json(run_dir / "state.json")
    web_log = run_dir / "web_stdout.log"

    with jobs_lock:
        job = jobs.get(run_id, {}).copy()

    proc = job.get("process")
    if proc is not None:
        return_code = proc.poll()
        if return_code is None:
            status = "running"
        elif return_code == 0:
            status = "done" if manifest and not manifest.get("error") else "failed"
        else:
            status = "failed"
    else:
        status = "done" if manifest and not manifest.get("error") else "unknown"

    if manifest and manifest.get("error"):
        status = "failed"
    if state and state.get("error"):
        status = "failed"

    payload = {
        "run_id": run_id,
        "status": status,
        "return_code": None if proc is None else proc.poll(),
        "run_dir": str(run_dir),
        "log_dir": str(run_dir / "logs"),
        "manifest": manifest,
        "state": state,
        "artifact_urls": _artifact_urls(manifest),
        "log_tail": _tail(web_log),
    }
    with exports_lock:
        payload["viewer_exports"] = [
            export.copy()
            for export in viewer_exports.values()
            if export.get("run_id") == run_id
        ]
    return payload


def _viewer_export_url(path_value: str | None) -> str | None:
    return _outputs_url(path_value)


def _run_viewer_export(export_key: str, command: str, out_video: Path, log_path: Path) -> None:
    with exports_lock:
        if export_key in viewer_exports:
            viewer_exports[export_key]["status"] = "running"
            viewer_exports[export_key]["log_path"] = str(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + command + "\n\n")

    proc = subprocess.run(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(proc.stdout or "")
        log.write(f"\n[exit code: {proc.returncode}]\n")

    status = "done" if proc.returncode == 0 and out_video.exists() else "failed"
    error = None if status == "done" else (proc.stdout or f"Output not found: {out_video}")
    with exports_lock:
        if export_key in viewer_exports:
            viewer_exports[export_key]["status"] = status
            viewer_exports[export_key]["return_code"] = proc.returncode
            viewer_exports[export_key]["output_video"] = str(out_video) if out_video.exists() else None
            viewer_exports[export_key]["output_video_url"] = _viewer_export_url(str(out_video)) if out_video.exists() else None
            viewer_exports[export_key]["error"] = error


def _viewer_asset_paths(run_id: str) -> dict[str, str | None]:
    run_dir = OUTPUT_ROOT / run_id
    manifest = _load_json(run_dir / "manifest.json") or {}
    state = _load_json(run_dir / "state.json") or {}
    return {
        "point_cloud_path": state.get("point_cloud_path") or manifest.get("point_cloud_path"),
        "motion_path": (
            manifest.get("artifact_flame_motion_npz")
            or state.get("artifact_flame_motion_npz")
            or state.get("flame_motion_npz")
        ),
        "audio_path": (
            manifest.get("artifact_enhanced_reply_wav")
            or manifest.get("artifact_reply_wav")
            or state.get("artifact_enhanced_reply_wav")
            or state.get("artifact_reply_wav")
            or state.get("reply_wav")
        ),
    }


def _watch_process(run_id: str, proc: subprocess.Popen[str], log_path: Path) -> None:
    assert proc.stdout is not None
    with log_path.open("a", encoding="utf-8") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
    return_code = proc.wait()
    cache_key = None
    with jobs_lock:
        if run_id in jobs:
            jobs[run_id]["return_code"] = return_code
            cache_key = jobs[run_id].get("cache_key")
    if return_code == 0 and cache_key:
        try:
            _store_cache(str(cache_key), run_id)
        except Exception as exc:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[cache] store failed: {exc}\n")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/settings")
def get_settings() -> JSONResponse:
    with settings_lock:
        key = runtime_settings.get("OPENAI_API_KEY", "")
        base_url = runtime_settings.get("OPENAI_BASE_URL", "")
        model = runtime_settings.get("LLM_MODEL", "")
    return JSONResponse(
        {
            "has_openai_api_key": bool(key),
            "openai_key_preview": f"{key[:8]}...{key[-4:]}" if len(key) >= 16 else "",
            "openai_base_url": base_url,
            "llm_model": model,
        }
    )


@app.post("/api/settings")
def set_settings(settings: RuntimeSettings) -> JSONResponse:
    with settings_lock:
        if settings.openai_api_key.strip():
            runtime_settings["OPENAI_API_KEY"] = settings.openai_api_key.strip()
        runtime_settings["OPENAI_BASE_URL"] = settings.openai_base_url.strip() or "https://openrouter.ai/api/v1"
        runtime_settings["LLM_MODEL"] = settings.llm_model.strip() or "openai/gpt-oss-120b:free"
    return get_settings()


@app.get("/api/avatars")
def get_avatars() -> JSONResponse:
    return JSONResponse({"avatars": _list_available_avatars()})


@app.post("/api/jobs")
async def create_job(
    audio: UploadFile = File(...),
    avatar_id: str = Form("306"),
    no_llm: bool = Form(False),
    no_video_export: bool = Form(False),
    prepare_only: bool = Form(False),
) -> JSONResponse:
    with settings_lock:
        active_settings = runtime_settings.copy()

    if not no_llm and not active_settings.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail=(
                "OPENAI_API_KEY is not configured. "
                "Enter your key in Settings and save it; "
                "or enable Skip LLM for a test run."
            ),
        )

    filename = audio.filename or "recording.wav"
    if not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Please upload or record a .wav file.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _safe_slug(Path(filename).stem)
    run_id = f"web_{stem}_{stamp}"
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _prune_output_runs(keep=5)

    upload_path = UPLOAD_ROOT / f"{run_id}.wav"
    with upload_path.open("wb") as f:
        shutil.copyfileobj(audio.file, f)

    audio_sha256 = _sha256_file(upload_path)
    cache_key = _cache_key(
        audio_sha256=audio_sha256,
        avatar_id=str(avatar_id),
        no_llm=bool(no_llm),
        no_video_export=bool(no_video_export),
        prepare_only=bool(prepare_only),
        settings=active_settings,
    )

    if _cache_manifest_path(cache_key).exists():
        restored = _restore_cached_run(
            cache_key=cache_key,
            run_id=run_id,
            run_dir=run_dir,
            avatar_id=str(avatar_id),
            input_wav=upload_path,
            prepare_only=bool(prepare_only),
        )
        if restored:
            with jobs_lock:
                jobs[run_id] = {
                    "process": None,
                    "created_at": stamp,
                    "input_wav": str(upload_path),
                    "avatar_id": str(avatar_id),
                    "return_code": 0,
                    "cache_key": cache_key,
                    "cache_hit": True,
                }
            return JSONResponse(_job_payload(run_id))

    web_log = run_dir / "web_stdout.log"
    cmd = [
        "bash",
        str(SCRIPT),
        str(upload_path),
        str(avatar_id),
        "--run_id",
        run_id,
    ]
    if prepare_only:
        cmd.append("--prepare_only")
    if no_video_export:
        cmd.append("--no_video_export")
    if no_llm:
        cmd.append("--no_llm")

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(ROOT / "cache" / "hf"))
    env.setdefault("XDG_CACHE_HOME", str(ROOT / "cache" / "xdg"))
    env.setdefault("MODELSCOPE_CACHE", str(ROOT / "cache" / "modelscope"))
    env.setdefault("NLTK_DATA", str(ROOT / "cache" / "nltk_data"))
    env["OPENAI_API_KEY"] = active_settings.get("OPENAI_API_KEY", "")
    env["OPENAI_BASE_URL"] = active_settings.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    env["LLM_MODEL"] = active_settings.get("LLM_MODEL", "openai/gpt-oss-120b:free")

    with web_log.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    with jobs_lock:
        jobs[run_id] = {
            "process": proc,
            "created_at": stamp,
            "input_wav": str(upload_path),
            "avatar_id": str(avatar_id),
            "return_code": None,
            "cache_key": cache_key,
            "cache_hit": False,
        }
    thread = threading.Thread(target=_watch_process, args=(run_id, proc, web_log), daemon=True)
    thread.start()

    return JSONResponse(_job_payload(run_id))


@app.get("/api/jobs/{run_id}")
def get_job(run_id: str) -> JSONResponse:
    if not re.match(r"^[A-Za-z0-9_.-]+$", run_id):
        raise HTTPException(status_code=400, detail="Invalid run id.")
    run_dir = OUTPUT_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(_job_payload(run_id))


@app.get("/api/jobs/{run_id}/viewer_assets")
def get_viewer_assets(run_id: str) -> JSONResponse:
    if not re.match(r"^[A-Za-z0-9_.-]+$", run_id):
        raise HTTPException(status_code=400, detail="Invalid run id.")
    run_dir = OUTPUT_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found.")

    assets = _viewer_asset_paths(run_id)
    point_cloud_path = assets["point_cloud_path"]
    flame_motion = assets["motion_path"]
    audio = assets["audio_path"]

    has_point_cloud = bool(point_cloud_path and Path(point_cloud_path).exists())
    payload = {
        "run_id": run_id,
        "fps": 25,
        "frame_count": _npz_frame_count(flame_motion),
        "point_cloud_url": f"/api/jobs/{run_id}/viewer/point_cloud" if has_point_cloud else None,
        "audio_url": _outputs_url(audio),
        "motion_url": _outputs_url(flame_motion),
        "point_cloud_path": point_cloud_path if has_point_cloud else None,
        "flame_motion_path": flame_motion if flame_motion and Path(flame_motion).exists() else None,
    }
    return JSONResponse(payload)


@app.get("/api/jobs/{run_id}/viewer/point_cloud")
def get_viewer_point_cloud(run_id: str) -> FileResponse:
    if not re.match(r"^[A-Za-z0-9_.-]+$", run_id):
        raise HTTPException(status_code=400, detail="Invalid run id.")
    run_dir = OUTPUT_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found.")

    manifest = _load_json(run_dir / "manifest.json") or {}
    state = _load_json(run_dir / "state.json") or {}
    point_cloud_path = state.get("point_cloud_path") or manifest.get("point_cloud_path")
    if not point_cloud_path:
        raise HTTPException(status_code=404, detail="Point cloud not found.")

    path = Path(point_cloud_path).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Point cloud path is outside project root.")
    if not path.exists() or path.suffix.lower() != ".ply":
        raise HTTPException(status_code=404, detail="Point cloud not found.")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@app.post("/api/jobs/{run_id}/viewer/render_frame")
def render_viewer_frame(run_id: str, request: ViewerFrameRequest) -> Response:
    if not re.match(r"^[A-Za-z0-9_.-]+$", run_id):
        raise HTTPException(status_code=400, detail="Invalid run id.")
    run_dir = OUTPUT_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found.")

    assets = _viewer_asset_paths(run_id)
    point_cloud_path = assets["point_cloud_path"]
    motion_path = assets["motion_path"]
    for label, value in [("point cloud", point_cloud_path), ("motion npz", motion_path)]:
        if not value or not Path(value).exists():
            raise HTTPException(status_code=400, detail=f"Missing {label} for render preview.")

    frame = max(0, int(request.frame))

    worker_url = os.environ.get("GAUSSIAN_RENDER_WORKER_URL", "http://127.0.0.1:8792").rstrip("/")
    try:
        with urllib.request.urlopen(f"{worker_url}/health", timeout=2) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        if not health.get("ok"):
            raise RuntimeError(f"unhealthy response: {health}")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Gaussian render worker unavailable: {exc}") from exc

    payload = json.dumps(
        {
            "point_path": point_cloud_path,
            "motion_path": motion_path,
            "camera": request.camera,
            "frame": frame,
            "width": int(request.width),
            "height": int(request.height),
            "image_format": "jpeg",
            "quality": 86,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{worker_url}/render_frame_bytes",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            image_bytes = resp.read()
            content_type = resp.headers.get_content_type() or "image/jpeg"
            frame_header = resp.headers.get("X-Frame", str(frame))
            frame_count_header = resp.headers.get("X-Frame-Count", "")
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise HTTPException(status_code=500, detail=f"Gaussian render worker failed: {exc}") from exc
    headers = {"Cache-Control": "no-store", "X-Frame": str(frame_header)}
    if frame_count_header:
        headers["X-Frame-Count"] = str(frame_count_header)
    return Response(content=image_bytes, media_type=content_type, headers=headers)


@app.post("/api/jobs/{run_id}/viewer/splat_motion")
def prepare_splat_motion(run_id: str, request: ViewerSplatMotionRequest) -> JSONResponse:
    if not re.match(r"^[A-Za-z0-9_.-]+$", run_id):
        raise HTTPException(status_code=400, detail="Invalid run id.")
    run_dir = OUTPUT_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found.")

    assets = _viewer_asset_paths(run_id)
    point_cloud_path = assets["point_cloud_path"]
    motion_path = assets["motion_path"]
    for label, value in [("point cloud", point_cloud_path), ("motion npz", motion_path)]:
        if not value or not Path(value).exists():
            raise HTTPException(status_code=400, detail=f"Missing {label} for splat motion.")

    frame_stride = max(1, int(request.frame_stride))
    max_frames = max(1, min(int(request.max_frames), 720))
    motion_dir = run_dir / "artifacts" / "splat_motion"
    motion_dir.mkdir(parents=True, exist_ok=True)
    output_bin = motion_dir / f"gaussians_f16_stride{frame_stride}_max{max_frames}.bin"
    meta_json = motion_dir / f"gaussians_f16_stride{frame_stride}_max{max_frames}.json"
    cached_meta = _load_json(meta_json)
    if cached_meta and output_bin.exists() and output_bin.stat().st_size == int(cached_meta.get("bytes", -1)):
        return JSONResponse({**cached_meta, "motion_url": _outputs_url(str(output_bin))})

    worker_url = os.environ.get("GAUSSIAN_RENDER_WORKER_URL", "http://127.0.0.1:8792").rstrip("/")
    try:
        with urllib.request.urlopen(f"{worker_url}/health", timeout=2) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        if not health.get("ok"):
            raise RuntimeError(f"unhealthy response: {health}")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Gaussian render worker unavailable: {exc}") from exc

    payload = json.dumps(
        {
            "point_path": point_cloud_path,
            "motion_path": motion_path,
            "output_path": str(output_bin),
            "frame_stride": frame_stride,
            "max_frames": max_frames,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{worker_url}/export_motion_positions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Gaussian motion export failed: {exc}") from exc
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=f"Gaussian motion export error: {result}")
    if not output_bin.exists():
        raise HTTPException(status_code=500, detail=f"Splat motion missing: {output_bin}")

    meta = {
        "ok": True,
        "motion": str(output_bin),
        "motion_url": _outputs_url(str(output_bin)),
        "dtype": result.get("dtype", "float16"),
        "point_count": int(result.get("point_count", 0)),
        "values_per_point": int(result.get("values_per_point", 3)),
        "source_frame_count": int(result.get("source_frame_count", 0)),
        "frame_count": int(result.get("frame_count", 0)),
        "frame_stride": int(result.get("frame_stride", frame_stride)),
        "first_frame": int(result.get("first_frame", 0)),
        "last_frame": int(result.get("last_frame", 0)),
        "bytes": int(result.get("bytes", output_bin.stat().st_size)),
    }
    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return JSONResponse(meta)


@app.post("/api/jobs/{run_id}/viewer/export")
def start_viewer_export(run_id: str, request: ViewerExportRequest) -> JSONResponse:
    if not re.match(r"^[A-Za-z0-9_.-]+$", run_id):
        raise HTTPException(status_code=400, detail="Invalid run id.")
    run_dir = OUTPUT_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    if request.render_mode not in {"gaussian", "white_mesh", "overlay"}:
        raise HTTPException(status_code=400, detail="Invalid render mode.")

    assets = _viewer_asset_paths(run_id)
    point_cloud_path = assets["point_cloud_path"]
    motion_path = assets["motion_path"]
    audio_path = assets["audio_path"]
    for label, value in [
        ("point cloud", point_cloud_path),
        ("motion npz", motion_path),
        ("audio", audio_path),
    ]:
        if not value or not Path(value).exists():
            raise HTTPException(status_code=400, detail=f"Missing {label} for export.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_id = f"custom_view_{stamp}"
    export_dir = run_dir / "artifacts" / "custom_views" / export_id
    export_dir.mkdir(parents=True, exist_ok=False)
    camera_json = export_dir / "camera.json"
    with camera_json.open("w", encoding="utf-8") as f:
        json.dump(request.camera, f, ensure_ascii=False, indent=2)

    out_video = export_dir / "custom_view.mp4"
    frames_dir = export_dir / "frames"
    log_path = export_dir / "export.log"
    gaussian_root = ROOT / "GSavatar_runs" / "GaussianAvatars"
    gaussian_python = gaussian_root / ".GSavatar_glibc" / "bin" / "python"
    exporter_py = ROOT / "tools" / "avatar_agent" / "export_gaussian_video.py"
    container_image = ROOT / "containers" / "gaussianav_jammy"
    ffmpeg = ROOT / "tools" / "ffmpeg-git-20240629-amd64-static" / "ffmpeg"

    q = shlex.quote
    inner_cmd = f"""
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false
    cd {q(str(gaussian_root))}
    {q(str(gaussian_python))} {q(str(exporter_py))} \
      --gaussian_root {q(str(gaussian_root))} \
      --point_path {q(str(point_cloud_path))} \
      --motion_path {q(str(motion_path))} \
      --audio_path {q(str(audio_path))} \
      --out_video {q(str(out_video))} \
      --frames_dir {q(str(frames_dir))} \
      --audio_filter '' \
      --render_mode {q(request.render_mode)} \
      --fps {q(str(request.fps))} \
      --width {q(str(request.width))} \
      --height {q(str(request.height))} \
      --camera_json {q(str(camera_json))} \
      --ffmpeg {q(str(ffmpeg))}
    """
    command = (
        "apptainer exec --fakeroot --writable --nv "
        "-B /scratch:/scratch,/home/svu:/home/svu "
        f"{q(str(container_image))} bash -lc {q(inner_cmd)}"
    )

    export_key = f"{run_id}:{export_id}"
    record = {
        "run_id": run_id,
        "export_id": export_id,
        "status": "queued",
        "return_code": None,
        "output_video": None,
        "output_video_url": None,
        "error": None,
        "log_path": str(log_path),
        "camera_json": str(camera_json),
        "render_mode": request.render_mode,
        "width": request.width,
        "height": request.height,
        "fps": request.fps,
    }
    with exports_lock:
        viewer_exports[export_key] = record

    thread = threading.Thread(
        target=_run_viewer_export,
        args=(export_key, command, out_video, log_path),
        daemon=True,
    )
    thread.start()
    return JSONResponse(record)


@app.get("/api/jobs/{run_id}/viewer/export/{export_id}")
def get_viewer_export(run_id: str, export_id: str) -> JSONResponse:
    if not re.match(r"^[A-Za-z0-9_.-]+$", run_id) or not re.match(r"^[A-Za-z0-9_.-]+$", export_id):
        raise HTTPException(status_code=400, detail="Invalid id.")
    export_key = f"{run_id}:{export_id}"
    with exports_lock:
        record = viewer_exports.get(export_key)
    if not record:
        export_dir = OUTPUT_ROOT / run_id / "artifacts" / "custom_views" / export_id
        if not export_dir.exists():
            raise HTTPException(status_code=404, detail="Export not found.")
        out_video = export_dir / "custom_view.mp4"
        record = {
            "run_id": run_id,
            "export_id": export_id,
            "status": "done" if out_video.exists() else "unknown",
            "return_code": None,
            "output_video": str(out_video) if out_video.exists() else None,
            "output_video_url": _viewer_export_url(str(out_video)) if out_video.exists() else None,
            "error": None,
            "log_path": str(export_dir / "export.log"),
        }
    record = record.copy()
    log_path = Path(record.get("log_path", ""))
    record["log_tail"] = _tail(log_path, max_chars=8000) if log_path.exists() else ""
    return JSONResponse(record)


@app.get("/api/jobs")
def list_jobs() -> JSONResponse:
    items = []
    for path in sorted(OUTPUT_ROOT.glob("web_*"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
        if path.is_dir() and path.name != "web_uploads" and (path / "state.json").exists():
            payload = _job_payload(path.name)
            payload.pop("log_tail", None)
            payload.pop("state", None)
            items.append(payload)
    return JSONResponse({"jobs": items})


@app.post("/api/jobs/{run_id}/stop")
def stop_job(run_id: str) -> JSONResponse:
    with jobs_lock:
        job = jobs.get(run_id)
    if not job or job.get("process") is None:
        raise HTTPException(status_code=404, detail="Running job not found.")
    proc: subprocess.Popen[str] = job["process"]
    if proc.poll() is None:
        proc.terminate()
    return JSONResponse(_job_payload(run_id))
