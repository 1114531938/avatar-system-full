from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import shutil
import sqlite3
import subprocess
import threading
import hashlib
import ast
import base64
import struct
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles


ROOT = Path(os.environ.get("AVATAR_SYSTEM_ROOT", "/scratch/e1554543/avatar_system_full"))
WEB_ROOT = ROOT / "apps" / "web"
STATIC_ROOT = WEB_ROOT / "static"
OUTPUT_ROOT = ROOT / "runtime" / "outputs"
UPLOAD_ROOT = OUTPUT_ROOT / "web_uploads"
CACHE_ROOT = ROOT / "runtime" / "cache" / "pipeline"
TTS_PREVIEW_ROOT = OUTPUT_ROOT / "tts_previews"
BOOTH_ROOT = OUTPUT_ROOT / "booth"
BOOTH_UPLOAD_ROOT = BOOTH_ROOT / "uploads"
BOOTH_EXPORT_ROOT = BOOTH_ROOT / "exports"
BOOTH_DB = BOOTH_ROOT / "booth.sqlite3"
SCRIPT = ROOT / "scripts" / "avatar.sh"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
TTS_PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
BOOTH_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
BOOTH_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Avatar Web Studio")
app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")

jobs: dict[str, dict[str, Any]] = {}
viewer_exports: dict[str, dict[str, Any]] = {}
booth_exports: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()
exports_lock = threading.Lock()
settings_lock = threading.Lock()
db_lock = threading.Lock()
runtime_settings: dict[str, str] = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
    "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    "LLM_MODEL": os.environ.get("LLM_MODEL", "openai/gpt-oss-120b:free"),
}


def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(BOOTH_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_booth_db() -> None:
    with db_lock, _db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT NOT NULL UNIQUE,
              password_hash TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS auth_sessions (
              token TEXT PRIMARY KEY,
              user_id INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              expires_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS reset_tokens (
              token TEXT PRIMARY KEY,
              user_id INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              expires_at TEXT NOT NULL,
              used_at TEXT,
              FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS booth_sessions (
              id TEXT PRIMARY KEY,
              user_id INTEGER NOT NULL,
              title TEXT NOT NULL,
              background_id TEXT NOT NULL,
              background_url TEXT,
              export_status TEXT NOT NULL DEFAULT 'idle',
              export_video_url TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS booth_turns (
              id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              user_id INTEGER NOT NULL,
              run_id TEXT,
              status TEXT NOT NULL,
              input_wav TEXT,
              input_video_path TEXT,
              input_video_url TEXT,
              background_id TEXT,
              match_result TEXT,
              reply_text TEXT,
              reply_video_url TEXT,
              manifest_json TEXT,
              error TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(session_id) REFERENCES booth_sessions(id),
              FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )


_init_booth_db()


def _hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 260000)
    return f"pbkdf2_sha256${salt}${digest.hex()}"


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        scheme, salt, digest = password_hash.split("$", 2)
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    return secrets.compare_digest(_hash_password(password, salt), password_hash)


def _validate_username(username: str) -> str:
    username = username.strip()
    if not re.match(r"^[A-Za-z0-9_.@-]{3,64}$", username):
        raise HTTPException(status_code=400, detail="Username must be 3-64 characters using letters, numbers, _, ., @, or -.")
    return username


def _validate_password(password: str) -> str:
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    return password


def _user_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {"id": int(row["id"]), "username": row["username"], "created_at": row["created_at"]}


def _create_auth_session(user_id: int) -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    created_at = _now_iso()
    expires_at = (datetime.now() + timedelta(days=7)).replace(microsecond=0).isoformat()
    with db_lock, _db() as conn:
        conn.execute(
            "INSERT INTO auth_sessions(token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, created_at, expires_at),
        )
    return token, expires_at


def _get_current_user(session_token: str | None) -> dict[str, Any]:
    if not session_token:
        raise HTTPException(status_code=401, detail="Login required.")
    with db_lock, _db() as conn:
        row = conn.execute(
            """
            SELECT users.* FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token = ? AND auth_sessions.expires_at > ?
            """,
            (session_token, _now_iso()),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Login required.")
    return _user_payload(row)


def _require_user(request: Request) -> dict[str, Any]:
    return _get_current_user(request.cookies.get("session_token"))


def _set_session_cookie(response: JSONResponse, token: str, expires_at: str) -> None:
    expires = datetime.fromisoformat(expires_at)
    max_age = max(0, int((expires - datetime.now()).total_seconds()))
    response.set_cookie(
        "session_token",
        token,
        max_age=max_age,
        httponly=True,
        samesite="lax",
        secure=False,
    )


def _clear_session_cookie(response: JSONResponse) -> None:
    response.delete_cookie("session_token", httponly=True, samesite="lax")


class RuntimeSettings(BaseModel):
    openai_api_key: str = ""
    openai_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-oss-120b:free"


class AuthRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class ResetPasswordRequest(BaseModel):
    username: str


class ResetPasswordConfirmRequest(BaseModel):
    token: str
    new_password: str


class TTSPreviewRequest(BaseModel):
    speaker_id: str = "6224"
    text: str = "你好，我是你的情感数字人助手，很高兴今天和你见面。"
    style_prompt: str = "Warm, natural, conversational speech."


def _load_avatar_labels() -> dict[str, str]:
    labels: dict[str, str] = {
        "306": "GaussianAvatars demo",
        "1001": "demo1 custom subject",
    }

    map_path = (
        ROOT
        / "integrations"
        / "gaussian_avatar"
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
    media_root = ROOT / "integrations" / "gaussian_avatar" / "media"
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


def _speaker_wiki_rows() -> dict[str, dict[str, str]]:
    wiki_path = ROOT / "integrations" / "emotivoice" / "data" / "youdao" / "text" / "README.md"
    rows: dict[str, dict[str, str]] = {}
    if not wiki_path.exists():
        return rows

    pattern = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|")
    for line in wiki_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        speaker_id, name, gender, description = [part.strip() for part in match.groups()]
        if not speaker_id.isdigit():
            continue
        rows[speaker_id] = {
            "name": name,
            "gender": gender,
            "description": description,
        }
    return rows


def _list_tts_speakers() -> list[dict[str, str]]:
    speaker_path = ROOT / "integrations" / "emotivoice" / "data" / "youdao" / "text" / "speaker2"
    wiki = _speaker_wiki_rows()
    if not speaker_path.exists():
        return [{"id": "6224", "label": "6224 · dave k · M"}]

    speakers: list[dict[str, str]] = []
    for line in speaker_path.read_text(encoding="utf-8", errors="replace").splitlines():
        speaker_id = line.strip()
        if not speaker_id:
            continue
        meta = wiki.get(speaker_id, {})
        name = meta.get("name", "")
        gender = meta.get("gender", "")
        description = meta.get("description", "")
        label_parts = [speaker_id]
        if name:
            label_parts.append(name)
        if gender:
            label_parts.append(gender)
        if description:
            label_parts.append(description)
        speakers.append(
            {
                "id": speaker_id,
                "label": " · ".join(label_parts),
                "name": name,
                "gender": gender,
                "description": description,
            }
        )
    return speakers


def _booth_backgrounds() -> list[dict[str, str]]:
    return [
        {
            "id": "soft_studio",
            "label": "Soft Studio",
            "style": "calm",
            "css": "linear-gradient(135deg, #f8fafc 0%, #dbeafe 48%, #ccfbf1 100%)",
        },
        {
            "id": "midnight_lab",
            "label": "Midnight Lab",
            "style": "focused",
            "css": "linear-gradient(135deg, #101827 0%, #21435a 52%, #0f766e 100%)",
        },
        {
            "id": "paper_room",
            "label": "Paper Room",
            "style": "warm",
            "css": "linear-gradient(135deg, #fff7ed 0%, #e0f2fe 55%, #f8fafc 100%)",
        },
    ]


def _avatar_tags(avatar_id: str, label: str) -> dict[str, str]:
    numeric = int(avatar_id) if str(avatar_id).isdigit() else 0
    presentation = "feminine" if numeric % 2 else "neutral"
    if any(term in label.lower() for term in ["female", "woman", "165"]):
        presentation = "feminine"
    elif any(term in label.lower() for term in ["male", "man", "074", "306"]):
        presentation = "masculine"
    return {
        "presentation": presentation,
        "age_band": "adult",
        "style": "empathetic",
        "demo_label": label,
    }


def _speaker_tags(speaker: dict[str, str]) -> dict[str, str]:
    label = " ".join(str(v) for v in speaker.values()).lower()
    gender = str(speaker.get("gender", "")).lower()
    presentation = "neutral"
    if gender.startswith("f") or "female" in label or "woman" in label:
        presentation = "feminine"
    elif gender.startswith("m") or "male" in label or "man" in label:
        presentation = "masculine"
    timbre = "mid"
    if any(term in label for term in ["low", "deep", "bass"]):
        timbre = "low"
    elif any(term in label for term in ["high", "bright"]):
        timbre = "high"
    return {"presentation": presentation, "voice_timbre": timbre, "style": "warm"}


def _ffmpeg_bin() -> str:
    runtime_ffmpeg = ROOT / "runtime" / "cache" / "bin" / "ffmpeg"
    if runtime_ffmpeg.exists():
        return str(runtime_ffmpeg)
    ffmpeg = ROOT / "tools" / "ffmpeg-git-20240629-amd64-static" / "ffmpeg"
    return str(ffmpeg) if ffmpeg.exists() else "ffmpeg"


def _ffprobe_bin() -> str:
    runtime_ffprobe = ROOT / "runtime" / "cache" / "bin" / "ffprobe"
    if runtime_ffprobe.exists():
        return str(runtime_ffprobe)
    ffprobe = ROOT / "tools" / "ffmpeg-git-20240629-amd64-static" / "ffprobe"
    return str(ffprobe) if ffprobe.exists() else "ffprobe"


def _extract_video_frames(video_path: Path, frames_dir: Path, max_frames: int = 3) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = frames_dir / "frame_%02d.jpg"
    command = [
        _ffmpeg_bin(),
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "fps=1,scale=512:-1",
        "-frames:v",
        str(max_frames),
        str(frame_pattern),
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return sorted(frames_dir.glob("frame_*.jpg"))[:max_frames]


def _call_vision_model(frame_paths: list[Path]) -> dict[str, Any]:
    vision_model = os.environ.get("VISION_MODEL", "").strip()
    if not vision_model or not frame_paths:
        return {"enabled": bool(vision_model), "status": "not_called"}
    with settings_lock:
        api_key = runtime_settings.get("OPENAI_API_KEY", "")
        base_url = runtime_settings.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    if not api_key:
        return {"enabled": True, "status": "skipped", "reason": "OPENAI_API_KEY is not configured."}

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Return strict JSON with non-sensitive demo tags for avatar matching. "
                "Do not infer race, ethnicity, identity, health, or other sensitive traits. "
                "Allowed keys: presentation, apparent_age_band, lighting, mood, confidence, notes."
            ),
        }
    ]
    for frame_path in frame_paths:
        encoded = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )
    payload = json.dumps(
        {
            "model": vision_model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        text = result["choices"][0]["message"]["content"]
        text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
        tags = json.loads(text)
        if not isinstance(tags, dict):
            tags = {"notes": str(tags)}
        return {"enabled": True, "status": "ok", "model": vision_model, "tags": tags}
    except Exception as exc:
        return {"enabled": True, "status": "failed", "model": vision_model, "reason": str(exc)}


def _infer_local_match(input_wav: Path | None, input_video: Path | None, frame_paths: list[Path] | None = None) -> dict[str, Any]:
    avatars = _list_available_avatars()
    speakers = _list_tts_speakers()
    vision = _call_vision_model(frame_paths or [])
    avatar = avatars[0] if avatars else {"id": "306", "label": "Avatar 306"}
    speaker = next((item for item in speakers if str(item.get("id")) == "6224"), speakers[0] if speakers else {"id": "6224"})
    presentation_hint = ((vision.get("tags") or {}).get("presentation") or "").lower() if isinstance(vision.get("tags"), dict) else ""
    if presentation_hint:
        for candidate in avatars:
            tags = _avatar_tags(str(candidate["id"]), str(candidate.get("label", candidate["id"])))
            if tags.get("presentation") == presentation_hint:
                avatar = candidate
                break
        for candidate in speakers:
            tags = _speaker_tags(candidate)
            if tags.get("presentation") == presentation_hint:
                speaker = candidate
                break
    avatar_tag = _avatar_tags(str(avatar["id"]), str(avatar.get("label", avatar["id"])))
    speaker_tag = _speaker_tags(speaker)
    return {
        "avatar_id": str(avatar["id"]),
        "tts_speaker_id": str(speaker["id"]),
        "strategy": "local_explainable_fallback",
        "vision": vision,
        "signals": {
            "audio_file": input_wav.name if input_wav else None,
            "video_file": input_video.name if input_video else None,
            "frame_count": len(frame_paths or []),
        },
        "avatar_tags": avatar_tag,
        "voice_tags": speaker_tag,
        "reason": (
            "Selected from available demo assets using non-sensitive presentation/style tags. "
            "No race or ethnicity classification is performed."
        ),
    }


def _emotivoice_phonemes(text: str) -> str:
    root = ROOT / "integrations" / "emotivoice"
    py = root / ".EmotiVoice" / "bin" / "python"
    frontend = root / "frontend.py"
    if not os.path.lexists(py):
        raise FileNotFoundError(f"EmotiVoice python not found: {py}")
    if not frontend.exists():
        raise FileNotFoundError(f"EmotiVoice frontend.py not found: {frontend}")

    temp_input = TTS_PREVIEW_ROOT / f"preview_input_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.txt"
    temp_input.write_text(text.strip() + "\n", encoding="utf-8")
    container_image = ROOT / "runtime" / "containers" / "gaussianav_jammy"
    inner_cmd = (
        f"cd {shlex.quote(str(root))} && "
        f"{shlex.quote(str(py))} {shlex.quote(str(frontend))} {shlex.quote(str(temp_input))}"
    )
    apptainer_flags = os.environ.get("APPTAINER_FLAGS", "--nv")
    command = (
        f"apptainer exec {apptainer_flags} "
        "-B /scratch:/scratch,/home/svu:/home/svu "
        f"{shlex.quote(str(container_image))} bash -lc "
        f"{shlex.quote(inner_cmd)}"
    )
    try:
        proc = subprocess.run(
            ["bash", "-lc", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=True,
        )
    finally:
        temp_input.unlink(missing_ok=True)

    phonemes = " ".join(proc.stdout.strip().split())
    if not phonemes:
        raise RuntimeError(f"EmotiVoice frontend returned empty phonemes: {proc.stderr}")
    return phonemes


def _tts_worker_synthesize(test_file: Path, output_wav: Path, timeout: float = 120) -> dict[str, Any]:
    url = os.environ.get("TTS_WORKER_URL", "http://127.0.0.1:8788").rstrip("/")
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        if not health.get("ok"):
            raise RuntimeError(f"TTS worker is unhealthy: {health}")
    except Exception as exc:
        raise RuntimeError(f"TTS worker is not available at {url}. Start it with scripts/avatar.sh worker tts.") from exc

    payload = json.dumps({"test_file": str(test_file), "output_wav": str(output_wav)}).encode("utf-8")
    request = urllib.request.Request(
        f"{url}/synthesize",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    if not result.get("ok"):
        raise RuntimeError(f"TTS preview synthesis failed: {result}")
    if not output_wav.exists():
        raise FileNotFoundError(f"TTS preview output was not created: {output_wav}")
    return result


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


def _prune_output_runs(keep: int = 5, preserve_uploads: set[Path] | None = None) -> None:
    keep = max(1, int(keep))
    preserved = {path.resolve() for path in (preserve_uploads or set())}
    run_dirs = [path for path in OUTPUT_ROOT.iterdir() if _is_web_run_dir(path)]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    kept_names = {path.name for path in run_dirs[:keep]}
    for old_dir in run_dirs[keep:]:
        shutil.rmtree(old_dir, ignore_errors=True)
        upload_wav = UPLOAD_ROOT / f"{old_dir.name}.wav"
        if upload_wav.resolve() in preserved:
            continue
        upload_wav.unlink(missing_ok=True)
    for upload_wav in UPLOAD_ROOT.glob("web_*.wav"):
        if upload_wav.resolve() in preserved:
            continue
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
    tts_speaker_id: str,
    settings: dict[str, str],
) -> str:
    cache_obj = {
        "schema": 2,
        "audio_sha256": audio_sha256,
        "avatar_id": str(avatar_id),
        "no_llm": bool(no_llm),
        "no_video_export": bool(no_video_export),
        "prepare_only": bool(prepare_only),
        "tts_speaker_id": str(tts_speaker_id),
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
    tts_speaker_id: str,
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
        "tts_speaker_id": str(tts_speaker_id),
        "base_name": run_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "log_dir": str(logs_dir),
        "prepare_only": bool(prepare_only),
        "launch_viewer": False,
        "current_stage": "done",
        "finished_stages": [
            "cache_restore",
            "input_agent",
            "perception",
            "task1",
            "dialogue_agent",
            "plan_agent",
            "emotivoice_prepare",
            "render_agent",
            "emotivoice_tts",
            "deeptalk",
            "flame_merge",
            "viewer",
            "artifact_export",
            "embodiment_agent",
        ],
        "failed_stage": None,
        "error": None,
        "perception_json": cached_manifest.get("perception_json"),
        "task1_input_json": cached_manifest.get("task1_input_json"),
        "perception_result_json": cached_manifest.get("perception_result_json"),
        "task1_reply_json": cached_manifest.get("task1_reply_json"),
        "plan_json": cached_manifest.get("plan_json") or cached_manifest.get("reply_plan_json"),
        "reply_plan_json": cached_manifest.get("reply_plan_json") or cached_manifest.get("plan_json"),
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
        "agent_pipeline_version": cached_manifest.get("agent_pipeline_version", 1),
        "run_id": run_id,
        "input_wav": str(input_wav),
        "input_video": cached_manifest.get("input_video"),
        "video_frames_dir": cached_manifest.get("video_frames_dir"),
        "avatar_id": str(avatar_id),
        "perception_json": state["perception_json"],
        "task1_input_json": state["task1_input_json"],
        "perception_result_json": state["perception_result_json"],
        "task1_reply_json": state["task1_reply_json"],
        "plan_json": state["plan_json"],
        "reply_plan_json": state["reply_plan_json"],
        "selected_avatar_id": cached_manifest.get("selected_avatar_id") or str(avatar_id),
        "selected_tts_speaker_id": cached_manifest.get("selected_tts_speaker_id") or str(tts_speaker_id),
        "background": cached_manifest.get("background"),
        "session_id": cached_manifest.get("session_id"),
        "turn_id": cached_manifest.get("turn_id"),
        "reply_text": state["reply_text"],
        "reply_style": state["reply_style"],
        "tts_speaker_id": str(tts_speaker_id),
        "emotivoice_txt": state["emotivoice_txt"],
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
    with jobs_lock:
        booth_context = jobs.get(run_id, {}).get("booth")
    if booth_context:
        _finalize_booth_turn(run_id)


@app.get("/")
def index() -> FileResponse:
    if os.environ.get("BOOTH_DEFAULT_ROUTE") == "1":
        return FileResponse(STATIC_ROOT / "booth.html")
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/booth")
def booth() -> FileResponse:
    return FileResponse(STATIC_ROOT / "booth.html")


@app.get("/studio")
def studio() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.post("/api/auth/register")
def auth_register(request: AuthRequest) -> JSONResponse:
    username = _validate_username(request.username)
    password = _validate_password(request.password)
    created_at = _now_iso()
    try:
        with db_lock, _db() as conn:
            cur = conn.execute(
                "INSERT INTO users(username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, _hash_password(password), created_at),
            )
            user_id = int(cur.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Username already exists.") from exc
    token, expires_at = _create_auth_session(user_id)
    payload = {"ok": True, "user": {"id": user_id, "username": username, "created_at": created_at}}
    response = JSONResponse(payload)
    _set_session_cookie(response, token, expires_at)
    return response


@app.post("/api/auth/login")
def auth_login(request: AuthRequest) -> JSONResponse:
    username = _validate_username(request.username)
    with db_lock, _db() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not row or not _verify_password(request.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token, expires_at = _create_auth_session(int(row["id"]))
    response = JSONResponse({"ok": True, "user": _user_payload(row)})
    _set_session_cookie(response, token, expires_at)
    return response


@app.post("/api/auth/logout")
def auth_logout(request: Request) -> JSONResponse:
    token = request.cookies.get("session_token")
    if token:
        with db_lock, _db() as conn:
            conn.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))
    response = JSONResponse({"ok": True})
    _clear_session_cookie(response)
    return response


@app.get("/api/auth/me")
def auth_me(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True, "user": _require_user(request)})


@app.post("/api/auth/change_password")
def auth_change_password(request: Request, payload: ChangePasswordRequest) -> JSONResponse:
    user = _require_user(request)
    new_password = _validate_password(payload.new_password)
    with db_lock, _db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
        if not row or not _verify_password(payload.current_password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Current password is incorrect.")
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (_hash_password(new_password), user["id"]))
    return JSONResponse({"ok": True})


@app.post("/api/auth/reset_password/request")
def auth_reset_request(payload: ResetPasswordRequest) -> JSONResponse:
    username = _validate_username(payload.username)
    token = secrets.token_urlsafe(32)
    created_at = _now_iso()
    expires_at = (datetime.now() + timedelta(hours=1)).replace(microsecond=0).isoformat()
    with db_lock, _db() as conn:
        row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if row:
            conn.execute(
                "INSERT INTO reset_tokens(token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token, int(row["id"]), created_at, expires_at),
            )
            print(f"[booth_auth] password reset token for {username}: {token}", flush=True)
            return JSONResponse({"ok": True, "reset_token": token, "expires_at": expires_at})
    return JSONResponse({"ok": True, "reset_token": None})


@app.post("/api/auth/reset_password/confirm")
def auth_reset_confirm(payload: ResetPasswordConfirmRequest) -> JSONResponse:
    new_password = _validate_password(payload.new_password)
    with db_lock, _db() as conn:
        row = conn.execute(
            "SELECT * FROM reset_tokens WHERE token = ? AND used_at IS NULL AND expires_at > ?",
            (payload.token, _now_iso()),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=400, detail="Reset token is invalid or expired.")
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (_hash_password(new_password), int(row["user_id"])))
        conn.execute("UPDATE reset_tokens SET used_at = ? WHERE token = ?", (_now_iso(), payload.token))
        conn.execute("DELETE FROM auth_sessions WHERE user_id = ?", (int(row["user_id"]),))
    return JSONResponse({"ok": True})


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


@app.get("/api/tts_speakers")
def get_tts_speakers() -> JSONResponse:
    speakers = _list_tts_speakers()
    return JSONResponse({"speakers": speakers, "default_speaker_id": "6224"})


@app.post("/api/tts_preview")
def create_tts_preview(request: TTSPreviewRequest) -> JSONResponse:
    speaker_id = request.speaker_id.strip() or "6224"
    text = request.text.strip() or "你好，我是你的情感数字人助手，很高兴今天和你见面。"
    style_prompt = request.style_prompt.strip() or "Warm, natural, conversational speech."

    valid_speaker_ids = {speaker["id"] for speaker in _list_tts_speakers()}
    if valid_speaker_ids and speaker_id not in valid_speaker_ids:
        raise HTTPException(status_code=400, detail=f"Unknown EmotiVoice speaker id: {speaker_id}")

    cache_obj = {"speaker_id": speaker_id, "text": text, "style_prompt": style_prompt, "schema": 1}
    preview_key = hashlib.sha256(
        json.dumps(cache_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    preview_dir = TTS_PREVIEW_ROOT / preview_key
    preview_dir.mkdir(parents=True, exist_ok=True)
    test_file = preview_dir / "input.txt"
    output_wav = preview_dir / "preview.wav"

    if not output_wav.exists():
        try:
            phonemes = _emotivoice_phonemes(text)
            test_file.write_text(
                f"{speaker_id}|{style_prompt}|<sos/eos> {phonemes} <sos/eos>|{text}\n",
                encoding="utf-8",
            )
            _tts_worker_synthesize(test_file, output_wav)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(
        {
            "ok": True,
            "speaker_id": speaker_id,
            "text": text,
            "audio_url": f"/outputs/tts_previews/{preview_key}/preview.wav",
            "input_txt": str(test_file),
            "output_wav": str(output_wav),
        }
    )


def _safe_upload_suffix(filename: str, default: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if not suffix or len(suffix) > 12 or not re.match(r"^\.[A-Za-z0-9]+$", suffix):
        return default
    return suffix


def _start_pipeline_job(
    *,
    input_wav: Path,
    source_filename: str,
    avatar_id: str,
    tts_speaker_id: str,
    no_llm: bool,
    no_video_export: bool,
    prepare_only: bool,
    booth_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
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

    tts_speaker_id = str(tts_speaker_id).strip() or "6224"
    valid_speaker_ids = {speaker["id"] for speaker in _list_tts_speakers()}
    if valid_speaker_ids and tts_speaker_id not in valid_speaker_ids:
        raise HTTPException(status_code=400, detail=f"Unknown EmotiVoice speaker id: {tts_speaker_id}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _safe_slug(Path(source_filename).stem)
    run_prefix = "booth" if booth_context else "web"
    run_id = f"{run_prefix}_{stem}_{stamp}"
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _prune_output_runs(keep=5, preserve_uploads={input_wav})

    audio_sha256 = _sha256_file(input_wav)
    cache_key = _cache_key(
        audio_sha256=audio_sha256,
        avatar_id=str(avatar_id),
        no_llm=bool(no_llm),
        no_video_export=bool(no_video_export),
        prepare_only=bool(prepare_only),
        tts_speaker_id=tts_speaker_id,
        settings=active_settings,
    )

    if _cache_manifest_path(cache_key).exists():
        restored = _restore_cached_run(
            cache_key=cache_key,
            run_id=run_id,
            run_dir=run_dir,
            avatar_id=str(avatar_id),
            tts_speaker_id=tts_speaker_id,
            input_wav=input_wav,
            prepare_only=bool(prepare_only),
        )
        if restored:
            with jobs_lock:
                jobs[run_id] = {
                    "process": None,
                    "created_at": stamp,
                    "input_wav": str(input_wav),
                    "avatar_id": str(avatar_id),
                    "tts_speaker_id": tts_speaker_id,
                    "return_code": 0,
                    "cache_key": cache_key,
                    "cache_hit": True,
                    "booth": booth_context,
                }
            if booth_context:
                _finalize_booth_turn(run_id)
            return _job_payload(run_id)

    web_log = run_dir / "web_stdout.log"
    cmd = [
        "bash",
        str(SCRIPT),
        "agent",
        str(input_wav),
        str(avatar_id),
        "--run_id",
        run_id,
        "--tts_speaker_id",
        tts_speaker_id,
    ]
    if prepare_only:
        cmd.append("--prepare_only")
    if no_video_export:
        cmd.append("--no_video_export")
    if no_llm:
        cmd.append("--no_llm")
    if booth_context:
        if booth_context.get("input_video_path"):
            cmd.extend(["--input_video", str(booth_context["input_video_path"])])
        if booth_context.get("background_id"):
            cmd.extend(["--background", str(booth_context["background_id"])])
        if booth_context.get("session_id"):
            cmd.extend(["--session_id", str(booth_context["session_id"])])
        if booth_context.get("turn_id"):
            cmd.extend(["--turn_id", str(booth_context["turn_id"])])

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
        if booth_context:
            log.write("[booth] " + json.dumps(booth_context, ensure_ascii=False) + "\n\n")

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
            "input_wav": str(input_wav),
            "avatar_id": str(avatar_id),
            "tts_speaker_id": tts_speaker_id,
            "return_code": None,
            "cache_key": cache_key,
            "cache_hit": False,
            "booth": booth_context,
        }
    thread = threading.Thread(target=_watch_process, args=(run_id, proc, web_log), daemon=True)
    thread.start()
    return _job_payload(run_id)


@app.post("/api/jobs")
async def create_job(
    audio: UploadFile = File(...),
    avatar_id: str = Form("306"),
    tts_speaker_id: str = Form("6224"),
    no_llm: bool = Form(False),
    no_video_export: bool = Form(False),
    prepare_only: bool = Form(False),
) -> JSONResponse:
    filename = audio.filename or "recording.wav"
    if not filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Please upload or record a .wav file.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _safe_slug(Path(filename).stem)
    upload_path = UPLOAD_ROOT / f"web_{stem}_{stamp}.wav"
    with upload_path.open("wb") as f:
        shutil.copyfileobj(audio.file, f)
    return JSONResponse(
        _start_pipeline_job(
            input_wav=upload_path,
            source_filename=filename,
            avatar_id=str(avatar_id),
            tts_speaker_id=str(tts_speaker_id),
            no_llm=bool(no_llm),
            no_video_export=bool(no_video_export),
            prepare_only=bool(prepare_only),
        )
    )


def _parse_json_field(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _booth_turn_payload(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    payload["match_result"] = _parse_json_field(payload.get("match_result"), {})
    payload["manifest"] = _parse_json_field(payload.pop("manifest_json", None), {})
    return payload


def _booth_session_payload(row: sqlite3.Row, include_turns: bool = False) -> dict[str, Any]:
    payload = dict(row)
    if include_turns:
        with db_lock, _db() as conn:
            turns = conn.execute(
                "SELECT * FROM booth_turns WHERE session_id = ? ORDER BY created_at ASC",
                (payload["id"],),
            ).fetchall()
        payload["turns"] = [_booth_turn_payload(turn) for turn in turns]
    return payload


def _require_booth_session(session_id: str, user_id: int) -> sqlite3.Row:
    with db_lock, _db() as conn:
        row = conn.execute(
            "SELECT * FROM booth_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Booth session not found.")
    return row


def _booth_background_color(background_id: str) -> str:
    return {
        "soft_studio": "0xDCEDEA",
        "midnight_lab": "0x101827",
        "paper_room": "0xF7EFE4",
    }.get(background_id, "0xDCEDEA")


def _media_has_audio(path: Path) -> bool:
    command = [
        _ffprobe_bin(),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=20)
    except Exception:
        return False
    return "audio" in (proc.stdout or "")


def _extract_wav_from_video(video_path: Path, wav_path: Path) -> None:
    command = [
        _ffmpeg_bin(),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(wav_path),
    ]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    if proc.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size == 0:
        raise HTTPException(
            status_code=400,
            detail="Could not extract microphone audio from the uploaded user video.",
        )


def _normalize_booth_segment(source: Path, output: Path, background_id: str, background_path: Path | None) -> tuple[bool, str]:
    source = source.resolve()
    output = output.resolve()
    has_audio = _media_has_audio(source)
    if background_path and background_path.exists():
        command = [
            _ffmpeg_bin(),
            "-y",
            "-loop",
            "1",
            "-i",
            str(background_path),
            "-i",
            str(source),
        ]
        if not has_audio:
            command.extend(["-f", "lavfi", "-t", "0.1", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"])
        command.extend([
            "-filter_complex",
            (
                "[0:v]scale=1280:720:force_original_aspect_ratio=increase,"
                "crop=1280:720,setsar=1[bg];"
                "[1:v]scale=1120:630:force_original_aspect_ratio=decrease,setsar=1[fg];"
                "[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
            ),
            "-map",
            "[v]",
        ])
        if has_audio:
            command.extend(["-map", "1:a:0", "-c:a", "aac", "-b:a", "160k"])
        else:
            command.extend(["-map", "2:a:0", "-c:a", "aac"])
    else:
        color = _booth_background_color(background_id)
        command = [
            _ffmpeg_bin(),
            "-y",
            "-i",
            str(source),
        ]
        if not has_audio:
            command.extend(["-f", "lavfi", "-t", "0.1", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"])
        command.extend([
            "-filter_complex",
            (
                f"color=c={color}:s=1280x720[bg];"
                "[0:v]scale=1120:630:force_original_aspect_ratio=decrease,setsar=1[fg];"
                "[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
            ),
            "-map",
            "[v]",
        ])
        if has_audio:
            command.extend(["-map", "0:a:0", "-c:a", "aac", "-b:a", "160k"])
        else:
            command.extend(["-map", "1:a:0", "-c:a", "aac"])
    command.extend(["-c:v", "libx264", "-preset", "veryfast", "-r", "25", "-shortest", str(output)])
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = "$ " + " ".join(command) + "\n\n" + (proc.stdout or "")
    return proc.returncode == 0 and output.exists(), log


def _finalize_booth_turn(run_id: str) -> None:
    with jobs_lock:
        booth_context = jobs.get(run_id, {}).get("booth") or {}
    turn_id = booth_context.get("turn_id")
    session_id = booth_context.get("session_id")
    if not turn_id or not session_id:
        return

    payload = _job_payload(run_id)
    manifest = payload.get("manifest") or {}
    urls = payload.get("artifact_urls") or {}
    status = payload.get("status", "unknown")
    error = None
    if status == "failed":
        state = payload.get("state") or {}
        error = manifest.get("error") or state.get("error") or payload.get("log_tail", "")[-1200:]
    reply_video_url = urls.get("output_video")
    reply_text = manifest.get("reply_text")
    now = _now_iso()
    with db_lock, _db() as conn:
        conn.execute(
            """
            UPDATE booth_turns
            SET status = ?, reply_text = ?, reply_video_url = ?, manifest_json = ?, error = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, reply_text, reply_video_url, json.dumps(manifest, ensure_ascii=False), error, now, turn_id),
        )
        conn.execute("UPDATE booth_sessions SET updated_at = ? WHERE id = ?", (now, session_id))


@app.get("/api/booth/config")
def booth_config(request: Request) -> JSONResponse:
    user = None
    try:
        user = _require_user(request)
    except HTTPException:
        user = None
    return JSONResponse(
        {
            "ok": True,
            "user": user,
            "backgrounds": _booth_backgrounds(),
            "avatars": [
                {**avatar, "tags": _avatar_tags(str(avatar["id"]), str(avatar.get("label", avatar["id"])))}
                for avatar in _list_available_avatars()
            ],
            "speakers": [
                {**speaker, "tags": _speaker_tags(speaker)}
                for speaker in _list_tts_speakers()
            ],
            "vision_model_configured": bool(os.environ.get("VISION_MODEL")),
        }
    )


@app.post("/api/booth/sessions")
def create_booth_session(request: Request, title: str = Form("Emotional Avatar Booth"), background_id: str = Form("soft_studio")) -> JSONResponse:
    user = _require_user(request)
    session_id = secrets.token_urlsafe(12)
    now = _now_iso()
    with db_lock, _db() as conn:
        conn.execute(
            """
            INSERT INTO booth_sessions(id, user_id, title, background_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, user["id"], title.strip() or "Emotional Avatar Booth", background_id.strip() or "soft_studio", now, now),
        )
        row = conn.execute("SELECT * FROM booth_sessions WHERE id = ?", (session_id,)).fetchone()
    return JSONResponse({"ok": True, "session": _booth_session_payload(row, include_turns=True)})


@app.get("/api/booth/sessions")
def list_booth_sessions(request: Request) -> JSONResponse:
    user = _require_user(request)
    with db_lock, _db() as conn:
        rows = conn.execute(
            "SELECT * FROM booth_sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT 50",
            (user["id"],),
        ).fetchall()
    return JSONResponse({"ok": True, "sessions": [_booth_session_payload(row) for row in rows]})


@app.get("/api/booth/sessions/{session_id}")
def get_booth_session(request: Request, session_id: str) -> JSONResponse:
    user = _require_user(request)
    row = _require_booth_session(session_id, int(user["id"]))
    with db_lock, _db() as conn:
        running = conn.execute(
            "SELECT run_id FROM booth_turns WHERE session_id = ? AND status IN ('running', 'unknown')",
            (session_id,),
        ).fetchall()
    for item in running:
        if item["run_id"]:
            job_payload = _job_payload(item["run_id"])
            if job_payload.get("status") in {"done", "failed"}:
                _finalize_booth_turn(item["run_id"])
    with db_lock, _db() as conn:
        fresh = conn.execute("SELECT * FROM booth_sessions WHERE id = ?", (session_id,)).fetchone()
    return JSONResponse({"ok": True, "session": _booth_session_payload(fresh, include_turns=True)})


@app.post("/api/booth/sessions/{session_id}/turns")
async def create_booth_turn(
    request: Request,
    session_id: str,
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    background_image: Optional[UploadFile] = File(None),
    background_id: str = Form("soft_studio"),
    no_llm: bool = Form(False),
) -> JSONResponse:
    user = _require_user(request)
    _require_booth_session(session_id, int(user["id"]))
    audio_filename = audio.filename if audio and audio.filename else ""
    video_filename = video.filename if video and video.filename else ""
    if not audio_filename and not video_filename:
        raise HTTPException(status_code=400, detail="Record a video or audio turn before sending.")
    if audio_filename and not audio_filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Booth audio must be a .wav file.")

    turn_id = secrets.token_urlsafe(12)
    turn_dir = BOOTH_UPLOAD_ROOT / session_id / turn_id
    turn_dir.mkdir(parents=True, exist_ok=True)

    input_video_path: Path | None = None
    input_video_url: str | None = None
    if video_filename:
        suffix = _safe_upload_suffix(video_filename, ".webm")
        if suffix not in {".webm", ".mp4", ".mov", ".m4v"}:
            raise HTTPException(status_code=400, detail="Booth video must be webm, mp4, mov, or m4v.")
        input_video_path = turn_dir / f"input_video{suffix}"
        with input_video_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)
        input_video_url = _outputs_url(str(input_video_path))

    input_wav = turn_dir / "input.wav"
    if audio_filename and audio:
        with input_wav.open("wb") as f:
            shutil.copyfileobj(audio.file, f)
    elif input_video_path:
        _extract_wav_from_video(input_video_path, input_wav)

    if not input_wav.exists() or input_wav.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="No usable microphone audio was captured.")
    frame_paths = _extract_video_frames(input_video_path, turn_dir / "video_frames") if input_video_path else []

    background_url: str | None = None
    if background_image and background_image.filename:
        suffix = _safe_upload_suffix(background_image.filename, ".jpg")
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise HTTPException(status_code=400, detail="Background image must be jpg, png, or webp.")
        background_path = turn_dir / f"background{suffix}"
        with background_path.open("wb") as f:
            shutil.copyfileobj(background_image.file, f)
        background_url = _outputs_url(str(background_path))

    match = _infer_local_match(input_wav, input_video_path, frame_paths)
    avatar_id = str(match.get("avatar_id") or "306")
    tts_speaker_id = str(match.get("tts_speaker_id") or "6224")
    with settings_lock:
        no_llm_effective = bool(no_llm) or not bool(runtime_settings.get("OPENAI_API_KEY"))

    now = _now_iso()
    with db_lock, _db() as conn:
        conn.execute(
            """
            INSERT INTO booth_turns(
              id, session_id, user_id, status, input_wav, input_video_path, input_video_url,
              background_id, match_result, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn_id,
                session_id,
                user["id"],
                "queued",
                str(input_wav),
                str(input_video_path) if input_video_path else None,
                input_video_url,
                background_id,
                json.dumps(match, ensure_ascii=False),
                now,
                now,
            ),
        )
        conn.execute(
            "UPDATE booth_sessions SET background_id = ?, background_url = COALESCE(?, background_url), updated_at = ? WHERE id = ?",
            (background_id, background_url, now, session_id),
        )

    if os.environ.get("BOOTH_INPUT_TEST_ONLY") == "1":
        with db_lock, _db() as conn:
            conn.execute(
                "UPDATE booth_turns SET status = ?, updated_at = ? WHERE id = ?",
                ("captured", _now_iso(), turn_id),
            )
            row = conn.execute("SELECT * FROM booth_turns WHERE id = ?", (turn_id,)).fetchone()
        return JSONResponse(
            {
                "ok": True,
                "mode": "input_test_only",
                "turn": _booth_turn_payload(row),
                "job": {
                    "status": "captured",
                    "input_wav": str(input_wav),
                    "input_video_path": str(input_video_path) if input_video_path else None,
                    "input_video_url": input_video_url,
                },
            }
        )

    try:
        job_payload = _start_pipeline_job(
            input_wav=input_wav,
            source_filename=audio_filename or video_filename or "booth_turn.wav",
            avatar_id=avatar_id,
            tts_speaker_id=tts_speaker_id,
            no_llm=no_llm_effective,
            no_video_export=False,
            prepare_only=False,
            booth_context={
                "session_id": session_id,
                "turn_id": turn_id,
                "input_video_path": str(input_video_path) if input_video_path else None,
                "background_id": background_id,
                "background_url": background_url,
                "match_result": match,
            },
        )
    except Exception as exc:
        with db_lock, _db() as conn:
            conn.execute(
                "UPDATE booth_turns SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                ("failed", str(exc), _now_iso(), turn_id),
            )
        raise

    with db_lock, _db() as conn:
        conn.execute(
            "UPDATE booth_turns SET status = ?, run_id = ?, updated_at = ? WHERE id = ?",
            (job_payload.get("status", "running"), job_payload.get("run_id"), _now_iso(), turn_id),
        )
        row = conn.execute("SELECT * FROM booth_turns WHERE id = ?", (turn_id,)).fetchone()
    return JSONResponse({"ok": True, "turn": _booth_turn_payload(row), "job": job_payload})


def _run_booth_export(session_id: str, user_id: int, export_id: str) -> None:
    export_dir = BOOTH_EXPORT_ROOT / session_id / export_id
    export_dir.mkdir(parents=True, exist_ok=True)
    list_path = export_dir / "inputs.txt"
    out_video = export_dir / "conversation.mp4"
    log_path = export_dir / "export.log"
    with db_lock, _db() as conn:
        session = conn.execute(
            "SELECT * FROM booth_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        turns = conn.execute(
            "SELECT * FROM booth_turns WHERE session_id = ? AND user_id = ? ORDER BY created_at ASC",
            (session_id, user_id),
        ).fetchall()

    video_paths: list[Path] = []
    for turn in turns:
        if turn["input_video_path"] and Path(turn["input_video_path"]).exists():
            video_paths.append(Path(turn["input_video_path"]))
        manifest = _parse_json_field(turn["manifest_json"], {})
        reply_path = manifest.get("output_video")
        if reply_path and Path(reply_path).exists():
            video_paths.append(Path(reply_path))

    with db_lock, _db() as conn:
        conn.execute("UPDATE booth_sessions SET export_status = ?, updated_at = ? WHERE id = ?", ("running", _now_iso(), session_id))

    if not video_paths:
        with db_lock, _db() as conn:
            conn.execute("UPDATE booth_sessions SET export_status = ?, updated_at = ? WHERE id = ?", ("failed", _now_iso(), session_id))
        return

    with list_path.open("w", encoding="utf-8") as f:
        for path in video_paths:
            f.write(f"file '{str(path).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n")

    normalized: list[Path] = []
    log_chunks: list[str] = []
    background_path = None
    if session and session["background_url"]:
        candidate = OUTPUT_ROOT / str(session["background_url"]).removeprefix("/outputs/")
        if candidate.exists():
            background_path = candidate
    background_id = str(session["background_id"] if session else "soft_studio")
    for index, source in enumerate(video_paths):
        segment = export_dir / f"segment_{index:03d}.mp4"
        ok, log_text = _normalize_booth_segment(source, segment, background_id, background_path)
        log_chunks.append(log_text)
        if ok:
            normalized.append(segment)

    if not normalized:
        status = "failed"
        log_path.write_text("\n\n".join(log_chunks) or "No exportable video segments.", encoding="utf-8")
        url = None
        with db_lock, _db() as conn:
            conn.execute(
                "UPDATE booth_sessions SET export_status = ?, export_video_url = ?, updated_at = ? WHERE id = ?",
                (status, url, _now_iso(), session_id),
            )
        return

    with list_path.open("w", encoding="utf-8") as f:
        for path in normalized:
            safe_path = str(path).replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    command = [
        _ffmpeg_bin(),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(out_video),
    ]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path.write_text("\n\n".join(log_chunks + ["$ " + " ".join(command), proc.stdout or ""]), encoding="utf-8")
    status = "done" if proc.returncode == 0 and out_video.exists() else "failed"
    url = _outputs_url(str(out_video)) if status == "done" else None
    with db_lock, _db() as conn:
        conn.execute(
            "UPDATE booth_sessions SET export_status = ?, export_video_url = ?, updated_at = ? WHERE id = ?",
            (status, url, _now_iso(), session_id),
        )
    with exports_lock:
        booth_exports[f"{session_id}:{export_id}"] = {
            "session_id": session_id,
            "export_id": export_id,
            "status": status,
            "output_video_url": url,
            "log_path": str(log_path),
        }


@app.post("/api/booth/sessions/{session_id}/export")
def export_booth_session(request: Request, session_id: str) -> JSONResponse:
    user = _require_user(request)
    _require_booth_session(session_id, int(user["id"]))
    export_id = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with exports_lock:
        booth_exports[f"{session_id}:{export_id}"] = {
            "session_id": session_id,
            "export_id": export_id,
            "status": "queued",
            "output_video_url": None,
        }
    thread = threading.Thread(target=_run_booth_export, args=(session_id, int(user["id"]), export_id), daemon=True)
    thread.start()
    return JSONResponse({"ok": True, "export_id": export_id, "status": "queued"})


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
    gaussian_root = ROOT / "integrations" / "gaussian_avatar"
    gaussian_python = gaussian_root / ".GSavatar_glibc" / "bin" / "python"
    exporter_py = ROOT / "src" / "avatar_system" / "export_gaussian_video.py"
    container_image = ROOT / "runtime" / "containers" / "gaussianav_jammy"
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
    apptainer_flags = os.environ.get("APPTAINER_FLAGS", "--nv")
    command = (
        f"apptainer exec {apptainer_flags} "
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
