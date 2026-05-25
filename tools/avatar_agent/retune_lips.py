from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path

import numpy as np
import yaml

from shell_runner import run_bash, run_bash_in_container


def temporal_shift(x: np.ndarray, shift_frames: int) -> np.ndarray:
    if shift_frames == 0:
        return x
    out = np.empty_like(x)
    if shift_frames > 0:
        out[:shift_frames] = x[0]
        out[shift_frames:] = x[:-shift_frames]
    else:
        n = -shift_frames
        out[:-n] = x[n:]
        out[-n:] = x[-1]
    return out


def smooth_sequence(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(x, [(pad, pad), (0, 0)], mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    out = np.empty_like(x)
    for dim in range(x.shape[1]):
        out[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return out


def tune_existing_flame_npz(src_npz: str, out_npz: str, args: argparse.Namespace) -> None:
    src = np.load(src_npz, allow_pickle=True)
    out = {key: src[key] for key in src.files}

    if "expr" not in out or "jaw_pose" not in out:
        raise RuntimeError(f"Expected expr and jaw_pose in {src_npz}")

    expr = np.asarray(out["expr"], dtype=np.float32)
    jaw = np.asarray(out["jaw_pose"], dtype=np.float32)
    expr = temporal_shift(expr, args.temporal_shift_frames)
    jaw = temporal_shift(jaw, args.temporal_shift_frames)
    expr = smooth_sequence(expr, args.smooth_window) * np.float32(args.expr_scale)
    jaw = smooth_sequence(jaw, args.smooth_window) * np.float32(args.jaw_scale)

    src_jaw = np.asarray(out["jaw_pose"], dtype=np.float32)
    src_jaw = smooth_sequence(temporal_shift(src_jaw, args.temporal_shift_frames), args.smooth_window)
    if args.jaw_open_scale is not None:
        jaw[:, 0] = src_jaw[:, 0] * np.float32(args.jaw_open_scale)
    if args.jaw_side_scale is not None:
        jaw[:, 1] = src_jaw[:, 1] * np.float32(args.jaw_side_scale)
    if args.jaw_twist_scale is not None:
        jaw[:, 2] = src_jaw[:, 2] * np.float32(args.jaw_twist_scale)
    if args.mouth_open_bias:
        jaw[:, 0] += np.float32(args.mouth_open_bias)

    out["expr"] = expr
    out["jaw_pose"] = jaw
    out["timestep_id"] = np.arange(expr.shape[0], dtype=np.int32)
    out["n_processed_frames"] = np.array(expr.shape[0], dtype=np.int32)

    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create lip-sync tuning variants from an existing run")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--config", default="pipeline_config.yaml")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--expr_scale", type=float, default=1.0)
    parser.add_argument("--jaw_scale", type=float, default=1.0)
    parser.add_argument("--jaw_open_scale", type=float, default=None)
    parser.add_argument("--jaw_side_scale", type=float, default=None)
    parser.add_argument("--jaw_twist_scale", type=float, default=None)
    parser.add_argument("--mouth_open_bias", type=float, default=0.0)
    parser.add_argument("--temporal_shift_frames", type=int, default=0)
    parser.add_argument("--smooth_window", type=int, default=1)
    parser.add_argument("--render_mode", choices=["gaussian", "white_mesh", "overlay"], default="white_mesh")
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(os.path.join(args.run_dir, "state.json"), "r", encoding="utf-8") as f:
        state = json.load(f)

    p = config["paths"]
    runtime = config.get("runtime", {})
    run_dir = os.path.abspath(args.run_dir)
    tag = args.tag or (
        f"expr{args.expr_scale:g}_jaw{args.jaw_scale:g}_open"
        f"{args.jaw_open_scale if args.jaw_open_scale is not None else 'base'}"
        f"_shift{args.temporal_shift_frames}_smooth{args.smooth_window}"
    )
    out_dir = os.path.join(run_dir, "lip_tuning", tag)
    os.makedirs(out_dir, exist_ok=True)

    deeptalk_npy = os.path.join(run_dir, "outputs", "deeptalk.npy")
    local_deeptalk_exists = os.path.exists(deeptalk_npy)
    template_npz = state.get("template_npz") or os.path.join(
        p["gaussian_root"], "media", str(state["avatar_id"]), "flame_param.npz"
    )
    point_path = state.get("point_cloud_path") or os.path.join(
        p["gaussian_root"], "media", str(state["avatar_id"]), "point_cloud.ply"
    )
    reply_wav = os.path.join(run_dir, "outputs", "reply.wav")
    if not os.path.exists(reply_wav):
        reply_wav = state["reply_wav"]

    tuned_npz = os.path.join(out_dir, "flame_motion.npz")
    q = shlex.quote
    source_motion = deeptalk_npy
    source_kind = "deeptalk_npy"
    if local_deeptalk_exists:
        extra = []
        for flag, value in [
            ("jaw-open-scale", args.jaw_open_scale),
            ("jaw-side-scale", args.jaw_side_scale),
            ("jaw-twist-scale", args.jaw_twist_scale),
        ]:
            if value is not None:
                extra.append(f" --{flag} {q(str(value))}")
        merge_cmd = f"""
        {q(os.path.join(p["deeptalk_venv"], "bin/python"))} {q(p["deeptalk_merge_script"])} \
          --deeptalk_motion {q(deeptalk_npy)} \
          --template {q(template_npz)} \
          --out {q(tuned_npz)} \
          --expr-scale {q(str(args.expr_scale))} \
          --jaw-scale {q(str(args.jaw_scale))} \
          --mouth-open-bias {q(str(args.mouth_open_bias))} \
          --temporal-shift-frames {q(str(args.temporal_shift_frames))} \
          --smooth-window {q(str(args.smooth_window))}{''.join(extra)} \
          --zero-translation
        """
        run_bash(merge_cmd, os.path.join(out_dir, "merge.log"))
    else:
        base_npz = os.path.join(run_dir, "artifacts", "flame_motion.npz")
        if not os.path.exists(base_npz):
            base_npz = state["flame_motion_npz"]
        source_motion = base_npz
        source_kind = "flame_motion_npz"
        tune_existing_flame_npz(base_npz, tuned_npz, args)
        with open(os.path.join(out_dir, "merge.log"), "w", encoding="utf-8") as f:
            f.write(f"retuned existing FLAME motion: {base_npz}\n")

    result = {
        "run_dir": run_dir,
        "tag": tag,
        "source_kind": source_kind,
        "source_motion": source_motion,
        "deeptalk_npy": deeptalk_npy if local_deeptalk_exists else None,
        "reply_wav": reply_wav,
        "flame_motion_npz": tuned_npz,
        "expr_scale": args.expr_scale,
        "jaw_scale": args.jaw_scale,
        "jaw_open_scale": args.jaw_open_scale,
        "jaw_side_scale": args.jaw_side_scale,
        "jaw_twist_scale": args.jaw_twist_scale,
        "mouth_open_bias": args.mouth_open_bias,
        "temporal_shift_frames": args.temporal_shift_frames,
        "smooth_window": args.smooth_window,
    }

    if not args.no_video:
        gaussian_root = p["gaussian_root"]
        out_video = os.path.join(out_dir, f"{args.render_mode}.mp4")
        frames_dir = os.path.join(out_dir, "frames")
        enhanced_audio = os.path.join(out_dir, "reply_enhanced.wav")
        ffmpeg = runtime.get("ffmpeg", "/scratch/e1554543/avatar_system_full/tools/ffmpeg-git-20240629-amd64-static/ffmpeg")
        exporter_py = os.path.abspath(os.path.join(os.path.dirname(__file__), "export_gaussian_video.py"))
        export_cmd = f"""
        cd {q(gaussian_root)}
        {q(os.path.join(p["gaussian_venv"], "bin/python"))} {q(exporter_py)} \
          --gaussian_root {q(gaussian_root)} \
          --point_path {q(point_path)} \
          --motion_path {q(tuned_npz)} \
          --audio_path {q(reply_wav)} \
          --out_video {q(out_video)} \
          --frames_dir {q(frames_dir)} \
          --enhanced_audio_out {q(enhanced_audio)} \
          --render_mode {q(args.render_mode)} \
          --fps {q(str(runtime.get("video_fps", 25)))} \
          --width {q(str(runtime.get("video_width", 550)))} \
          --height {q(str(runtime.get("video_height", 802)))} \
          --ffmpeg {q(ffmpeg)}
        """
        return_code, _, output = run_bash_in_container(
            export_cmd,
            p["gaussian_container_image"],
            os.path.join(out_dir, "export.log"),
            check=False,
        )
        result["video"] = out_video if return_code == 0 and os.path.exists(out_video) else None
        result["video_export_error"] = None if result["video"] else output

    with open(os.path.join(out_dir, "tuning_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
