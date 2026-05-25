#!/usr/bin/env python3
"""
Convert a WAV file into a GaussianAvatars-compatible flame_param.npz
using a demo/template flame_param.npz for static fields.

Goal:
- Keep the demo avatar identity/static geometry from the template
- Create a simple, stable audio-driven motion file from WAV
- Prioritize "can run in local_viewer.py" over physical accuracy

This script only estimates a mild jaw motion from audio energy.
It does NOT recover full high-quality FLAME motion from audio.
"""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import numpy as np


def load_wav_mono(wav_path: str):
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported wav sample width: {sample_width}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # remove DC
    audio = audio - audio.mean()

    peak = np.max(np.abs(audio)) + 1e-8
    audio = audio / peak
    return audio.astype(np.float32), sr


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    k = int(k)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / float(k)
    y = np.convolve(xp, kernel, mode="same")
    return y[pad:pad + len(x)]


def audio_to_jaw(audio: np.ndarray, sr: int, fps: int, jaw_scale: float, jaw_sign: float) -> np.ndarray:
    # 40 ms analysis window
    win_size = max(1, int(0.040 * sr))
    hop = sr / float(fps)
    T = max(1, int(math.ceil(len(audio) / hop)))

    rms = np.zeros((T,), dtype=np.float32)
    half = win_size // 2
    for t in range(T):
        center = int(round(t * hop))
        left = max(0, center - half)
        right = min(len(audio), center + half)
        seg = audio[left:right]
        if len(seg) == 0:
            val = 0.0
        else:
            val = float(np.sqrt(np.mean(seg * seg) + 1e-8))
        rms[t] = val

    # smooth
    rms = moving_average(rms, max(3, fps // 6))
    rms = moving_average(rms, max(3, fps // 8))

    # robust normalize
    p95 = np.percentile(rms, 95)
    if p95 < 1e-6:
        norm = np.zeros_like(rms)
    else:
        norm = np.clip(rms / p95, 0.0, 1.0)

    # compress dynamic range slightly
    norm = np.power(norm, 0.7)

    jaw = jaw_sign * jaw_scale * norm
    return jaw.astype(np.float32)


def build_motion_from_template(
    template_path: str,
    wav_path: str,
    out_path: str,
    fps: int = 25,
    jaw_scale: float = 0.35,
    jaw_sign: float = 1.0,
    expr_scale: float = 0.0,
):
    tpl_npz = np.load(template_path, allow_pickle=True)
    tpl = {k: tpl_npz[k] for k in tpl_npz.files}

    audio, sr = load_wav_mono(wav_path)
    jaw_open = audio_to_jaw(audio, sr, fps=fps, jaw_scale=jaw_scale, jaw_sign=jaw_sign)
    T = jaw_open.shape[0]

    if "shape" not in tpl:
        raise KeyError("Template is missing 'shape'")
    if "static_offset" not in tpl:
        raise KeyError("Template is missing 'static_offset'")

    out = {}

    # Keep all non-dynamic / non-time-varying fields from template by default
    for k, v in tpl.items():
        out[k] = v

    # Infer dimensions from template when available
    expr_dim = int(tpl["expr"].shape[1]) if "expr" in tpl and tpl["expr"].ndim == 2 else 100
    if "dynamic_offset" in tpl and tpl["dynamic_offset"].ndim == 3:
        V = int(tpl["dynamic_offset"].shape[1])
    else:
        V = int(tpl["static_offset"].shape[1])

    out["expr"] = np.zeros((T, expr_dim), dtype=np.float32)
    out["rotation"] = np.zeros((T, 3), dtype=np.float32)
    out["translation"] = np.zeros((T, 3), dtype=np.float32)
    out["neck_pose"] = np.zeros((T, 3), dtype=np.float32)
    out["jaw_pose"] = np.zeros((T, 3), dtype=np.float32)
    out["eyes_pose"] = np.zeros((T, 6), dtype=np.float32)
    out["dynamic_offset"] = np.zeros((T, V, 3), dtype=np.float32)

    # Minimal stable audio-driven motion:
    # put mouth opening into jaw pitch axis
    out["jaw_pose"][:, 0] = jaw_open

    # Optional tiny expression coupling (kept very small by default)
    if expr_scale > 0 and expr_dim >= 2:
        out["expr"][:, 0] = jaw_open * expr_scale
        out["expr"][:, 1] = jaw_open * (expr_scale * 0.5)

    out["shape"] = np.asarray(tpl["shape"], dtype=np.float32)
    out["static_offset"] = np.asarray(tpl["static_offset"], dtype=np.float32)

    out["timestep_id"] = np.arange(T, dtype=np.int32)
    out["n_processed_frames"] = np.array(T, dtype=np.int32)

    # Keep optional camera/light/texture metadata if present
    for k in ["focal_length", "tex_extra", "lights", "image_size"]:
        if k in tpl:
            out[k] = tpl[k]

    np.savez(out_path, **out)

    print("saved:", out_path)
    print("fps:", fps)
    print("audio_sr:", sr)
    print("n_frames:", T)
    print("keys:", sorted(out.keys()))
    print("jaw range:", float(out["jaw_pose"][:, 0].min()), float(out["jaw_pose"][:, 0].max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="Input wav path, e.g. 1.wav")
    parser.add_argument("--template", required=True, help="Template flame_param.npz, e.g. media/306/flame_param.npz")
    parser.add_argument("--out", required=True, help="Output npz path, e.g. flame_param_from_1wav.npz")
    parser.add_argument("--fps", type=int, default=25, help="Target FLAME frame rate")
    parser.add_argument("--jaw-scale", type=float, default=0.35, help="Jaw opening scale")
    parser.add_argument("--jaw-sign", type=float, default=1.0, help="Use -1.0 if mouth moves in the wrong direction")
    parser.add_argument("--expr-scale", type=float, default=0.0, help="Tiny optional expression coupling; keep 0.0 first")
    args = parser.parse_args()

    build_motion_from_template(
        template_path=args.template,
        wav_path=args.wav,
        out_path=args.out,
        fps=args.fps,
        jaw_scale=args.jaw_scale,
        jaw_sign=args.jaw_sign,
        expr_scale=args.expr_scale,
    )


if __name__ == "__main__":
    main()
