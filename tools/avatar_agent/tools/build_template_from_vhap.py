#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_npz(path: str) -> dict:
    obj = np.load(path, allow_pickle=True)
    return {k: obj[k] for k in obj.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GaussianAvatars template flame_param.npz from VHAP export")
    parser.add_argument("--canonical", required=True, help="VHAP-exported canonical_flame_param.npz")
    parser.add_argument("--tracked", default=None, help="Optional tracked_flame_params*.npz for dynamic_offset shape hints")
    parser.add_argument("--out", required=True, help="Output flame_param.npz")
    args = parser.parse_args()

    canonical = load_npz(args.canonical)
    tracked = load_npz(args.tracked) if args.tracked else {}

    required = ["shape", "expr", "rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation"]
    missing = [k for k in required if k not in canonical]
    if missing:
        raise KeyError(f"canonical file missing keys: {missing}")

    shape = np.asarray(canonical["shape"], dtype=np.float32)
    expr = np.asarray(canonical["expr"], dtype=np.float32)
    rotation = np.asarray(canonical["rotation"], dtype=np.float32)
    neck_pose = np.asarray(canonical["neck_pose"], dtype=np.float32)
    jaw_pose = np.asarray(canonical["jaw_pose"], dtype=np.float32)
    eyes_pose = np.asarray(canonical["eyes_pose"], dtype=np.float32)
    translation = np.asarray(canonical["translation"], dtype=np.float32)
    static_offset = np.asarray(canonical.get("static_offset", tracked.get("static_offset")), dtype=np.float32) \
        if ("static_offset" in canonical or "static_offset" in tracked) else None

    if expr.ndim != 2 or expr.shape[0] < 1:
        raise ValueError(f"Unexpected canonical expr shape: {expr.shape}")
    if static_offset is None:
        raise KeyError("No static_offset found in canonical or tracked VHAP outputs")

    if static_offset.ndim == 2:
        static_offset = static_offset[None, ...]
    if static_offset.ndim != 3:
        raise ValueError(f"Unexpected static_offset shape: {static_offset.shape}")

    n_vertices = static_offset.shape[1]
    n_frames = expr.shape[0]

    if "dynamic_offset" in tracked:
        dynamic_offset = np.asarray(tracked["dynamic_offset"], dtype=np.float32)
        if dynamic_offset.ndim != 3 or dynamic_offset.shape[1] != n_vertices:
            raise ValueError(f"Unexpected tracked dynamic_offset shape: {dynamic_offset.shape}")
    else:
        dynamic_offset = np.zeros((n_frames, n_vertices, 3), dtype=np.float32)

    out = {
        "shape": shape,
        "expr": expr,
        "rotation": rotation,
        "neck_pose": neck_pose,
        "jaw_pose": jaw_pose,
        "eyes_pose": eyes_pose,
        "translation": translation,
        "static_offset": static_offset,
        "dynamic_offset": dynamic_offset,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)

    print("saved:", out_path)
    for k, v in out.items():
        print(k, v.shape, v.dtype)


if __name__ == "__main__":
    main()
