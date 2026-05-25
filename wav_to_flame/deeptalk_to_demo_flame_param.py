import argparse
from pathlib import Path
import numpy as np


def temporal_shift(x: np.ndarray, shift_frames: int) -> np.ndarray:
    if shift_frames == 0:
        return x
    out = np.empty_like(x)
    if shift_frames > 0:
        # Positive means delay the motion: repeat the first frame at the beginning.
        out[:shift_frames] = x[0]
        out[shift_frames:] = x[:-shift_frames]
    else:
        # Negative means advance the motion: useful when lips are visibly late.
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
    for d in range(x.shape[1]):
        out[:, d] = np.convolve(padded[:, d], kernel, mode="valid")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deeptalk_motion", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expr-scale", type=float, default=1.0)
    parser.add_argument("--jaw-scale", type=float, default=1.0)
    parser.add_argument("--jaw-open-scale", type=float, default=None)
    parser.add_argument("--jaw-side-scale", type=float, default=None)
    parser.add_argument("--jaw-twist-scale", type=float, default=None)
    parser.add_argument("--mouth-open-bias", type=float, default=0.0)
    parser.add_argument("--temporal-shift-frames", type=int, default=0)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--zero-translation", action="store_true")
    args = parser.parse_args()

    tpl_npz = np.load(args.template, allow_pickle=True)
    tpl = {k: tpl_npz[k] for k in tpl_npz.files}

    x = np.load(args.deeptalk_motion, allow_pickle=True)

    if not isinstance(x, np.ndarray):
        raise RuntimeError(f"Unexpected motion type: {type(x)}")

    if x.ndim != 2 or x.shape[1] != 53:
        raise RuntimeError(f"Expected DEEPTalk ndarray of shape (T,53), got {x.shape}")

    x = temporal_shift(x, args.temporal_shift_frames)
    x = smooth_sequence(x, args.smooth_window)

    T = x.shape[0]

    # 假设：前50维是 expression，后3维是 jaw pose
    expr_50 = x[:, :50].astype(np.float32) * args.expr_scale
    jaw_3 = x[:, 50:53].astype(np.float32) * args.jaw_scale
    if args.jaw_open_scale is not None:
        jaw_3[:, 0] = x[:, 50].astype(np.float32) * args.jaw_open_scale
    if args.jaw_side_scale is not None:
        jaw_3[:, 1] = x[:, 51].astype(np.float32) * args.jaw_side_scale
    if args.jaw_twist_scale is not None:
        jaw_3[:, 2] = x[:, 52].astype(np.float32) * args.jaw_twist_scale
    if args.mouth_open_bias:
        jaw_3[:, 0] += np.float32(args.mouth_open_bias)

    # demo 模板里 expr 是 100 维，就把前 50 维填进去，后 50 维补 0
    expr_dim = int(tpl["expr"].shape[1]) if "expr" in tpl else 100
    if expr_dim < 50:
        raise RuntimeError(f"Template expr dim is too small: {expr_dim}")

    expr_full = np.zeros((T, expr_dim), dtype=np.float32)
    expr_full[:, :50] = expr_50

    V = int(tpl["static_offset"].shape[1])

    out = {}
    out["shape"] = np.asarray(tpl["shape"], dtype=np.float32)
    out["static_offset"] = np.asarray(tpl["static_offset"], dtype=np.float32)

    out["expr"] = expr_full
    out["jaw_pose"] = jaw_3

    out["rotation"] = np.zeros((T, 3), dtype=np.float32)
    out["neck_pose"] = np.zeros((T, 3), dtype=np.float32)
    out["eyes_pose"] = np.zeros((T, 6), dtype=np.float32)
    out["translation"] = np.zeros((T, 3), dtype=np.float32)
    out["dynamic_offset"] = np.zeros((T, V, 3), dtype=np.float32)

    # 保留模板里可能用到的额外字段
    for k in ["focal_length", "tex_extra", "lights", "image_size"]:
        if k in tpl:
            out[k] = tpl[k]

    out["timestep_id"] = np.arange(T, dtype=np.int32)
    out["n_processed_frames"] = np.array(T, dtype=np.int32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)

    print("saved:", out_path)
    print("T =", T)
    print("expr shape =", out["expr"].shape)
    print("jaw shape =", out["jaw_pose"].shape)
    print("jaw min =", out["jaw_pose"].min(axis=0))
    print("jaw max =", out["jaw_pose"].max(axis=0))
    print("dynamic_offset shape =", out["dynamic_offset"].shape)


if __name__ == "__main__":
    main() 
