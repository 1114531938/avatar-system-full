from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from avatar_system.pipeline.config import project_path
except ModuleNotFoundError:
    project_path = None


@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


def prepare_camera(width: int, height: int, camera_json: Path):
    from utils.viewer_utils import OrbitCamera

    cam = OrbitCamera(
        width,
        height,
        r=1,
        fovy=20,
        convention="opencv",
        save_path=str(camera_json),
    )

    @dataclass
    class Cam:
        FoVx = float(np.radians(cam.fovx))
        FoVy = float(np.radians(cam.fovy))
        image_height = cam.image_height
        image_width = cam.image_width
        world_view_transform = torch.tensor(cam.world_view_transform).float().cuda().T
        full_proj_transform = torch.tensor(cam.full_proj_transform).float().cuda().T
        camera_center = torch.tensor(cam.pose[:3, 3]).cuda()

    return Cam


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return fallback.astype(np.float32)
    return (v / norm).astype(np.float32)


def prepare_camera_from_three_payload(width: int, height: int, payload: dict):
    from utils.viewer_utils import projection_from_intrinsics

    target = np.array(payload.get("export_target", [0, 0, 0]), dtype=np.float32)
    if "direction" in payload:
        direction = _normalize(
            np.array(payload["direction"], dtype=np.float32),
            np.array([0, 0, 1], dtype=np.float32),
        )
        base_radius = float(payload.get("base_radius", 1.0))
        radius_norm = float(payload.get("radius_norm", 1.0))
        radius_norm = float(np.clip(radius_norm, 0.55, 2.5))
        position = target + direction * base_radius * radius_norm
    else:
        position = np.array(payload["position"], dtype=np.float32)
        target = np.array(payload.get("target", [0, 0, 0]), dtype=np.float32)
    up = np.array(payload.get("up", [0, 1, 0]), dtype=np.float32)
    fovy = float(payload.get("fov", 20.0))
    znear = float(payload.get("near", 0.01))
    zfar = float(payload.get("far", 10.0))

    if "direction" not in payload:
        # Legacy camera payloads used the centered frontend point cloud coordinates.
        center_offset = np.array(payload.get("center_offset", [0, 0, 0]), dtype=np.float32)
        position = position + center_offset
        target = target + center_offset

    z_axis = _normalize(target - position, np.array([0, 0, -1], dtype=np.float32))
    x_axis = _normalize(np.cross(z_axis, up), np.array([1, 0, 0], dtype=np.float32))
    y_axis = _normalize(np.cross(z_axis, x_axis), np.array([0, -1, 0], dtype=np.float32))

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = position
    world_view_transform = np.linalg.inv(c2w).astype(np.float32)

    focal = height / (2 * np.tan(np.radians(fovy) / 2))
    fovx = 2 * np.arctan(width / (2 * focal))
    intrinsics = np.array([focal, focal, width // 2, height // 2], dtype=np.float32)
    projection = projection_from_intrinsics(
        intrinsics[None],
        (height, width),
        near=znear,
        far=zfar,
        z_sign=1,
    )[0].astype(np.float32)
    full_proj_transform = projection @ world_view_transform

    class Cam:
        pass

    cam = Cam()
    cam.FoVx = float(fovx)
    cam.FoVy = float(np.radians(fovy))
    cam.image_height = height
    cam.image_width = width
    cam.world_view_transform = torch.tensor(world_view_transform).float().cuda().T
    cam.full_proj_transform = torch.tensor(full_proj_transform).float().cuda().T
    cam.camera_center = torch.tensor(position).float().cuda()
    return cam


def prepare_camera_from_three(width: int, height: int, camera_json: Path):
    with camera_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return prepare_camera_from_three_payload(width, height, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GaussianAvatar motion to mp4")
    parser.add_argument("--gaussian_root", required=True)
    parser.add_argument("--point_path", required=True)
    parser.add_argument("--motion_path", required=True)
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--out_video", required=True)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--enhanced_audio_out", default=None)
    parser.add_argument(
        "--audio_filter",
        default="highpass=f=70,lowpass=f=7600,loudnorm=I=-16:LRA=11:TP=-1.5,aresample=48000",
        help="ffmpeg audio filter used for the muxed voice; use an empty string to disable.",
    )
    parser.add_argument(
        "--render_mode",
        choices=["gaussian", "white_mesh", "overlay"],
        default="gaussian",
        help="gaussian is the final avatar; white_mesh is a clean white FLAME mesh; overlay blends mesh over gaussian.",
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=550)
    parser.add_argument("--height", type=int, default=802)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument(
        "--disable_micro_expression",
        action="store_true",
        help="Render the base Gaussian avatar even when adjacent micro-expression weights exist.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--keep_frames", action="store_true")
    parser.add_argument(
        "--camera_json",
        default=None,
        help="Optional Three.js camera JSON with position/target/up/fov/center_offset.",
    )
    parser.add_argument(
        "--no_sync_to_audio",
        action="store_true",
        help="Deprecated alias for --sync_mode preserve_motion.",
    )
    parser.add_argument(
        "--sync_mode",
        choices=["preserve_motion", "stretch_video"],
        default="preserve_motion",
        help="preserve_motion keeps the DEEPTalk fps and pads audio tail; stretch_video changes fps to exact audio duration.",
    )
    args = parser.parse_args()

    gaussian_root = Path(args.gaussian_root).resolve()
    sys.path.insert(0, str(gaussian_root))
    os.chdir(gaussian_root)

    from gaussian_renderer import FlameGaussianModel, GaussianModel, render
    from mesh_renderer import NVDiffRenderer

    point_path = Path(args.point_path).resolve()
    motion_path = Path(args.motion_path).resolve()
    audio_path = Path(args.audio_path).resolve()
    out_video = Path(args.out_video).resolve()
    frames_dir = Path(args.frames_dir).resolve()
    enhanced_audio_out = Path(args.enhanced_audio_out).resolve() if args.enhanced_audio_out else None

    if not point_path.exists():
        raise FileNotFoundError(f"point_path not found: {point_path}")
    if not motion_path.exists():
        raise FileNotFoundError(f"motion_path not found: {motion_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"audio_path not found: {audio_path}")

    out_video.parent.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = args.ffmpeg
    fallback_ffmpeg = (
        project_path("runtime", "cache", "bin", "ffmpeg")
        if project_path is not None
        else Path("/scratch/e1554543/avatar_system_full/runtime/cache/bin/ffmpeg")
    )
    if not Path(ffmpeg).exists() and fallback_ffmpeg.exists():
        ffmpeg = str(fallback_ffmpeg)

    mux_audio_path = audio_path
    if args.audio_filter.strip():
        if enhanced_audio_out is None:
            enhanced_audio_out = out_video.parent / "reply_enhanced.wav"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(audio_path),
                "-af",
                args.audio_filter,
                "-ar",
                "48000",
                "-ac",
                "2",
                str(enhanced_audio_out),
            ],
            check=True,
        )
        mux_audio_path = enhanced_audio_out

    with torch.no_grad():
        if (point_path.parent / "flame_param.npz").exists():
            gaussians = FlameGaussianModel(args.sh_degree)
        else:
            gaussians = GaussianModel(args.sh_degree)

        gaussians.load_ply(
            point_path,
            has_target=False,
            motion_path=motion_path,
            load_micro_expression=not args.disable_micro_expression,
        )
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.camera_json:
            cam = prepare_camera_from_three(args.width, args.height, Path(args.camera_json).resolve())
        else:
            cam = prepare_camera(args.width, args.height, out_video.parent / "export_camera.json")
        pipeline = PipelineConfig()
        mesh_renderer = None
        if args.render_mode in {"white_mesh", "overlay"}:
            mesh_renderer = NVDiffRenderer(use_opengl=False, lighting_type="front")
        mesh_color = None

        num_frames = int(getattr(gaussians, "num_timesteps", 1) or 1)
        print(f"Rendering {num_frames} frames to {frames_dir}", flush=True)
        for frame_idx in range(num_frames):
            if gaussians.binding is not None:
                gaussians.select_mesh_by_timestep(frame_idx)
            image_hwc = None
            if args.render_mode in {"gaussian", "overlay"}:
                image = render(cam, gaussians, pipeline, background)["render"]
                image_hwc = image.permute(1, 2, 0).contiguous()
            if args.render_mode in {"white_mesh", "overlay"}:
                if gaussians.binding is None:
                    raise RuntimeError("white_mesh/overlay requires a FlameGaussianModel with mesh binding")
                if mesh_color is None:
                    mesh_color = torch.ones(
                        (1, gaussians.faces.shape[0], 3),
                        dtype=torch.float32,
                        device="cuda",
                    )
                mesh_out = mesh_renderer.render_from_camera(
                    gaussians.verts,
                    gaussians.faces,
                    cam,
                    background_color=[1.0, 1.0, 1.0],
                    face_colors=mesh_color,
                )
                rgba_mesh = mesh_out["rgba"].squeeze(0)
                rgb_mesh = rgba_mesh[:, :, :3]
                alpha_mesh = rgba_mesh[:, :, 3:]
                if args.render_mode == "white_mesh":
                    image_hwc = rgb_mesh
                else:
                    image_hwc = rgb_mesh * alpha_mesh * 0.45 + image_hwc * (1 - alpha_mesh * 0.45)
            image_np = (
                image_hwc.mul(255)
                .add_(0.5)
                .clamp_(0, 255)
                .to("cpu", torch.uint8)
                .numpy()
            )
            Image.fromarray(image_np).save(frames_dir / f"{frame_idx:05d}.png")
            if frame_idx == 0 or (frame_idx + 1) % 10 == 0 or frame_idx + 1 == num_frames:
                print(f"Rendered frame {frame_idx + 1}/{num_frames}", flush=True)

    effective_fps = float(args.fps)
    sync_mode = "preserve_motion" if args.no_sync_to_audio else args.sync_mode
    if sync_mode == "stretch_video":
        with wave.open(str(mux_audio_path), "rb") as wav:
            audio_duration = wav.getnframes() / float(wav.getframerate())
        if audio_duration > 0:
            effective_fps = num_frames / audio_duration

    tmp_video = out_video.with_suffix(".silent.mp4")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            f"{effective_fps:.6f}",
            "-i",
            str(frames_dir / "%05d.png"),
            "-pix_fmt",
            "yuv420p",
            str(tmp_video),
        ],
        check=True,
    )
    print(f"Wrote silent video: {tmp_video}", flush=True)
    mux_tmp_video = out_video.with_suffix(".muxing.mp4")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(tmp_video),
            "-i",
            str(mux_audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-shortest",
            "-movflags",
            "+faststart",
            str(mux_tmp_video),
        ],
        check=True,
    )
    os.replace(mux_tmp_video, out_video)
    print(f"Wrote muxed video: {out_video}", flush=True)
    tmp_video.unlink(missing_ok=True)

    if not args.keep_frames:
        for frame in frames_dir.glob("*.png"):
            frame.unlink()
        try:
            frames_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
