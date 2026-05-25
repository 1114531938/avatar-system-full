from __future__ import annotations

import argparse
import json
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = Path("/scratch/e1554543/avatar_system_full/tools/avatar_agent")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS_ROOT))

from export_gaussian_video import prepare_camera_from_three, prepare_camera_from_three_payload  # noqa: E402
from gaussian_renderer import FlameGaussianModel, GaussianModel, render  # noqa: E402


@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class GaussianRenderEngine:
    def __init__(self, sh_degree: int = 3) -> None:
        self.sh_degree = sh_degree
        self.lock = threading.Lock()
        self.cache: dict[tuple[str, str], Any] = {}
        self.pipeline = PipelineConfig()
        self.background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    def _load_gaussians(self, point_path: str, motion_path: str):
        key = (str(Path(point_path).resolve()), str(Path(motion_path).resolve()))
        if key in self.cache:
            return self.cache[key]

        point = Path(point_path).resolve()
        motion = Path(motion_path).resolve()
        if not point.exists():
            raise FileNotFoundError(f"point_path not found: {point}")
        if not motion.exists():
            raise FileNotFoundError(f"motion_path not found: {motion}")

        if (point.parent / "flame_param.npz").exists():
            gaussians = FlameGaussianModel(self.sh_degree)
        else:
            gaussians = GaussianModel(self.sh_degree)
        print(f"[gaussian_render_worker] loading point={point} motion={motion}", flush=True)
        gaussians.load_ply(point, has_target=False, motion_path=motion)
        self.cache[key] = gaussians
        return gaussians

    def render_frame(
        self,
        *,
        point_path: str,
        motion_path: str,
        camera_json: str,
        output_image: str,
        frame: int,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        with self.lock:
            gaussians = self._load_gaussians(point_path, motion_path)
            num_frames = int(getattr(gaussians, "num_timesteps", 1) or 1)
            frame_idx = max(0, min(int(frame), num_frames - 1))
            if gaussians.binding is not None:
                gaussians.select_mesh_by_timestep(frame_idx)
            cam = prepare_camera_from_three(width, height, Path(camera_json).resolve())
            with torch.no_grad():
                image = render(cam, gaussians, self.pipeline, self.background)["render"]
            image_np = (
                image.permute(1, 2, 0)
                .contiguous()
                .mul(255)
                .add_(0.5)
                .clamp_(0, 255)
                .to("cpu", torch.uint8)
                .numpy()
            )

        out_path = Path(output_image).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image_np).save(out_path)
        return {
            "output_image": str(out_path),
            "frame": frame_idx,
            "frame_count": num_frames,
            "width": width,
            "height": height,
        }

    def render_frame_bytes(
        self,
        *,
        point_path: str,
        motion_path: str,
        camera_payload: dict[str, Any],
        frame: int,
        width: int,
        height: int,
        image_format: str = "jpeg",
        quality: int = 88,
    ) -> tuple[bytes, dict[str, Any]]:
        with self.lock:
            gaussians = self._load_gaussians(point_path, motion_path)
            num_frames = int(getattr(gaussians, "num_timesteps", 1) or 1)
            frame_idx = max(0, min(int(frame), num_frames - 1))
            if gaussians.binding is not None:
                gaussians.select_mesh_by_timestep(frame_idx)
            cam = prepare_camera_from_three_payload(width, height, camera_payload)
            with torch.no_grad():
                image = render(cam, gaussians, self.pipeline, self.background)["render"]
            image_np = (
                image.permute(1, 2, 0)
                .contiguous()
                .mul(255)
                .add_(0.5)
                .clamp_(0, 255)
                .to("cpu", torch.uint8)
                .numpy()
            )

        fmt = str(image_format or "jpeg").lower()
        if fmt in {"jpg", "jpeg"}:
            pil_format = "JPEG"
            mime = "image/jpeg"
            save_kwargs = {"quality": max(40, min(int(quality), 95)), "optimize": False}
        elif fmt == "webp":
            pil_format = "WEBP"
            mime = "image/webp"
            save_kwargs = {"quality": max(40, min(int(quality), 95)), "method": 4}
        else:
            pil_format = "PNG"
            mime = "image/png"
            save_kwargs = {"compress_level": 1}

        image_pil = Image.fromarray(image_np)
        if pil_format == "JPEG":
            image_pil = image_pil.convert("RGB")

        import io

        buffer = io.BytesIO()
        image_pil.save(buffer, format=pil_format, **save_kwargs)
        return buffer.getvalue(), {
            "frame": frame_idx,
            "frame_count": num_frames,
            "width": width,
            "height": height,
            "mime": mime,
            "image_format": fmt,
        }

    def export_motion_positions(
        self,
        *,
        point_path: str,
        motion_path: str,
        output_path: str,
        frame_stride: int = 1,
        max_frames: int = 360,
    ) -> dict[str, Any]:
        with self.lock:
            gaussians = self._load_gaussians(point_path, motion_path)
            num_frames = int(getattr(gaussians, "num_timesteps", 1) or 1)
            frame_stride = max(1, int(frame_stride))
            export_frames = list(range(0, min(num_frames, int(max_frames)), frame_stride))
            if not export_frames:
                export_frames = [0]
            out_path = Path(output_path).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            point_count = None
            with tmp_path.open("wb") as f:
                for frame_idx in export_frames:
                    if gaussians.binding is not None:
                        gaussians.select_mesh_by_timestep(frame_idx)
                    with torch.no_grad():
                        xyz = gaussians.get_xyz.detach()
                        scaling = torch.log(gaussians.get_scaling.detach().clamp_min(1e-8))
                        rotation = gaussians.get_rotation.detach()
                        attrs = torch.cat([xyz, scaling, rotation], dim=1).to("cpu", torch.float16).numpy()
                    if point_count is None:
                        point_count = int(attrs.shape[0])
                    attrs.tofile(f)
            tmp_path.replace(out_path)
        return {
            "output_path": str(out_path),
            "dtype": "float16",
            "point_count": int(point_count or 0),
            "values_per_point": 10,
            "source_frame_count": num_frames,
            "frame_count": len(export_frames),
            "frame_stride": frame_stride,
            "first_frame": export_frames[0],
            "last_frame": export_frames[-1],
            "bytes": out_path.stat().st_size,
        }


class GaussianRenderRequestHandler(BaseHTTPRequestHandler):
    engine: GaussianRenderEngine

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("[gaussian_render_worker] " + format % args + "\n")
        sys.stdout.flush()

    def do_GET(self) -> None:
        if self.path != "/health":
            _json_response(self, 404, {"ok": False, "error": "not found"})
            return
        _json_response(
            self,
            200,
            {
                "ok": True,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "cached_models": len(self.engine.cache),
            },
        )

    def do_POST(self) -> None:
        if self.path not in {"/render_frame", "/render_frame_bytes", "/export_motion_positions"}:
            _json_response(self, 404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/render_frame":
                result = self.engine.render_frame(
                    point_path=str(payload["point_path"]),
                    motion_path=str(payload["motion_path"]),
                    camera_json=str(payload["camera_json"]),
                    output_image=str(payload["output_image"]),
                    frame=int(payload.get("frame", 0)),
                    width=int(payload.get("width", 550)),
                    height=int(payload.get("height", 802)),
                )
                _json_response(self, 200, {"ok": True, **result})
            elif self.path == "/render_frame_bytes":
                image_bytes, meta = self.engine.render_frame_bytes(
                    point_path=str(payload["point_path"]),
                    motion_path=str(payload["motion_path"]),
                    camera_payload=dict(payload["camera"]),
                    frame=int(payload.get("frame", 0)),
                    width=int(payload.get("width", 550)),
                    height=int(payload.get("height", 802)),
                    image_format=str(payload.get("image_format", "jpeg")),
                    quality=int(payload.get("quality", 88)),
                )
                self.send_response(200)
                self.send_header("Content-Type", meta["mime"])
                self.send_header("Content-Length", str(len(image_bytes)))
                self.send_header("X-Frame", str(meta["frame"]))
                self.send_header("X-Frame-Count", str(meta["frame_count"]))
                self.send_header("X-Image-Format", str(meta["image_format"]))
                self.end_headers()
                self.wfile.write(image_bytes)
            else:
                result = self.engine.export_motion_positions(
                    point_path=str(payload["point_path"]),
                    motion_path=str(payload["motion_path"]),
                    output_path=str(payload["output_path"]),
                    frame_stride=int(payload.get("frame_stride", 1)),
                    max_frames=int(payload.get("max_frames", 360)),
                )
                _json_response(self, 200, {"ok": True, **result})
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-lived Gaussian frame render worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument("--sh_degree", type=int, default=3)
    args = parser.parse_args()

    torch.set_num_threads(1)
    print("[gaussian_render_worker] starting...", flush=True)
    GaussianRenderRequestHandler.engine = GaussianRenderEngine(sh_degree=args.sh_degree)
    print(
        f"[gaussian_render_worker] ready on http://{args.host}:{args.port} "
        f"device={'cuda' if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )
    server = ThreadingHTTPServer((args.host, args.port), GaussianRenderRequestHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
