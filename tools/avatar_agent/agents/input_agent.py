from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


class InputAgent:
    """Prepare multimodal user inputs for the avatar pipeline."""

    def __init__(self, config: dict):
        self.config = config

    def run(self, state) -> None:
        if not getattr(state, "input_video", None):
            return
        video_path = Path(state.input_video)
        if not video_path.exists():
            return
        frames_dir = Path(state.run_dir) / "input_agent" / "video_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        state.video_frames_dir = str(frames_dir)

        ffmpeg = Path("/scratch/e1554543/avatar_system_full/tools/ffmpeg-git-20240629-amd64-static/ffmpeg")
        ffmpeg_bin = str(ffmpeg) if ffmpeg.exists() else "ffmpeg"
        frame_pattern = frames_dir / "frame_%02d.jpg"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "fps=1,scale=360:-1",
            "-frames:v",
            "3",
            str(frame_pattern),
        ]
        log_path = Path(state.log_dir) / "input_agent.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n\n")
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            log.write(proc.stdout or "")
            log.write(f"\n[exit code: {proc.returncode}]\n")
        state.extra.setdefault("input_agent", {})["video_frames_dir"] = str(frames_dir)
        state.extra["input_agent"]["frame_count"] = len(list(frames_dir.glob("frame_*.jpg")))
