from __future__ import annotations

import os
import shutil
import shlex
import subprocess
from pathlib import Path

from avatar_system.pipeline.config import project_path
from avatar_system.pipeline.manifest_utils import find_first_value, load_json, save_json
from avatar_system.pipeline.shell_runner import run_bash_in_container
from avatar_system.tools.perception_tool import PerceptionTool


class InputAgent:
    """Prepare audio/video input and normalize perception outputs."""

    def __init__(self, config: dict):
        self.config = config
        self.perception_tool = PerceptionTool(config)

    def _resolve_ffmpeg(self) -> str | None:
        env_ffmpeg = os.environ.get("AVATAR_FFMPEG") or os.environ.get("FFMPEG")
        if env_ffmpeg and Path(env_ffmpeg).exists():
            return env_ffmpeg
        local_ffmpeg = project_path("tools", "ffmpeg-git-20240629-amd64-static", "ffmpeg")
        if local_ffmpeg.exists():
            return str(local_ffmpeg)
        runtime_ffmpeg = project_path("runtime", "cache", "bin", "ffmpeg")
        if runtime_ffmpeg.exists():
            return str(runtime_ffmpeg)
        return shutil.which("ffmpeg")

    def _run_ffmpeg(self, cmd: list[str], log_path: Path) -> int:
        ffmpeg_bin = self._resolve_ffmpeg()
        with log_path.open("w", encoding="utf-8") as log:
            if ffmpeg_bin:
                cmd[0] = ffmpeg_bin
                log.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n\n")
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                log.write(proc.stdout or "")
                log.write(f"\n[exit code: {proc.returncode}]\n")
                return proc.returncode

            container_image = self.config.get("paths", {}).get("gaussian_container_image")
            if not container_image or not Path(container_image).exists():
                raise FileNotFoundError(
                    "ffmpeg not found on host and Gaussian container is unavailable. "
                    "Set AVATAR_FFMPEG=/path/to/ffmpeg or restore runtime/containers/gaussianav_jammy."
                )

            container_cmd = ["/usr/bin/ffmpeg", *cmd[1:]]
            shell_cmd = " ".join(shlex.quote(part) for part in container_cmd)
            log.write("$ apptainer exec ... " + shell_cmd + "\n\n")

        code, output, _ = run_bash_in_container(shell_cmd, container_image, str(log_path), check=False)
        return code

    def _prepare_video(self, state) -> None:
        if not getattr(state, "input_video", None):
            return
        video_path = Path(state.input_video)
        if not video_path.exists():
            return
        frames_dir = Path(state.run_dir) / "input" / "video_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        state.video_frames_dir = str(frames_dir)

        frame_pattern = frames_dir / "frame_%02d.jpg"
        cmd = [
            "ffmpeg",
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
        return_code = self._run_ffmpeg(cmd, log_path)
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed while extracting video frames; see {log_path}")
        state.extra.setdefault("input_agent", {})["video_frames_dir"] = str(frames_dir)
        state.extra["input_agent"]["frame_count"] = len(list(frames_dir.glob("frame_*.jpg")))

    def _write_perception_result(self, state) -> None:
        perception_data = load_json(getattr(state, "perception_json", None))
        task1_data = load_json(getattr(state, "task1_input_json", None))
        detected_emotion = find_first_value(
            perception_data,
            {"emotion", "detected_emotion", "dominant_emotion", "response_emotion"},
        ) or find_first_value(
            task1_data,
            {"emotion", "detected_emotion", "dominant_emotion", "response_emotion"},
        )
        asr_text = find_first_value(
            perception_data,
            {"asr_text", "transcript", "text", "recognized_text"},
        ) or find_first_value(
            task1_data,
            {"asr_text", "transcript", "text", "recognized_text"},
        )
        payload = {
            "schema": 1,
            "agent": "InputAgent",
            "input_wav": state.input_wav,
            "input_video": getattr(state, "input_video", None),
            "video_frames_dir": getattr(state, "video_frames_dir", None),
            "perception_json": getattr(state, "perception_json", None),
            "task1_input_json": getattr(state, "task1_input_json", None),
            "detected_emotion": detected_emotion,
            "asr_text": asr_text,
            "source_files": {
                "audio": state.input_wav,
                "video": getattr(state, "input_video", None),
                "perception_json": getattr(state, "perception_json", None),
                "task1_input_json": getattr(state, "task1_input_json", None),
            },
        }
        out_path = Path(state.run_dir) / "input" / "perception_result.json"
        save_json(out_path, payload)
        state.perception_result_json = str(out_path)
        state.extra.setdefault("input_agent", {})["perception_result_json"] = str(out_path)

    def run(self, state, run_stage) -> None:
        run_stage("input_agent", lambda: self._prepare_video(state))
        run_stage("perception", lambda: self.perception_tool.run(state))
        self._write_perception_result(state)
