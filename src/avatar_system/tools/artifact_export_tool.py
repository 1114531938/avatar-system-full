from __future__ import annotations

import os
import shutil
import shlex

from avatar_system.pipeline.config import project_path
from avatar_system.pipeline.shell_runner import build_apptainer_exec_command, run_bash_in_container


class ArtifactExportTool:
    def __init__(self, config: dict):
        self.config = config

    def run(self, state, export_video: bool = True):
        p = self.config["paths"]
        runtime = self.config.get("runtime", {})

        artifact_dir = os.path.join(state.run_dir, "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        state.artifact_dir = artifact_dir

        if state.reply_wav and os.path.exists(state.reply_wav):
            out_wav = os.path.join(artifact_dir, "reply.wav")
            shutil.copy2(state.reply_wav, out_wav)
            state.artifact_reply_wav = out_wav

        if state.flame_motion_npz and os.path.exists(state.flame_motion_npz):
            out_npz = os.path.join(artifact_dir, "flame_motion.npz")
            shutil.copy2(state.flame_motion_npz, out_npz)
            state.artifact_flame_motion_npz = out_npz

        if not export_video:
            return
        if not state.point_cloud_path or not state.flame_motion_npz or not state.reply_wav:
            state.video_export_error = "Missing point cloud, motion npz, or reply wav; skipped video export."
            return

        gaussian_root = p["gaussian_root"]
        container_image = p["gaussian_container_image"]
        gaussian_python = os.path.join(p["gaussian_venv"], "bin/python")
        exporter_py = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "export_gaussian_video.py")
        )
        out_video = os.path.join(artifact_dir, "final_video.mp4")
        enhanced_audio = os.path.join(artifact_dir, "reply_enhanced.wav")
        frames_dir = os.path.join(artifact_dir, "frames")
        ffmpeg = runtime.get(
            "ffmpeg",
            "/usr/bin/ffmpeg",
        )
        audio_filter = runtime.get(
            "audio_filter",
            "highpass=f=70,lowpass=f=7600,loudnorm=I=-16:LRA=11:TP=-1.5,aresample=48000",
        )

        q = shlex.quote
        base_cmd = f"""
        cd {q(gaussian_root)}
        {q(gaussian_python)} {q(exporter_py)} \
          --gaussian_root {q(gaussian_root)} \
          --point_path {q(state.point_cloud_path)} \
          --motion_path {q(state.flame_motion_npz)} \
          --audio_path {q(state.reply_wav)} \
          --out_video {q(out_video)} \
          --frames_dir {q(frames_dir)} \
          --enhanced_audio_out {q(enhanced_audio)} \
          --audio_filter {q(audio_filter)} \
          --render_mode gaussian \
          --fps {q(str(runtime.get("video_fps", 25)))} \
          --width {q(str(runtime.get("video_width", 550)))} \
          --height {q(str(runtime.get("video_height", 802)))} \
          --ffmpeg {q(ffmpeg)}
        """
        state.video_export_command = build_apptainer_exec_command(base_cmd, container_image)

        return_code, _, stderr = run_bash_in_container(
            base_cmd,
            container_image,
            os.path.join(state.log_dir, "artifact_export.log"),
            check=False,
        )
        if return_code != 0:
            state.video_export_error = stderr
            return

        if os.path.exists(out_video):
            state.output_video = out_video
        else:
            state.video_export_error = f"Video export command finished but output was not found: {out_video}"
            return
        if os.path.exists(enhanced_audio):
            state.artifact_enhanced_reply_wav = enhanced_audio

        white_video = os.path.join(artifact_dir, "white_model.mp4")
        if bool(runtime.get("export_white_model_video", True)):
            white_frames_dir = os.path.join(artifact_dir, "white_frames")
            white_cmd = f"""
            cd {q(gaussian_root)}
            {q(gaussian_python)} {q(exporter_py)} \
              --gaussian_root {q(gaussian_root)} \
              --point_path {q(state.point_cloud_path)} \
              --motion_path {q(state.flame_motion_npz)} \
              --audio_path {q(enhanced_audio if os.path.exists(enhanced_audio) else state.reply_wav)} \
              --out_video {q(white_video)} \
              --frames_dir {q(white_frames_dir)} \
              --audio_filter '' \
              --render_mode white_mesh \
              --fps {q(str(runtime.get("video_fps", 25)))} \
              --width {q(str(runtime.get("video_width", 550)))} \
              --height {q(str(runtime.get("video_height", 802)))} \
              --ffmpeg {q(ffmpeg)}
            """
            state.video_export_command += "\n\n" + build_apptainer_exec_command(white_cmd, container_image)
            white_return_code, _, white_stderr = run_bash_in_container(
                white_cmd,
                container_image,
                os.path.join(state.log_dir, "artifact_export.log"),
                check=False,
            )
            if white_return_code == 0 and os.path.exists(white_video):
                state.output_white_model_video = white_video
            elif white_return_code != 0:
                state.video_export_error = f"White model export failed, final_video is still available:\n{white_stderr}"
