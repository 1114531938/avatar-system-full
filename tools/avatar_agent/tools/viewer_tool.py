from __future__ import annotations

import os
import shlex

from shell_runner import build_apptainer_exec_command, run_bash, run_bash_detached


class ViewerTool:
    def __init__(self, config: dict):
        self.config = config

    def build_command(self, state) -> str:
        p = self.config["paths"]

        gaussian_root = p["gaussian_root"]
        gaussian_venv = p["gaussian_venv"]
        container_image = p["gaussian_container_image"]

        if not state.point_cloud_path or not os.path.exists(state.point_cloud_path):
            raise FileNotFoundError(f"point cloud missing: {state.point_cloud_path}")
        if not state.flame_motion_npz or not os.path.exists(state.flame_motion_npz):
            raise FileNotFoundError(f"flame motion npz missing: {state.flame_motion_npz}")

        q = shlex.quote
        inner_cmd = (
            f"cd {q(gaussian_root)} && "
            f"export OMP_NUM_THREADS=1 && "
            f"export OPENBLAS_NUM_THREADS=1 && "
            f"export MKL_NUM_THREADS=1 && "
            f"export NUMEXPR_NUM_THREADS=1 && "
            f"{q(os.path.join(gaussian_venv, 'bin/python'))} local_viewer.py "
            f"--point_path {q(state.point_cloud_path)} "
            f"--motion_path {q(state.flame_motion_npz)}"
        )

        return build_apptainer_exec_command(inner_cmd, container_image)

    def run(self, state, launch: bool = True):
        cmd = self.build_command(state)
        state.viewer_command = cmd

        log_file = os.path.join(state.log_dir, "viewer.log")
        detached = bool(self.config["runtime"].get("viewer_detached", True))

        if launch:
            if detached:
                pid = run_bash_detached(cmd, log_file=log_file)
                state.viewer_pid = pid
                state.viewer_started = True
            else:
                run_bash(cmd, log_file=log_file)
                state.viewer_started = True
        else:
            state.viewer_started = False
