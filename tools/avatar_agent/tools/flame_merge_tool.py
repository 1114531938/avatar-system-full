from __future__ import annotations

import os
import shlex

from shell_runner import run_bash


class FlameMergeTool:
    def __init__(self, config: dict):
        self.config = config

    def run(self, state):
        p = self.config["paths"]
        merge_cfg = self.config["merge"]

        gaussian_root = p["gaussian_root"]
        merge_script = p["deeptalk_merge_script"]
        deeptalk_py = os.path.join(p["deeptalk_venv"], "bin/python")

        media_dir = os.path.join(gaussian_root, "media", str(state.avatar_id))
        template_npz = os.path.join(media_dir, "flame_param.npz")
        point_cloud_path = os.path.join(media_dir, "point_cloud.ply")
        out_npz = os.path.join(media_dir, f"flame_param_from_{state.base_name}_deeptalk.npz")

        if not os.path.exists(template_npz):
            raise FileNotFoundError(f"Template npz not found: {template_npz}")
        if not os.path.exists(point_cloud_path):
            raise FileNotFoundError(f"Point cloud not found: {point_cloud_path}")
        if not os.path.exists(deeptalk_py):
            raise FileNotFoundError(f"DEEPTalk python not found: {deeptalk_py}")

        q = shlex.quote
        zero_translation_flag = " --zero-translation" if bool(merge_cfg.get("zero_translation", True)) else ""
        extra_flags = []
        for flag_name, cfg_name in [
            ("jaw-open-scale", "jaw_open_scale"),
            ("jaw-side-scale", "jaw_side_scale"),
            ("jaw-twist-scale", "jaw_twist_scale"),
            ("mouth-open-bias", "mouth_open_bias"),
            ("temporal-shift-frames", "temporal_shift_frames"),
            ("smooth-window", "smooth_window"),
        ]:
            if cfg_name in merge_cfg and merge_cfg[cfg_name] is not None:
                extra_flags.append(f" --{flag_name} {q(str(merge_cfg[cfg_name]))}")
        extra_flags = "".join(extra_flags)

        cmd = f"""
        {q(deeptalk_py)} {q(merge_script)} \
          --deeptalk_motion {q(state.deeptalk_npy)} \
          --template {q(template_npz)} \
          --out {q(out_npz)} \
          --expr-scale {q(str(merge_cfg.get("expr_scale", 1.0)))} \
          --jaw-scale {q(str(merge_cfg.get("jaw_scale", 1.0)))}{extra_flags}{zero_translation_flag}
        """
        run_bash(cmd, os.path.join(state.log_dir, "flame_merge.log"))

        if not os.path.exists(out_npz):
            raise FileNotFoundError(f"Final flame motion npz not found: {out_npz}")

        state.template_npz = template_npz
        state.point_cloud_path = point_cloud_path
        state.flame_motion_npz = out_npz
