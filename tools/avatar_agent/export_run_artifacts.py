from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

from manifest_utils import save_json
from pipeline.config import load_pipeline_config
from tools.artifact_export_tool import ArtifactExportTool


def main() -> None:
    parser = argparse.ArgumentParser(description="Export artifacts from an existing avatar_agent run")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--config", default="pipeline_config.yaml")
    parser.add_argument("--no_video_export", action="store_true")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)

    run_dir = os.path.abspath(args.run_dir)
    state_path = os.path.join(run_dir, "state.json")
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"state.json not found: {state_path}")

    with open(state_path, "r", encoding="utf-8") as f:
        state_data = json.load(f)

    defaults = {
        "artifact_dir": None,
        "artifact_reply_wav": None,
        "artifact_flame_motion_npz": None,
        "artifact_enhanced_reply_wav": None,
        "output_video": None,
        "output_white_model_video": None,
        "video_export_command": None,
        "video_export_error": None,
    }
    defaults.update(state_data)
    state = SimpleNamespace(**defaults)

    ArtifactExportTool(config).run(state, export_video=(not args.no_video_export))
    state_data.update(vars(state))
    save_json(state_path, state_data)

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    manifest.update(
        {
            "run_id": state.run_id,
            "reply_wav": state.reply_wav,
            "flame_motion_npz": state.flame_motion_npz,
            "artifact_dir": state.artifact_dir,
            "artifact_reply_wav": state.artifact_reply_wav,
            "artifact_enhanced_reply_wav": state.artifact_enhanced_reply_wav,
            "artifact_flame_motion_npz": state.artifact_flame_motion_npz,
            "output_video": state.output_video,
            "output_white_model_video": state.output_white_model_video,
            "video_export_command": state.video_export_command,
            "video_export_error": state.video_export_error,
            "run_dir": state.run_dir,
            "log_dir": state.log_dir,
        }
    )
    save_json(manifest_path, manifest)
    if state.artifact_dir:
        save_json(os.path.join(state.artifact_dir, "manifest.json"), manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
