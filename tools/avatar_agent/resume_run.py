from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from orchestrator import Orchestrator
from state import PipelineState


def save_json(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume an avatar_agent run from state.json")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--config", default="pipeline_config.yaml")
    parser.add_argument("--from_stage", default=None, help="Stage to remove from finished_stages before resuming")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--no_video_export", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.no_video_export:
        config.setdefault("runtime", {})["export_video"] = False

    state_path = os.path.join(args.run_dir, "state.json")
    manifest_path = os.path.join(args.run_dir, "manifest.json")
    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    known_fields = PipelineState.__dataclass_fields__
    state_kwargs = {k: v for k, v in data.items() if k in known_fields}
    state = PipelineState(**state_kwargs)

    if args.prepare_only:
        state.prepare_only = True
        state.launch_viewer = False

    if args.from_stage:
        if args.from_stage in state.finished_stages:
            idx = state.finished_stages.index(args.from_stage)
            state.finished_stages = state.finished_stages[:idx]
        state.current_stage = args.from_stage
        state.failed_stage = None
        state.error = None

    def save_state(s: PipelineState):
        save_json(state_path, s.to_dict())

    final_state = Orchestrator(config).run(state, save_state=save_state)
    manifest = {
        "run_id": final_state.run_id,
        "input_wav": final_state.input_wav,
        "avatar_id": final_state.avatar_id,
        "task1_reply_json": final_state.task1_reply_json,
        "reply_text": final_state.reply_text,
        "reply_wav": final_state.reply_wav,
        "deeptalk_npy": final_state.deeptalk_npy,
        "flame_motion_npz": final_state.flame_motion_npz,
        "point_cloud_path": final_state.point_cloud_path,
        "viewer_command": final_state.viewer_command,
        "viewer_started": final_state.viewer_started,
        "viewer_pid": final_state.viewer_pid,
        "artifact_dir": final_state.artifact_dir,
        "artifact_reply_wav": final_state.artifact_reply_wav,
        "artifact_enhanced_reply_wav": final_state.artifact_enhanced_reply_wav,
        "artifact_flame_motion_npz": final_state.artifact_flame_motion_npz,
        "output_video": final_state.output_video,
        "output_white_model_video": final_state.output_white_model_video,
        "video_export_command": final_state.video_export_command,
        "video_export_error": final_state.video_export_error,
        "finished_stages": final_state.finished_stages,
        "failed_stage": final_state.failed_stage,
        "error": final_state.error,
        "run_dir": final_state.run_dir,
        "log_dir": final_state.log_dir,
    }
    save_json(manifest_path, manifest)
    if final_state.artifact_dir:
        save_json(os.path.join(final_state.artifact_dir, "manifest.json"), manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
