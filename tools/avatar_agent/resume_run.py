from __future__ import annotations

import argparse
import json
import os

from manifest_utils import save_json, write_manifest
from orchestrator import Orchestrator
from pipeline.config import load_pipeline_config
from state import PipelineState


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume an avatar_agent run from state.json")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--config", default="pipeline_config.yaml")
    parser.add_argument("--from_stage", default=None, help="Stage to remove from finished_stages before resuming")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--no_video_export", action="store_true")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    if args.no_video_export:
        config.setdefault("runtime", {})["export_video"] = False

    state_path = os.path.join(args.run_dir, "state.json")
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
    manifest = write_manifest(final_state)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
