from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from manifest_utils import save_json, write_manifest
from pipeline.config import load_pipeline_config
from state import PipelineState
from orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="End-to-end avatar pipeline orchestrator")
    parser.add_argument("--input_wav", required=True, help="Path to input wav")
    parser.add_argument("--input_video", default=None, help="Optional user video path for Booth multimodal input")
    parser.add_argument("--avatar_id", default="306", help="Avatar media id, e.g. 306")
    parser.add_argument("--config", default="pipeline_config.yaml", help="YAML config path")
    parser.add_argument("--run_id", default=None, help="Optional run id")
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Run all stages except actually launching viewer; print viewer command only",
    )
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Skip the LLM conversion in perception/build_task1_input; useful when OPENAI_API_KEY is not set.",
    )
    parser.add_argument(
        "--no_video_export",
        action="store_true",
        help="Collect wav/npz/manifest but skip final mp4 export.",
    )
    parser.add_argument(
        "--tts_speaker_id",
        default=None,
        help="Override EmotiVoice speaker id from the YAML config.",
    )
    parser.add_argument("--background", default=None, help="Optional Booth background id")
    parser.add_argument("--session_id", default=None, help="Optional Booth session id")
    parser.add_argument("--turn_id", default=None, help="Optional Booth turn id")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    if args.no_llm:
        config.setdefault("perception", {})["no_llm"] = True
    if args.no_video_export:
        config.setdefault("runtime", {})["export_video"] = False
    if args.tts_speaker_id:
        config.setdefault("tts", {})["speaker_id"] = str(args.tts_speaker_id).strip()

    input_wav = os.path.abspath(args.input_wav)
    if not os.path.exists(input_wav):
        raise FileNotFoundError(f"input wav not found: {input_wav}")

    base_name = Path(input_wav).stem
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"{base_name}_{now}"

    run_root = config["runtime"]["run_root"]
    run_dir = os.path.join(run_root, run_id)
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    state = PipelineState(
        input_wav=input_wav,
        input_video=os.path.abspath(args.input_video) if args.input_video else None,
        avatar_id=str(args.avatar_id),
        base_name=base_name,
        run_id=run_id,
        run_dir=run_dir,
        log_dir=log_dir,
        prepare_only=args.prepare_only,
        launch_viewer=(not args.prepare_only),
    )
    state.tts_speaker_id = str(config.get("tts", {}).get("speaker_id", ""))
    state.selected_avatar_id = str(args.avatar_id)
    state.selected_tts_speaker_id = state.tts_speaker_id
    state.background = args.background
    state.session_id = args.session_id
    state.turn_id = args.turn_id

    state_path = os.path.join(run_dir, "state.json")
    def save_state(s: PipelineState):
        save_json(state_path, s.to_dict())

    save_state(state)

    orchestrator = Orchestrator(config)
    final_state = orchestrator.run(state, save_state=save_state)

    manifest = write_manifest(final_state)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
