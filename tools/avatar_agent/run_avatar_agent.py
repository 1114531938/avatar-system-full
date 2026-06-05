from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

from state import PipelineState
from orchestrator import Orchestrator


def save_json(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
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
    manifest_path = os.path.join(run_dir, "manifest.json")

    def save_state(s: PipelineState):
        save_json(state_path, s.to_dict())

    save_state(state)

    orchestrator = Orchestrator(config)
    final_state = orchestrator.run(state, save_state=save_state)

    manifest = {
        "run_id": final_state.run_id,
        "input_wav": final_state.input_wav,
        "avatar_id": final_state.avatar_id,
        "task1_reply_json": final_state.task1_reply_json,
        "input_video": final_state.input_video,
        "video_frames_dir": final_state.video_frames_dir,
        "plan_json": final_state.plan_json,
        "selected_avatar_id": final_state.selected_avatar_id,
        "selected_tts_speaker_id": final_state.selected_tts_speaker_id,
        "background": final_state.background,
        "session_id": final_state.session_id,
        "turn_id": final_state.turn_id,
        "reply_text": final_state.reply_text,
        "tts_speaker_id": final_state.tts_speaker_id,
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
