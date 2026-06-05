from __future__ import annotations

import json
from pathlib import Path


class PlanAgent:
    """Record the user-facing reply plan and selected render persona."""

    def __init__(self, config: dict):
        self.config = config

    def run(self, state) -> None:
        selected_avatar_id = getattr(state, "selected_avatar_id", None) or state.avatar_id
        selected_tts_speaker_id = getattr(state, "selected_tts_speaker_id", None) or getattr(state, "tts_speaker_id", None)
        plan = {
            "schema": 1,
            "strategy": "explainable_coarse_tags",
            "selected_avatar_id": selected_avatar_id,
            "selected_tts_speaker_id": selected_tts_speaker_id,
            "background": getattr(state, "background", None),
            "perception_json": getattr(state, "perception_json", None),
            "task1_input_json": getattr(state, "task1_input_json", None),
            "reason": "Plan Agent uses non-sensitive demo metadata and existing perception output.",
        }
        plan_path = Path(state.run_dir) / "plan_agent" / "plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        state.plan_json = str(plan_path)
        state.selected_avatar_id = str(selected_avatar_id) if selected_avatar_id else None
        state.selected_tts_speaker_id = str(selected_tts_speaker_id) if selected_tts_speaker_id else None
        state.extra.setdefault("plan_agent", {}).update(plan)
