from __future__ import annotations


class RenderAgent:
    """Marker wrapper for the existing TTS, motion, and Gaussian render stages."""

    def __init__(self, config: dict):
        self.config = config

    def before_render(self, state) -> None:
        state.extra.setdefault("render_agent", {})["status"] = "running"

    def after_render(self, state) -> None:
        state.extra.setdefault("render_agent", {})["status"] = "done"
        state.extra["render_agent"]["output_video"] = state.output_video
