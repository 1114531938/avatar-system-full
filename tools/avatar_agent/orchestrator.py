from __future__ import annotations

from tools.perception_tool import PerceptionTool
from tools.task1_tool import Task1Tool
from tools.emotivoice_prepare_tool import EmotiVoicePrepareTool
from tools.emotivoice_tts_tool import EmotiVoiceTTSTool
from tools.deeptalk_tool import DEEPTalkTool
from tools.flame_merge_tool import FlameMergeTool
from tools.viewer_tool import ViewerTool
from tools.artifact_export_tool import ArtifactExportTool
from agents import InputAgent, PlanAgent, RenderAgent


class Orchestrator:
    def __init__(self, config: dict):
        self.config = config
        self.input_agent = InputAgent(config)
        self.plan_agent = PlanAgent(config)
        self.render_agent = RenderAgent(config)
        self.steps = [
            ("perception", PerceptionTool(config)),
            ("task1", Task1Tool(config)),
            ("emotivoice_prepare", EmotiVoicePrepareTool(config)),
            ("emotivoice_tts", EmotiVoiceTTSTool(config)),
            ("deeptalk", DEEPTalkTool(config)),
            ("flame_merge", FlameMergeTool(config)),
        ]
        self.viewer_tool = ViewerTool(config)
        self.artifact_export_tool = ArtifactExportTool(config)

    def run(self, state, save_state=None):
        def _save():
            if save_state is not None:
                save_state(state)

        try:
            total_steps = len(self.steps) + 5
            step_index = 0
            if "input_agent" not in state.finished_stages:
                step_index += 1
                print(f"\n[{step_index}/{total_steps}] input_agent starting...", flush=True)
                state.current_stage = "input_agent"
                _save()
                self.input_agent.run(state)
                state.finished_stages.append("input_agent")
                _save()
                print(f"[{step_index}/{total_steps}] input_agent done.", flush=True)

            for stage_name, tool in self.steps:
                if stage_name in state.finished_stages:
                    step_index += 1
                    continue

                step_index += 1
                print(f"\n[{step_index}/{total_steps}] {stage_name} starting...", flush=True)
                state.current_stage = stage_name
                _save()

                tool.run(state)

                state.finished_stages.append(stage_name)
                _save()
                print(f"[{step_index}/{total_steps}] {stage_name} done.", flush=True)

                if stage_name == "task1" and "plan_agent" not in state.finished_stages:
                    step_index += 1
                    print(f"\n[{step_index}/{total_steps}] plan_agent starting...", flush=True)
                    state.current_stage = "plan_agent"
                    _save()
                    self.plan_agent.run(state)
                    state.finished_stages.append("plan_agent")
                    _save()
                    print(f"[{step_index}/{total_steps}] plan_agent done.", flush=True)

                if stage_name == "emotivoice_prepare" and "render_agent" not in state.finished_stages:
                    step_index += 1
                    print(f"\n[{step_index}/{total_steps}] render_agent starting...", flush=True)
                    state.current_stage = "render_agent"
                    _save()
                    self.render_agent.before_render(state)
                    state.finished_stages.append("render_agent")
                    _save()
                    print(f"[{step_index}/{total_steps}] render_agent done.", flush=True)

            step_index += 1
            print(f"\n[{step_index}/{total_steps}] viewer starting...", flush=True)
            state.current_stage = "viewer"
            _save()

            self.viewer_tool.run(
                state,
                launch=(not state.prepare_only and state.launch_viewer),
            )

            state.finished_stages.append("viewer")
            _save()
            print(f"[{step_index}/{total_steps}] viewer done.", flush=True)

            step_index += 1
            print(f"\n[{step_index}/{total_steps}] artifact_export starting...", flush=True)
            state.current_stage = "artifact_export"
            _save()
            self.artifact_export_tool.run(
                state,
                export_video=bool(self.config.get("runtime", {}).get("export_video", True)),
            )
            self.render_agent.after_render(state)

            state.finished_stages.append("artifact_export")
            state.current_stage = "done"
            _save()
            print(f"[{step_index}/{total_steps}] artifact_export done.", flush=True)
            return state

        except Exception as e:
            state.failed_stage = state.current_stage
            state.error = str(e)
            _save()
            raise
