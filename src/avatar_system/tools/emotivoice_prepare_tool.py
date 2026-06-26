from __future__ import annotations

import os
import shlex

from avatar_system.pipeline.shell_runner import run_bash_in_container


class EmotiVoicePrepareTool:
    def __init__(self, config: dict):
        self.config = config

    def run(self, state):
        p = self.config["paths"]
        env_cfg = self.config["env"]
        tts_cfg = self.config["tts"]

        root = p["emotivoice_root"]
        venv = p["emotivoice_venv"]
        container_image = p["gaussian_container_image"]
        input_dir = p["emotivoice_input_dir"]
        frontend_py = p["emotivoice_frontend_py"]
        converter_script = p["json_to_emotivoice_script"]
        py = os.path.join(venv, "bin/python")

        if not os.path.lexists(py):
            raise FileNotFoundError(f"EmotiVoice python not found: {py}")

        os.makedirs(input_dir, exist_ok=True)
        out_txt = os.path.join(input_dir, f"{state.base_name}_emotivoice.txt")

        q = shlex.quote
        wrap_flag = " --wrap_sos_eos" if bool(tts_cfg.get("wrap_sos_eos", True)) else ""

        cmd = f"""
        cd {q(root)}
        export NLTK_DATA={q(env_cfg["NLTK_DATA"])}
        export OPENAI_API_KEY={q(os.environ.get("OPENAI_API_KEY", ""))}
        export OPENAI_BASE_URL={q(os.environ.get("OPENAI_BASE_URL", env_cfg.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")))}
        export LLM_MODEL={q(os.environ.get("LLM_MODEL", env_cfg.get("LLM_MODEL", "openai/gpt-oss-120b:free")))}
        {q(py)} {q(converter_script)} \
          --input_json {q(state.task1_reply_json)} \
          --output_txt {q(out_txt)} \
          --frontend_py {q(frontend_py)} \
          --speaker_id {q(str(tts_cfg["speaker_id"]))} \
          --prompt_mode {q(str(tts_cfg["prompt_mode"]))}{wrap_flag}
        """
        run_bash_in_container(cmd, container_image, os.path.join(state.log_dir, "emotivoice_prepare.log"))

        if not os.path.exists(out_txt):
            raise FileNotFoundError(f"EmotiVoice input txt not found: {out_txt}")

        state.emotivoice_txt = out_txt
