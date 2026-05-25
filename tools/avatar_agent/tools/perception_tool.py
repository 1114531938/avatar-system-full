from __future__ import annotations

import glob
import json
import os
import shlex
import urllib.error
import urllib.request

from shell_runner import run_bash_in_container, first_existing


class PerceptionTool:
    def __init__(self, config: dict):
        self.config = config

    def _resolve_outputs(self, state, out_dir: str, task1_out_dir: str) -> None:
        base = state.base_name
        task1_candidates = [
            os.path.join(task1_out_dir, f"{base}_task1_input.json"),
            os.path.join(task1_out_dir, f"{base}.json"),
        ]
        task1_path = first_existing(*task1_candidates)
        if not task1_path:
            matches = glob.glob(os.path.join(task1_out_dir, f"*{base}*task1*.json"))
            if not matches:
                matches = glob.glob(os.path.join(task1_out_dir, "*.json"))
            if not matches:
                raise FileNotFoundError(f"Could not find task1 input json in {task1_out_dir}")
            task1_path = max(matches, key=os.path.getmtime)

        perception_candidates = [
            os.path.join(out_dir, f"{base}.json"),
            os.path.join(out_dir, f"{base}_perception.json"),
            os.path.join(out_dir, f"{base}_merged.json"),
        ]
        perception_path = first_existing(*perception_candidates)
        if not perception_path:
            matches = glob.glob(os.path.join(out_dir, f"*{base}*.json"))
            perception_path = max(matches, key=os.path.getmtime) if matches else None

        state.task1_input_json = task1_path
        state.perception_json = perception_path

    def _try_worker(self, state, out_dir: str, task1_out_dir: str) -> bool:
        worker_cfg = self.config.get("perception_worker", {})
        if not bool(worker_cfg.get("enabled", False)):
            return False

        env_cfg = self.config["env"]
        perception_cfg = self.config["perception"]
        url = str(worker_cfg.get("url", "http://127.0.0.1:8791")).rstrip("/")
        timeout = float(worker_cfg.get("timeout", 180))
        log_path = os.path.join(state.log_dir, "perception.log")

        def _log(message: str) -> None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")

        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                health = json.loads(resp.read().decode("utf-8"))
            if not health.get("ok"):
                _log(f"[perception_worker] unhealthy response: {health}")
                return False
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            _log(f"[perception_worker] unavailable, falling back to subprocess: {exc}")
            return False

        payload = json.dumps(
            {
                "wav": state.input_wav,
                "perception_out": out_dir,
                "task1_out": task1_out_dir,
                "model": str(perception_cfg["model"]),
                "language": str(perception_cfg["language"]),
                "speaker_id": str(perception_cfg["speaker_id"]),
                "ser_model": str(perception_cfg["ser_model"]),
                "no_llm": bool(perception_cfg.get("no_llm", False)),
                "llm_model": os.environ.get("LLM_MODEL", env_cfg.get("LLM_MODEL", "openai/gpt-oss-120b:free")),
                "llm_base_url": os.environ.get("OPENAI_BASE_URL", env_cfg.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")),
                "llm_api_key": os.environ.get("OPENAI_API_KEY", ""),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{url}/run",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            _log(f"[perception_worker] run failed, falling back to subprocess: {exc}")
            return False

        if not result.get("ok"):
            _log(f"[perception_worker] run returned error, falling back: {result}")
            return False

        task1_path = result.get("task1_input_json")
        perception_path = result.get("perception_json")
        if not task1_path or not os.path.exists(task1_path):
            _log(f"[perception_worker] task1 output missing after run, falling back: {task1_path}")
            return False

        state.task1_input_json = task1_path
        state.perception_json = perception_path if perception_path and os.path.exists(perception_path) else None
        safe_result = dict(result)
        safe_result.pop("llm_api_key", None)
        _log(f"[perception_worker] ran via worker: {json.dumps(safe_result, ensure_ascii=False)}")
        return True

    def run(self, state):
        p = self.config["paths"]
        env_cfg = self.config["env"]
        perception_cfg = self.config["perception"]

        root = p["perception_root"]
        venv = p["perception_venv"]
        container_image = p["gaussian_container_image"]
        out_dir = p["perception_out_dir"]
        task1_out_dir = p["perception_task1_out_dir"]

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(task1_out_dir, exist_ok=True)

        if self._try_worker(state, out_dir, task1_out_dir):
            return

        py = os.path.join(venv, "bin/python")
        if not os.path.lexists(py):
            raise FileNotFoundError(f"Perception python not found: {py}")

        q = shlex.quote
        no_llm_flag = " --no_llm" if bool(perception_cfg.get("no_llm", False)) else ""
        cmd = f"""
        export HF_HOME={q(env_cfg["HF_HOME"])}
        export XDG_CACHE_HOME={q(env_cfg["XDG_CACHE_HOME"])}
        export MODELSCOPE_CACHE={q(env_cfg["MODELSCOPE_CACHE"])}
        export OPENAI_API_KEY={q(os.environ.get("OPENAI_API_KEY", ""))}
        export OPENAI_BASE_URL={q(os.environ.get("OPENAI_BASE_URL", env_cfg.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")))}
        export LLM_MODEL={q(os.environ.get("LLM_MODEL", env_cfg.get("LLM_MODEL", "openai/gpt-oss-120b:free")))}
        mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$MODELSCOPE_CACHE"
        cd {q(root)}
        {q(py)} scripts/run_full_pipeline.py \
          --wav {q(state.input_wav)} \
          --perception_out {q(out_dir)} \
          --task1_out {q(task1_out_dir)} \
          --model {q(str(perception_cfg["model"]))} \
          --language {q(str(perception_cfg["language"]))} \
          --speaker_id {q(str(perception_cfg["speaker_id"]))} \
          --ser_model {q(str(perception_cfg["ser_model"]))}{no_llm_flag}
        """
        run_bash_in_container(cmd, container_image, os.path.join(state.log_dir, "perception.log"))

        self._resolve_outputs(state, out_dir, task1_out_dir)
