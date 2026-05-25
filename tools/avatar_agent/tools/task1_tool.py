from __future__ import annotations

import json
import os
import shlex
import urllib.error
import urllib.request

from shell_runner import run_bash_in_container


def _find_first_value(obj, keys):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, (str, int, float)):
                return str(v)
        for v in obj.values():
            found = _find_first_value(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_first_value(item, keys)
            if found is not None:
                return found
    return None


class Task1Tool:
    def __init__(self, config: dict):
        self.config = config

    def _load_reply_state(self, state, out_json: str) -> None:
        if not os.path.exists(out_json):
            raise FileNotFoundError(f"Task1 output not found: {out_json}")

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        reply_text = _find_first_value(
            data,
            keys={"reply_text", "response", "reply", "generated_text", "replyText"},
        )
        reply_style = _find_first_value(
            data,
            keys={"response_emotion", "style", "emotion", "reply_style"},
        )

        state.task1_reply_json = out_json
        state.reply_text = reply_text
        state.reply_style = reply_style

    def _try_worker(self, state, out_json: str) -> bool:
        worker_cfg = self.config.get("avamerg_worker", {})
        if not bool(worker_cfg.get("enabled", False)):
            return False

        url = str(worker_cfg.get("url", "http://127.0.0.1:8789")).rstrip("/")
        timeout = float(worker_cfg.get("timeout", 240))
        log_path = os.path.join(state.log_dir, "task1.log")

        def _log(message: str) -> None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")

        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                health = json.loads(resp.read().decode("utf-8"))
            if not health.get("ok"):
                _log(f"[avamerg_worker] unhealthy response: {health}")
                return False
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            _log(f"[avamerg_worker] unavailable, falling back to subprocess: {exc}")
            return False

        payload = json.dumps(
            {
                "input_json": state.task1_input_json,
                "out_json": out_json,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{url}/infer",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            _log(f"[avamerg_worker] infer failed, falling back to subprocess: {exc}")
            return False

        if not result.get("ok"):
            _log(f"[avamerg_worker] infer returned error, falling back: {result}")
            return False
        if not os.path.exists(out_json):
            _log(f"[avamerg_worker] output missing after infer, falling back: {out_json}")
            return False

        _log(f"[avamerg_worker] inferred via worker: {json.dumps(result, ensure_ascii=False)[:4000]}")
        return True

    def run(self, state):
        p = self.config["paths"]
        root = p["avamerg_root"]
        venv = p["avamerg_venv"]
        container_image = p["gaussian_container_image"]
        out_dir = p["avamerg_reply_out_dir"]
        py = os.path.join(venv, "bin/python")

        if not os.path.lexists(py):
            raise FileNotFoundError(f"AvaMERG python not found: {py}")

        os.makedirs(out_dir, exist_ok=True)
        out_json = os.path.join(out_dir, f"{state.base_name}_task1_reply.json")

        if self._try_worker(state, out_json):
            self._load_reply_state(state, out_json)
            return

        q = shlex.quote
        cmd = f"""
        export PYTHONPATH={q(os.path.join(root, "merg_code"))}:{q(root)}:$PYTHONPATH
        cd {q(root)}
        {q(py)} run_task1_infer.py \
          --input_json {q(state.task1_input_json)} \
          --out_json {q(out_json)}
        """
        run_bash_in_container(cmd, container_image, os.path.join(state.log_dir, "task1.log"))

        self._load_reply_state(state, out_json)
