from __future__ import annotations

import os
import shutil
import shlex
import json
import urllib.error
import urllib.request

from shell_runner import run_bash_in_container, latest_match


class EmotiVoiceTTSTool:
    def __init__(self, config: dict):
        self.config = config

    def _try_worker(self, state, out_wav: str) -> bool:
        worker_cfg = self.config.get("tts_worker", {})
        if not bool(worker_cfg.get("enabled", False)):
            return False

        url = str(worker_cfg.get("url", "http://127.0.0.1:8788")).rstrip("/")
        timeout = float(worker_cfg.get("timeout", 120))
        log_path = os.path.join(state.log_dir, "emotivoice_tts.log")

        def _log(message: str) -> None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")

        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                health = json.loads(resp.read().decode("utf-8"))
            if not health.get("ok"):
                _log(f"[tts_worker] unhealthy response: {health}")
                return False
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            _log(f"[tts_worker] unavailable, falling back to subprocess: {exc}")
            return False

        payload = json.dumps(
            {
                "test_file": state.emotivoice_txt,
                "output_wav": out_wav,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{url}/synthesize",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            _log(f"[tts_worker] synthesize failed, falling back to subprocess: {exc}")
            return False

        if not result.get("ok"):
            _log(f"[tts_worker] synthesize returned error, falling back to subprocess: {result}")
            return False
        if not os.path.exists(out_wav):
            _log(f"[tts_worker] output missing after synthesize, falling back: {out_wav}")
            return False

        _log(f"[tts_worker] synthesized via worker: {json.dumps(result, ensure_ascii=False)}")
        return True

    def run(self, state):
        p = self.config["paths"]
        root = p["emotivoice_root"]
        venv = p["emotivoice_venv"]
        container_image = p["gaussian_container_image"]
        wav_glob = p["emotivoice_output_wav_glob"]
        py = os.path.join(venv, "bin/python")

        if not os.path.lexists(py):
            raise FileNotFoundError(f"EmotiVoice python not found: {py}")

        run_output_dir = os.path.join(state.run_dir, "outputs")
        os.makedirs(run_output_dir, exist_ok=True)
        run_wav_path = os.path.join(run_output_dir, "reply.wav")

        if self._try_worker(state, run_wav_path):
            state.reply_wav = run_wav_path
            return

        q = shlex.quote
        cmd = f"""
        cd {q(root)}
        {q(py)} inference_am_vocoder_joint.py \
          --logdir prompt_tts_open_source_joint \
          --config_folder config/joint \
          --checkpoint g_00140000 \
          --test_file {q(state.emotivoice_txt)}
        """
        run_bash_in_container(cmd, container_image, os.path.join(state.log_dir, "emotivoice_tts.log"))

        wav_path = latest_match(wav_glob)
        if not wav_path or not os.path.exists(wav_path):
            raise FileNotFoundError(f"No wav found from EmotiVoice with glob: {wav_glob}")

        shutil.copy2(wav_path, run_wav_path)

        state.reply_wav = run_wav_path
