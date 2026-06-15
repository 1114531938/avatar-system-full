from __future__ import annotations

import os
import shutil
import shlex
import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from pipeline.config import project_path
from shell_runner import latest_match, run_bash


class DEEPTalkTool:
    def __init__(self, config: dict):
        self.config = config

    def _try_worker(self, state, out_npy: str) -> bool:
        worker_cfg = self.config.get("deeptalk_worker", {})
        if not bool(worker_cfg.get("enabled", False)):
            return False

        url = str(worker_cfg.get("url", "http://127.0.0.1:8790")).rstrip("/")
        timeout = float(worker_cfg.get("timeout", 180))
        log_path = os.path.join(state.log_dir, "deeptalk.log")

        def _log(message: str) -> None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")

        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                health = json.loads(resp.read().decode("utf-8"))
            if not health.get("ok"):
                _log(f"[deeptalk_worker] unhealthy response: {health}")
                return False
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            _log(f"[deeptalk_worker] unavailable, falling back to subprocess: {exc}")
            return False

        payload = json.dumps(
            {
                "audio_path": state.reply_wav,
                "output_npy": out_npy,
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
            _log(f"[deeptalk_worker] infer failed, falling back to subprocess: {exc}")
            return False

        if not result.get("ok"):
            _log(f"[deeptalk_worker] infer returned error, falling back: {result}")
            return False
        if not os.path.exists(out_npy):
            _log(f"[deeptalk_worker] output missing after infer, falling back: {out_npy}")
            return False

        _log(f"[deeptalk_worker] inferred via worker: {json.dumps(result, ensure_ascii=False)}")
        return True

    def run(self, state):
        p = self.config["paths"]
        env_cfg = self.config.get("env", {})
        root = p["deeptalk_root"]
        venv = p["deeptalk_venv"]
        configured_out_npy = p["deeptalk_output_npy"]
        out_dir = os.path.dirname(configured_out_npy)
        expected_out_npy = os.path.join(out_dir, f"{Path(state.reply_wav).stem}.npy")
        py = os.path.join(venv, "bin/python")

        if not os.path.exists(py):
            raise FileNotFoundError(f"DEEPTalk python not found: {py}")

        run_output_dir = os.path.join(state.run_dir, "outputs")
        os.makedirs(run_output_dir, exist_ok=True)
        run_npy_path = os.path.join(run_output_dir, "deeptalk.npy")

        if self._try_worker(state, run_npy_path):
            state.deeptalk_npy = run_npy_path
            return

        backup_paths = {}
        for out_npy in {configured_out_npy, expected_out_npy}:
            if not os.path.exists(out_npy):
                continue
            backup_dir = os.path.join(state.run_dir, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_npy = os.path.join(backup_dir, f"{Path(out_npy).stem}_{stamp}.npy")
            os.replace(out_npy, backup_npy)
            backup_paths[out_npy] = backup_npy

        q = shlex.quote
        infer_script = os.path.join(root, "infer_motion_only.py")
        cmd = f"""
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export TOKENIZERS_PARALLELISM=false
        export HF_HOME={q(env_cfg.get("HF_HOME", str(project_path("cache", "hf"))))}
        export XDG_CACHE_HOME={q(env_cfg.get("XDG_CACHE_HOME", str(project_path("cache", "xdg"))))}
        export MODELSCOPE_CACHE={q(env_cfg.get("MODELSCOPE_CACHE", str(project_path("cache", "modelscope"))))}
        export PATH={q(str(project_path("tools", "ffmpeg-git-20240629-amd64-static")))}:$PATH
        mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$MODELSCOPE_CACHE"
        cd {q(root)}
        {q(py)} {q(infer_script)} --audio_path {q(state.reply_wav)} --output_npy {q(run_npy_path)}
        """
        return_code, _, stderr = run_bash(cmd, os.path.join(state.log_dir, "deeptalk.log"), check=False)

        out_npy = None
        for candidate in (run_npy_path, expected_out_npy, configured_out_npy):
            if os.path.exists(candidate):
                out_npy = candidate
                break
        if out_npy is None:
            out_npy = latest_match(os.path.join(out_dir, "*.npy"))

        if not out_npy or not os.path.exists(out_npy):
            for original_path, backup_npy in backup_paths.items():
                if os.path.exists(backup_npy):
                    os.replace(backup_npy, original_path)
            if return_code != 0:
                raise RuntimeError(f"DEEPTalk failed before writing params: {stderr}")
            raise FileNotFoundError(f"DEEPTalk output not found in {out_dir}")

        if os.path.abspath(out_npy) != os.path.abspath(run_npy_path):
            shutil.copy2(out_npy, run_npy_path)

        state.deeptalk_npy = run_npy_path
