from __future__ import annotations

import json
import os
import shlex
import urllib.error
import urllib.request
import subprocess

from avatar_system.pipeline.shell_runner import run_bash_in_container

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


def _latest_user_utterance(batch) -> str:
    if not isinstance(batch, dict):
        return ""
    conversations = batch.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return ""
    history = conversations[0].get("dialogue_history") if isinstance(conversations[0], dict) else None
    if not isinstance(history, list):
        return ""
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        text = str(item.get("utterance") or item.get("text") or "").strip()
        if text:
            return text
    return ""


def _fallback_reply_from_batch(batch) -> str:
    utterance = _latest_user_utterance(batch)
    if not utterance:
        return "我刚才没有听清你的具体内容。你可以再说一遍，我会根据你说的话继续回应。"
    if len(utterance) > 80:
        utterance = utterance[:77].rstrip() + "..."
    return f"我听到你刚才说：“{utterance}”。我会先陪你把这部分慢慢说清楚。"


def _english_fallback_reply_from_batch(batch) -> str:
    utterance = _latest_user_utterance(batch)
    if not utterance:
        return "I did not catch the details clearly. Please say that again, and I will respond to what you share."
    if len(utterance) > 90:
        utterance = utterance[:87].rstrip() + "..."
    emotion = "what you are feeling"
    try:
        emotion = str(batch.get("conversations", [{}])[0].get("coe", {}).get("speaker_emotion") or emotion)
    except (AttributeError, IndexError, TypeError):
        pass
    return f"I hear the {emotion} in what you said: \"{utterance}\". Let's stay with that and take it one step at a time."


def _llm_english_reply_from_batch(batch, config: dict | None = None) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return ""

    utterance = _latest_user_utterance(batch)
    if not utterance:
        return ""

    config = config or {}
    model = (
        os.environ.get("TASK1_LLM_MODEL")
        or os.environ.get("LLM_MODEL")
        or str(config.get("task1_llm_model") or "gpt-4o-mini")
    )
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or str(config.get("openai_base_url") or "https://api.openai.com/v1")
    ).rstrip("/")

    emotion = ""
    context = ""
    try:
        conv = batch.get("conversations", [{}])[0]
        coe = conv.get("coe", {}) if isinstance(conv, dict) else {}
        emotion = str(coe.get("speaker_emotion") or "")
        context = str(coe.get("scene") or coe.get("video_summary") or "")
    except (AttributeError, IndexError, TypeError):
        pass

    system = (
        "You are the English dialogue brain for a friendly digital human in a short video call. "
        "Reply in natural, specific English. Be warm and conversational, but do not sound like a therapist "
        "unless the user is clearly discussing feelings. Use one or two concise sentences. "
        "Do not mention that you are an AI, subtitles, ASR, or internal models."
    )
    user = (
        f"User said: {utterance}\n"
        f"Detected emotion: {emotion or 'unknown'}\n"
        f"Visual/context notes: {context or 'none'}\n"
        "Write the digital human's next spoken reply in English."
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.7,
            "max_tokens": 90,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(config.get("task1_llm_timeout", 20))) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return ""

    try:
        text = str(result["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError):
        return ""
    if _looks_chinese(text):
        return ""
    return " ".join(text.split())


def _looks_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(text or ""))


class Task1Tool:
    def __init__(self, config: dict):
        self.config = config

    def _cuda_available(self) -> bool:
        try:
            proc = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return proc.returncode == 0

    def _write_fallback_reply(self, state, out_json: str, reason: str) -> None:
        with open(state.task1_input_json, "r", encoding="utf-8") as f:
            batch = json.load(f)

        reply_text = _fallback_reply_from_batch(batch)
        payload = {
            "input_json": state.task1_input_json,
            "batch_preview": batch,
            "used_method": "fallback",
            "fallback_reason": reason,
            "raw_result": {
                "reply_text": reply_text,
                "generated_text": reply_text,
            },
            "reply_text": reply_text,
            "response_emotion": batch.get("response_emotion", "warm") if isinstance(batch, dict) else "warm",
        }
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        log_path = os.path.join(state.log_dir, "task1.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[task1_fallback] {reason}; wrote {out_json}\n")

    def _load_reply_state(self, state, out_json: str) -> None:
        if not os.path.exists(out_json):
            raise FileNotFoundError(f"Task1 output not found: {out_json}")

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        preferred_text = _find_first_value(
            data,
            keys={"tts_text", "subtitle_text", "spoken_text", "english_tts_text"},
        )
        reply_text = _find_first_value(
            data,
            keys={"reply_text", "response", "reply", "generated_text", "replyText"},
        )
        reply_style = _find_first_value(
            data,
            keys={"response_emotion", "style", "emotion", "reply_style"},
        )

        fixed_chinese_reply = "谢谢你愿意和我说这些。你不用着急，想从哪里开始都可以。我会认真听你慢慢说。"
        if preferred_text and not _looks_chinese(preferred_text):
            reply_text = preferred_text
        elif reply_text == fixed_chinese_reply or _looks_chinese(reply_text or ""):
            batch = data.get("batch_preview")
            if not isinstance(batch, dict):
                try:
                    with open(state.task1_input_json, "r", encoding="utf-8") as f:
                        batch = json.load(f)
                except (OSError, json.JSONDecodeError):
                    batch = {}
            reply_text = _llm_english_reply_from_batch(batch, self.config) or _english_fallback_reply_from_batch(batch)

        if reply_text:
            data["reply_text"] = reply_text
            data["tts_text"] = reply_text
            raw_result = data.get("raw_result")
            if isinstance(raw_result, dict):
                raw_result["reply_text"] = reply_text
                raw_result["generated_text"] = reply_text
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

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
        out_dir = os.path.join(state.run_dir, "task1")
        py = os.path.join(venv, "bin/python")

        if not os.path.lexists(py):
            raise FileNotFoundError(f"AvaMERG python not found: {py}")

        os.makedirs(out_dir, exist_ok=True)
        out_json = os.path.join(out_dir, f"{state.base_name}_task1_reply.json")

        if self._try_worker(state, out_json):
            self._load_reply_state(state, out_json)
            return

        if not self._cuda_available():
            self._write_fallback_reply(state, out_json, "CUDA/GPU is unavailable, skipping AvaMERG 7B inference")
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
