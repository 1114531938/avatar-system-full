import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_reply_text(obj: Dict[str, Any]) -> str:
    # 优先读顶层 reply_text
    text = obj.get("reply_text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    # 再读 raw_result 里的常见字段
    raw = obj.get("raw_result", {})
    if isinstance(raw, dict):
        for key in ["reply_text", "generated_text", "response", "text", "pred", "output"]:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    raise ValueError("No valid reply_text found in json.")


def looks_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text or "")


def extract_tts_text(obj: Dict[str, Any]) -> str | None:
    for key in ["tts_text", "spoken_text", "english_tts_text"]:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raw = obj.get("raw_result", {})
    if isinstance(raw, dict):
        for key in ["tts_text", "spoken_text", "english_tts_text"]:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def translate_to_english(text: str, model: str, base_url: str, api_key: str) -> str:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is empty; cannot translate TTS text.")

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Translate the user's empathetic reply into natural spoken English for TTS. "
                    "Preserve the meaning and emotional warmth. Return only the English sentence."
                ),
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"TTS translation request failed: {exc}") from exc

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"TTS translation response has no choices: {data}")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    translated = message.get("content") if isinstance(message, dict) else None
    if not isinstance(translated, str) or not translated.strip():
        raise RuntimeError(f"TTS translation response has empty content: {data}")

    translated = translated.strip().strip("\"'")
    if looks_chinese(translated):
        raise RuntimeError(f"TTS translation still contains Chinese: {translated}")
    return translated


def resolve_tts_text(raw_text: str, obj: Dict[str, Any], args: argparse.Namespace) -> str:
    explicit = extract_tts_text(obj)
    if explicit:
        return explicit

    if args.tts_language != "en" or not looks_chinese(raw_text):
        return raw_text

    try:
        return translate_to_english(
            raw_text,
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
        )
    except RuntimeError as exc:
        print(f"[tts_translate] {exc}; falling back to original reply_text", file=sys.stderr)
        return raw_text


def extract_batch_preview(obj: Dict[str, Any]) -> Dict[str, Any]:
    batch = obj.get("batch_preview")
    if isinstance(batch, dict):
        return batch
    raise ValueError("No batch_preview found in json.")


def extract_goal_to_response(batch: Dict[str, Any]) -> str:
    conversations = batch.get("conversations", [])
    if not conversations:
        return ""

    conv = conversations[0]
    coe = conv.get("chain_of_empathy") or conv.get("coe") or {}
    if not isinstance(coe, dict):
        return ""

    value = coe.get("goal_to_response", "")
    return value.strip() if isinstance(value, str) else ""


def extract_response_emotion(batch: Dict[str, Any]) -> str:
    value = batch.get("response_emotion", "")
    return value.strip() if isinstance(value, str) else ""


def extract_speaker_id(batch: Dict[str, Any], override: str | None) -> str:
    if override:
        return str(override)

    profile = batch.get("response_profile", {})
    if isinstance(profile, dict):
        if "ID" in profile and str(profile["ID"]).strip():
            return str(profile["ID"]).strip()

    if "speaker_id" in batch and str(batch["speaker_id"]).strip():
        return str(batch["speaker_id"]).strip()

    return "8051"


def map_emotion_to_emotivoice_label(emotion: str) -> str:
    """
    把 task1 里的 response_emotion 粗略映射到 EmotiVoice 常见风格标签。
    你也可以直接不用这个映射，而是改成纯 goal_to_response。
    """
    if not emotion:
        return "Neutral"

    e = emotion.strip().lower()
    mapping = {
        "happy": "Happy",
        "joyful": "Happy",
        "cheerful": "Happy",
        "sad": "Sad",
        "angry": "Angry",
        "neutral": "Neutral",
        "calm": "Calm",
        "gentle": "Calm",
        "warm": "Calm",
        "surprised": "Surprised",
        "anxious": "Sad",
    }
    return mapping.get(e, emotion)


def build_style_prompt(
    batch: Dict[str, Any],
    prompt_mode: str,
    custom_prompt: str | None = None,
) -> str:
    if custom_prompt:
        return custom_prompt.strip()

    goal = extract_goal_to_response(batch)
    emotion = extract_response_emotion(batch)

    if prompt_mode == "goal":
        return goal or map_emotion_to_emotivoice_label(emotion) or "Neutral"

    if prompt_mode == "emotion":
        return map_emotion_to_emotivoice_label(emotion) or "Neutral"

    if prompt_mode == "goal_plus_emotion":
        emo = map_emotion_to_emotivoice_label(emotion) or "Neutral"
        if goal:
            return f"{emo}; {goal}"
        return emo

    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def run_frontend(frontend_py: str, text: str) -> str:
    """
    调用 EmotiVoice 的 frontend.py，把原始文本转成音素序列。
    假设 frontend.py 的用法是：
        python frontend.py input.txt
    并把结果打印到 stdout
    """
    frontend_path = Path(frontend_py)
    if not frontend_path.exists():
        raise FileNotFoundError(f"frontend.py not found: {frontend_path}")

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        f.write(text.strip() + "\n")
        temp_input = f.name

    try:
        proc = subprocess.run(
            [sys.executable, str(frontend_path), temp_input],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    finally:
        Path(temp_input).unlink(missing_ok=True)

    phonemes = " ".join(proc.stdout.strip().split())
    if not phonemes:
        raise RuntimeError(
            "frontend.py returned empty phoneme sequence.\n"
            f"stderr:\n{proc.stderr}"
        )
    return phonemes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help="上一步推理结果 json")
    parser.add_argument("--output_txt", type=str, required=True, help="输出的 EmotiVoice 输入文件")
    parser.add_argument(
        "--frontend_py",
        type=str,
        default="/scratch/e1554543/avatar_system_full/integrations/emotivoice/frontend.py",
        help="EmotiVoice frontend.py 路径",
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        default=None,
        help="手动指定 speaker id；不传则自动从 json 里取，没有就用 8051",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="goal",
        choices=["goal", "emotion", "goal_plus_emotion"],
        help="第二列 style prompt 的来源：goal / emotion / goal_plus_emotion",
    )
    parser.add_argument(
        "--custom_prompt",
        type=str,
        default=None,
        help="手动覆盖第二列提示词",
    )
    parser.add_argument(
        "--wrap_sos_eos",
        action="store_true",
        help="是否把音素包装成 <sos/eos> phonemes <sos/eos>",
    )
    parser.add_argument(
        "--tts_language",
        type=str,
        default="en",
        choices=["original", "en"],
        help="TTS 使用的文本语言：original 使用 reply_text 原文；en 会把中文 reply_text 翻译成英文后再送 TTS",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=os.environ.get("LLM_MODEL", "openai/gpt-oss-120b:free"),
        help="用于 TTS 文本翻译的 OpenAI-compatible model",
    )
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        help="用于 TTS 文本翻译的 OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="用于 TTS 文本翻译的 API key",
    )
    args = parser.parse_args()

    obj = load_json(args.input_json)
    batch = extract_batch_preview(obj)

    raw_text = extract_reply_text(obj)
    tts_text = resolve_tts_text(raw_text, obj, args)
    if tts_text and tts_text != obj.get("tts_text"):
        obj["tts_text"] = tts_text
        with open(args.input_json, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    speaker_id = extract_speaker_id(batch, args.speaker_id)
    style_prompt = build_style_prompt(
        batch=batch,
        prompt_mode=args.prompt_mode,
        custom_prompt=args.custom_prompt,
    )

    phoneme_seq = run_frontend(args.frontend_py, tts_text)

    if args.wrap_sos_eos:
        phoneme_field = f"<sos/eos> {phoneme_seq} <sos/eos>"
    else:
        phoneme_field = phoneme_seq

    output_line = f"{speaker_id}|{style_prompt}|{phoneme_field}|{tts_text}"

    out_path = Path(args.output_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_line + "\n")

    print("Saved:", out_path)
    print(output_line)


if __name__ == "__main__":
    main()
