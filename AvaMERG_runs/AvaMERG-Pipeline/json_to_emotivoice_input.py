import argparse
import json
import subprocess
import sys
import tempfile
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
        default="/scratch/e1554543/avatar_system_full/EmotiVoice_runs/repo/frontend.py",
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
    args = parser.parse_args()

    obj = load_json(args.input_json)
    batch = extract_batch_preview(obj)

    raw_text = extract_reply_text(obj)
    speaker_id = extract_speaker_id(batch, args.speaker_id)
    style_prompt = build_style_prompt(
        batch=batch,
        prompt_mode=args.prompt_mode,
        custom_prompt=args.custom_prompt,
    )

    phoneme_seq = run_frontend(args.frontend_py, raw_text)

    if args.wrap_sos_eos:
        phoneme_field = f"<sos/eos> {phoneme_seq} <sos/eos>"
    else:
        phoneme_field = phoneme_seq

    output_line = f"{speaker_id}|{style_prompt}|{phoneme_field}|{raw_text}"

    out_path = Path(args.output_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_line + "\n")

    print("Saved:", out_path)
    print(output_line)


if __name__ == "__main__":
    main()