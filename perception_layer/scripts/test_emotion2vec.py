import argparse
import json
from pathlib import Path

from funasr import AutoModel


RAW_TO_TASK = {
    "angry": "angry",
    "disgusted": "angry",
    "fearful": "anxious",
    "happy": "happy",
    "neutral": "neutral",
    "other": "neutral",
    "sad": "sad",
    "surprised": "neutral",
    "unknown": "neutral",
    "<unk>": "neutral",
}


def normalize_label(label: str) -> str:
    """
    把 emotion2vec 的原始标签统一映射到你系统后续使用的较粗粒度标签。
    支持：
    - happy
    - 开心/happy
    - <unk>
    """
    if not label:
        return "neutral"

    label = str(label).strip()

    # 处理 "开心/happy" 这种格式，优先取斜杠后的英文
    if "/" in label:
        parts = label.split("/")
        if len(parts) >= 2:
            label = parts[-1].strip()

    label = label.lower()
    return RAW_TO_TASK.get(label, "neutral")


def extract_label_score(res):
    """
    从 emotion2vec/FunASR 返回结果中提取 top1 标签和分数。
    关键点：不是取 labels[0]，而是取 scores 最大对应的标签。
    """
    raw_label = None
    score = None

    if isinstance(res, list) and len(res) > 0:
        item = res[0]
    else:
        item = res

    if isinstance(item, dict):
        labels = item.get("labels") or item.get("label")
        scores = item.get("scores") or item.get("score")

        # 情况1：labels/scores 都是 list，做 argmax
        if isinstance(labels, list) and isinstance(scores, list) and labels and scores:
            if len(labels) != len(scores):
                raise ValueError(
                    f"labels/scores length mismatch: {len(labels)} vs {len(scores)}"
                )
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            raw_label = labels[best_idx]
            score = float(scores[best_idx])

        # 情况2：单标签单分数
        elif isinstance(labels, str):
            raw_label = labels
            if isinstance(scores, (int, float)):
                score = float(scores)

        # 兜底
        if raw_label is None and isinstance(item.get("text"), str):
            raw_label = item["text"]

        if raw_label is None and isinstance(item.get("result"), str):
            raw_label = item["result"]

    if raw_label is None:
        raw_label = "unknown"
    if score is None:
        score = 0.0

    return str(raw_label), float(score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--model", type=str, default="iic/emotion2vec_plus_seed")
    args = parser.parse_args()

    wav_path = Path(args.wav).resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"wav not found: {wav_path}")

    print(f"[INFO] loading model: {args.model}")
    model = AutoModel(model=args.model)

    print(f"[INFO] running inference on: {wav_path}")
    res = model.generate(
        input=str(wav_path),
        granularity="utterance",
        extract_embedding=False,
    )

    raw_emotion, emotion_score = extract_label_score(res)
    emotion = normalize_label(raw_emotion)

    out = {
        "utterance_id": wav_path.stem,
        "wav_path": str(wav_path),
        "emotion": emotion,
        "emotion_score": emotion_score,
        "raw_emotion": raw_emotion,
        "emotion_source": args.model,
        "raw_result": res,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()