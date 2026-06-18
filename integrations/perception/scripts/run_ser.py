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
    if not label:
        return "neutral"

    label = str(label).strip()

    # 处理 "开心/happy" 这种格式
    if "/" in label:
        parts = label.split("/")
        if len(parts) >= 2:
            label = parts[-1].strip()

    label = label.lower()
    return RAW_TO_TASK.get(label, "neutral")


def extract_label_score(res):
    """
    从 emotion2vec/FunASR 返回结果中提取 top1 标签和分数。
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

        # labels/scores 都是 list，取最大概率对应标签
        if isinstance(labels, list) and isinstance(scores, list) and labels and scores:
            if len(labels) != len(scores):
                raise ValueError(
                    f"labels/scores length mismatch: {len(labels)} vs {len(scores)}"
                )
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            raw_label = labels[best_idx]
            score = float(scores[best_idx])

        # 单个标签
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


def resolve_default_model_path() -> str:
    """
    优先使用项目目录下已下载好的本地模型。
    找不到时再退回远程模型名。
    """
    candidates = [
        # 项目里已经有的本地模型目录（优先）
        "/scratch/e1554543/avatar_system_full/integrations/perception/models/emotion2vec",
        "/scratch/e1554543/avatar_system_full/runtime/cache/modelscope/hub/models/iic/emotion2vec_plus_seed",
        "/scratch/e1554543/avatar_system_full/runtime/cache/modelscope/hub/iic/emotion2vec_plus_seed",
    ]

    for p in candidates:
        try:
            if Path(p).exists():
                return p
        except PermissionError:
            continue

    # 如果本地没找到，再退回远程名
    return "iic/emotion2vec_plus_seed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="emotion2vec 模型路径；不传则自动查找本地默认路径",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav).resolve()
    out_path = Path(args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not wav_path.exists():
        raise FileNotFoundError(f"wav not found: {wav_path}")

    model_path = args.model if args.model else resolve_default_model_path()

    print(f"[INFO] Using SER model: {model_path}")

    model = AutoModel(
        model=model_path,
        disable_update=True,
    )

    res = model.generate(
        input=str(wav_path),
        granularity="utterance",
        extract_embedding=False,
    )

    raw_emotion, emotion_score = extract_label_score(res)
    emotion = normalize_label(raw_emotion)

    ret = {
        "utterance_id": wav_path.stem,
        "wav_path": str(wav_path),
        "emotion": emotion,
        "emotion_score": emotion_score,
        "raw_emotion": raw_emotion,
        "emotion_source": str(model_path),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ret, f, ensure_ascii=False, indent=2)

    print(json.dumps(ret, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
