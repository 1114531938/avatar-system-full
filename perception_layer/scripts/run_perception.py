import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_default_ser_model_path() -> str:
    """
    优先使用项目目录下已下载好的 emotion2vec_plus_seed。
    如果本地缓存里没有，就退回远程模型名。
    """
    candidates = [
        "/scratch/e1554543/avatar_system_full/cache/modelscope/models/iic/emotion2vec_plus_seed",
        "/scratch/e1554543/avatar_system_full/cache/modelscope/hub/models/iic/emotion2vec_plus_seed",
        "/scratch/e1554543/avatar_system_full/cache/modelscope/hub/iic/emotion2vec_plus_seed",
    ]

    for p in candidates:
        try:
            if Path(p).exists():
                return p
        except PermissionError:
            continue

    return "iic/emotion2vec_plus_seed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)   # 这里传输出目录
    parser.add_argument("--model", type=str, default="small")
    parser.add_argument("--language", type=str, default="Chinese")
    parser.add_argument("--speaker_id", type=str, default="user")
    parser.add_argument(
        "--ser_model",
        type=str,
        default=None,
        help="emotion2vec 模型路径或模型名；不传则自动查找默认路径"
    )
    args = parser.parse_args()

    wav_path = Path(args.wav).resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"wav not found: {wav_path}")

    # 把 out_json 当成输出目录使用
    out_dir = Path(args.out_json).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 根据 wav 文件名自动生成输出 json 名称
    out_path = out_dir / f"{wav_path.stem}_perception.json"

    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    asr_json = tmp_dir / f"{wav_path.stem}_asr.json"
    ser_json = tmp_dir / f"{wav_path.stem}_ser.json"

    ser_model = args.ser_model if args.ser_model else resolve_default_ser_model_path()

    print(f"[INFO] WAV: {wav_path}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] SER model: {ser_model}")

    py = sys.executable

    subprocess.run([
        py, "scripts/run_asr.py",
        "--wav", str(wav_path),
        "--out_json", str(asr_json),
        "--model", args.model,
        "--language", args.language
    ], check=True)

    subprocess.run([
        py, "scripts/run_ser.py",
        "--wav", str(wav_path),
        "--out_json", str(ser_json),
        "--model", str(ser_model)
    ], check=True)

    asr_ret = load_json(asr_json)
    ser_ret = load_json(ser_json)

    merged = {
        "utterance_id": wav_path.stem,
        "wav_path": str(wav_path),
        "text": asr_ret["text"],
        "emotion": ser_ret["emotion"],
        "emotion_score": ser_ret["emotion_score"],
        "raw_emotion": ser_ret.get("raw_emotion", ""),
        "asr_source": asr_ret["asr_source"],
        "emotion_source": ser_ret["emotion_source"],
        "speaker_id": args.speaker_id,
        "language": "zh"
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {out_path}")
    print(json.dumps(merged, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
