import argparse
import json
from pathlib import Path
import whisper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--model", type=str, default="small")
    parser.add_argument("--language", type=str, default="auto")
    args = parser.parse_args()

    wav_path = Path(args.wav)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = whisper.load_model(args.model)
    language_arg = args.language.strip()
    language = None if language_arg.lower() in {"", "auto", "detect"} else language_arg
    result = model.transcribe(str(wav_path), language=language)

    ret = {
        "utterance_id": wav_path.stem,
        "wav_path": str(wav_path.resolve()),
        "text": result["text"].strip(),
        "asr_source": f"whisper-{args.model}",
        "language": result.get("language") or language_arg or "auto"
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ret, f, ensure_ascii=False, indent=2)

    print(json.dumps(ret, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
