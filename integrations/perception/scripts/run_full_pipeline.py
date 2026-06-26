import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def first_value(v, default=None):
    if isinstance(v, list):
        return v[0] if v else default
    return default if v is None else v


def resolve_perception_output_path(out_arg: str, wav_path: Path) -> Path:
    out_input = Path(out_arg)

    if out_input.suffix.lower() == ".json":
        out_path = out_input.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    out_dir = out_input.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{wav_path.stem}_perception.json"


def resolve_task1_output_path(
    out_arg: str,
    wav_path: Path,
    task1_obj: Optional[dict] = None
) -> Path:
    out_input = Path(out_arg)

    if out_input.suffix.lower() == ".json":
        out_path = out_input.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    out_dir = out_input.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    utt_id = None
    if task1_obj:
        dia_ids = task1_obj.get("dia_ids")
        if isinstance(dia_ids, list) and dia_ids:
            utt_id = dia_ids[0]

    if not utt_id:
        utt_id = wav_path.stem

    return out_dir / f"{utt_id}_task1_input.json"


def normalize_task1_obj(task1_obj: dict) -> dict:
    """
    把 build_task1_input.py 输出的 task1 json 修成更兼容 AvaMERG 的格式。
    """
    if not isinstance(task1_obj, dict):
        raise ValueError("task1 json must be an object")

    dia_ids = task1_obj.get("dia_ids")
    if not isinstance(dia_ids, list) or not dia_ids:
        task1_obj["dia_ids"] = ["custom_sample"]

    conversations = task1_obj.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        task1_obj["conversations"] = [{
            "dialogue_history": [],
            "response": "",
            "coe": {
                "speaker_emotion": "neutral",
                "event_scenario": "",
                "emotion_cause": "",
                "goal_to_response": ""
            }
        }]
        conversations = task1_obj["conversations"]

    conv = conversations[0]
    conv.setdefault("response", "")

    # coe / chain_of_empathy 双写，兼容不同脚本
    if "coe" in conv and "chain_of_empathy" not in conv:
        conv["chain_of_empathy"] = conv["coe"]
    elif "chain_of_empathy" in conv and "coe" not in conv:
        conv["coe"] = conv["chain_of_empathy"]
    elif "coe" not in conv and "chain_of_empathy" not in conv:
        conv["coe"] = {
            "speaker_emotion": "neutral",
            "event_scenario": "",
            "emotion_cause": "",
            "goal_to_response": ""
        }
        conv["chain_of_empathy"] = conv["coe"]

    # 统一 response_* 字段
    response_age = first_value(task1_obj.get("response_age"), None)
    response_gender = first_value(task1_obj.get("response_gender"), None)
    response_timbre = first_value(task1_obj.get("response_timbre"), None)
    response_emotion = first_value(task1_obj.get("response_emotion"), None)

    response_profile = task1_obj.get("response_profile")
    if isinstance(response_profile, list):
        response_profile = response_profile[0] if response_profile else {}
    if not isinstance(response_profile, dict):
        response_profile = {}

    age = response_profile.get("age") or response_age or "young"
    gender = response_profile.get("gender") or response_gender or "female"
    timbre = response_profile.get("timbre") or response_timbre or "mid"

    fixed_profile = {
        "age": age,
        "gender": gender,
        "timbre": timbre
    }
    if "ID" in response_profile:
        fixed_profile["ID"] = response_profile["ID"]

    task1_obj["response_profile"] = fixed_profile
    task1_obj["response_age"] = age
    task1_obj["response_gender"] = gender
    task1_obj["response_timbre"] = timbre
    task1_obj["response_emotion"] = response_emotion or "warm"

    return task1_obj


def normalize_task1_file(task1_path: Path):
    obj = load_json(task1_path)
    obj = normalize_task1_obj(obj)
    save_json(obj, task1_path)
    return obj


def main():
    parser = argparse.ArgumentParser()

    # step 1: perception
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--perception_out", type=str, required=True)
    parser.add_argument("--model", type=str, default="small")
    parser.add_argument("--language", type=str, default="auto")
    parser.add_argument("--speaker_id", type=str, default="user")
    parser.add_argument(
        "--ser_model",
        type=str,
        default=os.environ.get("SER_MODEL", "iic/emotion2vec_plus_seed"),
        help="SER 模型名或模型目录；默认使用 iic/emotion2vec_plus_seed"
    )

    # step 2: task1 builder
    parser.add_argument("--task1_out", type=str, required=True)
    parser.add_argument(
        "--llm_model",
        type=str,
        default=os.environ.get("LLM_MODEL", "openai/gpt-oss-120b:free")
    )
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    )
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "")
    )
    parser.add_argument("--no_llm", action="store_true")

    # optional step 3: directly run AvaMERG infer
    parser.add_argument("--infer_script", type=str, default=None)
    parser.add_argument("--infer_out", type=str, default=None)
    parser.add_argument("--infer_workdir", type=str, default=None)

    args = parser.parse_args()

    wav_path = Path(args.wav).resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    perception_out_path = resolve_perception_output_path(args.perception_out, wav_path)

    py = sys.executable

    cmd1 = [
        py, "scripts/run_perception.py",
        "--wav", str(wav_path),
        "--out_json", str(args.perception_out),
        "--model", args.model,
        "--language", args.language,
        "--speaker_id", args.speaker_id,
        "--ser_model", args.ser_model,
    ]

    print("[Step 1/3] Running perception...")
    print(" ".join(cmd1))
    subprocess.run(cmd1, check=True)

    cmd2 = [
        py, "scripts/build_task1_input.py",
        "--input_json", str(perception_out_path),
        "--out_json", str(args.task1_out),
        "--model", args.llm_model,
        "--base_url", args.llm_base_url,
    ]

    if args.no_llm:
        cmd2.append("--no_llm")
    else:
        if not args.llm_api_key:
            raise ValueError("OPENAI_API_KEY is empty. Please export it or pass --llm_api_key.")
        cmd2 += ["--api_key", args.llm_api_key]

    print("[Step 2/3] Building task1 input...")
    if "--api_key" in cmd2:
        safe_cmd2 = cmd2.copy()
        safe_cmd2[safe_cmd2.index("--api_key") + 1] = "***API_KEY_HIDDEN***"
        print(" ".join(safe_cmd2))
    else:
        print(" ".join(cmd2))

    subprocess.run(cmd2, check=True)

    provisional_task1_path = resolve_task1_output_path(args.task1_out, wav_path)

    print("[Step 3/3] Normalizing task1 json for AvaMERG compatibility...")
    task1_obj = normalize_task1_file(provisional_task1_path)

    final_task1_path = resolve_task1_output_path(args.task1_out, wav_path, task1_obj)

    print("\nDone.")
    print(f"Perception JSON: {perception_out_path}")
    print(f"Task1 JSON:      {final_task1_path}")

    if args.infer_script and args.infer_out:
        infer_script = Path(args.infer_script).resolve()
        if not infer_script.exists():
            raise FileNotFoundError(f"infer script not found: {infer_script}")

        cmd3 = [
            py, str(infer_script),
            "--input_json", str(final_task1_path),
            "--out_json", str(Path(args.infer_out).resolve())
        ]

        print("\n[Optional Infer] Running AvaMERG inference...")
        print(" ".join(cmd3))

        subprocess.run(
            cmd3,
            check=True,
            cwd=args.infer_workdir if args.infer_workdir else None
        )

        print(f"Infer JSON:      {Path(args.infer_out).resolve()}")


if __name__ == "__main__":
    main()
