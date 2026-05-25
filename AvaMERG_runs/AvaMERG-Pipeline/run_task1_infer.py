import sys
import os
import json
from pathlib import Path
from typing import Any, Dict

# 自动以当前脚本所在目录作为 ROOT，避免硬编码旧路径
ROOT = Path(__file__).resolve().parent
MERG_CODE = ROOT / "merg_code"

sys.path.insert(0, str(MERG_CODE))
sys.path.insert(0, str(ROOT))

import torch
from config import load_config
from model import load_model


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if torch.is_tensor(obj):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    return str(obj)


def build_args(
    save_path: str | None = None,
    log_path: str | None = None,
) -> Dict[str, Any]:
    """
    构造推理配置。
    优先使用传入参数；否则使用当前 ROOT 下的默认路径。
    """
    default_save_path = str(ROOT / "outputs" / "ckpt_debug8" / "0")
    default_log_path = str(ROOT / "outputs" / "tb_logs")

    args = {
        "model": "merg",
        "mode": "test",
        "audio_path": str(ROOT / "merg_data" / "train" / "audio_flat"),
        "video_path": str(ROOT / "merg_data" / "train" / "video_flat"),
        "save_path": save_path or default_save_path,
        "log_path": log_path or default_log_path,
        "assets_path": str(ROOT / "assets"),
        "local_rank": 0,
        "data_path": str(ROOT / "merg_data"),
        "debug_n": 1,
        "epochs": 1,
        "world_size": 1,
        "total_steps": 1,
        "dschf": None,
    }

    Path(args["log_path"]).mkdir(parents=True, exist_ok=True)
    args = load_config(args)
    return args


def build_single_batch(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    按 task1 输入 json 构造成单条 batch。
    """
    batch = {
        "dia_ids": sample["dia_ids"],
        "conversations": sample["conversations"],
        "response_age": sample["response_age"],
        "response_emotion": sample["response_emotion"],
        "response_gender": sample["response_gender"],
        "response_timbre": sample["response_timbre"],
        "response_profile": sample["response_profile"],
    }
    return batch


def attach_batch_to_args(cfg: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    给下游 generate/predict 提供尽可能多的兼容入口。
    不同仓库版本可能会从不同 key 里取 batch。
    """
    cfg["infer_batch"] = batch
    cfg["batch"] = batch
    cfg["sample_batch"] = batch
    cfg["input_batch"] = batch
    cfg["test_batch"] = batch
    return cfg


def try_infer(agent: Any, batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    依次尝试常见推理接口。
    注意：当前这个 agent.predict() 是不接收 batch 参数的。
    """
    tried = []
    errors = {}

    # 1) predict() —— 当前仓库最可能可用
    if hasattr(agent, "predict") and callable(getattr(agent, "predict")):
        tried.append("predict()")
        try:
            ret = agent.predict()
            return {"method": "predict()", "result": ret}
        except Exception as e:
            errors["predict()"] = repr(e)

    # 2) ds_engine.generate(agent.args)
    if hasattr(agent, "ds_engine") and hasattr(agent.ds_engine, "generate"):
        tried.append("ds_engine.generate(agent.args)")
        try:
            ret = agent.ds_engine.generate(agent.args)
            return {"method": "ds_engine.generate(agent.args)", "result": ret}
        except Exception as e:
            errors["ds_engine.generate(agent.args)"] = repr(e)

    # 3) generate(batch)
    if hasattr(agent, "generate") and callable(getattr(agent, "generate")):
        tried.append("generate(batch)")
        try:
            ret = agent.generate(batch)
            return {"method": "generate(batch)", "result": ret}
        except Exception as e:
            errors["generate(batch)"] = repr(e)

    # 4) test_model(batch)
    if hasattr(agent, "test_model") and callable(getattr(agent, "test_model")):
        tried.append("test_model(batch)")
        try:
            ret = agent.test_model(batch)
            return {"method": "test_model(batch)", "result": ret}
        except Exception as e:
            errors["test_model(batch)"] = repr(e)

    # 5) inference(batch)
    if hasattr(agent, "inference") and callable(getattr(agent, "inference")):
        tried.append("inference(batch)")
        try:
            ret = agent.inference(batch)
            return {"method": "inference(batch)", "result": ret}
        except Exception as e:
            errors["inference(batch)"] = repr(e)

    public_methods = [
        name for name in dir(agent)
        if not name.startswith("_") and callable(getattr(agent, name))
    ]

    return {
        "method": None,
        "error": "No usable inference method succeeded.",
        "tried": tried,
        "errors": errors,
        "public_methods": public_methods,
    }


def normalize_output(ret: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "used_method": ret.get("method"),
        "raw_result": to_jsonable(ret.get("result", None)),
    }

    if "error" in ret:
        out["error"] = ret["error"]
    if "tried" in ret:
        out["tried"] = ret["tried"]
    if "errors" in ret:
        out["errors"] = ret["errors"]
    if "public_methods" in ret:
        out["public_methods"] = ret["public_methods"]

    result = ret.get("result", None)

    if isinstance(result, str):
        out["reply_text"] = result

    elif isinstance(result, dict):
        for key in ["reply_text", "response", "generated_text", "text", "pred", "output"]:
            if key in result and isinstance(result[key], str):
                out["reply_text"] = result[key]
                break

    elif isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], str):
            out["reply_text"] = result[0]
        elif isinstance(result[0], dict):
            for key in ["reply_text", "response", "generated_text", "text", "pred", "output"]:
                if key in result[0] and isinstance(result[0][key], str):
                    out["reply_text"] = result[0][key]
                    break

    return out


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to sample*_task1_input.json"
    )
    parser.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="Where to save inference result"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Checkpoint path for MERG LoRA/adapter weights"
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="TensorBoard log dir"
    )
    args = parser.parse_args()

    sample = load_json(args.input_json)
    batch = build_single_batch(sample)

    print("[1] loading config...")
    cfg = build_args(save_path=args.save_path, log_path=args.log_path)
    cfg = attach_batch_to_args(cfg, batch)

    print("[2] loading model...")
    agent = load_model(cfg)
    print("[ok] model loaded")

    with torch.no_grad():
        ret = try_infer(agent, batch)

    out = {
        "input_json": str(Path(args.input_json).resolve()),
        "batch_preview": to_jsonable(batch),
    }
    out.update(normalize_output(ret))

    save_json(out, args.out_json)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()