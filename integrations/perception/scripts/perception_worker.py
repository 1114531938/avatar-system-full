from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import whisper
from funasr import AutoModel

from build_task1_input import (  # noqa: E402
    build_fallback_task1,
    call_llm,
    finalize_task1_result,
    get_output_path,
    save_json,
    validate_task1_schema,
)
from run_full_pipeline import (  # noqa: E402
    normalize_task1_file,
    resolve_perception_output_path,
    resolve_task1_output_path,
)
from run_ser import (  # noqa: E402
    extract_label_score,
    normalize_label,
    resolve_default_model_path,
)


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class PerceptionEngine:
    def __init__(self, asr_model: str, ser_model: str) -> None:
        self.lock = threading.Lock()
        self.asr_models: dict[str, Any] = {}
        self.ser_models: dict[str, Any] = {}
        self.default_asr_model = asr_model
        self.default_ser_model = ser_model or resolve_default_model_path()
        self._get_asr_model(self.default_asr_model)
        self._get_ser_model(self.default_ser_model)

    def _get_asr_model(self, model_name: str) -> Any:
        if model_name not in self.asr_models:
            print(f"[perception_worker] loading Whisper model: {model_name}", flush=True)
            self.asr_models[model_name] = whisper.load_model(model_name)
        return self.asr_models[model_name]

    def _get_ser_model(self, model_name: str) -> Any:
        model_path = model_name or resolve_default_model_path()
        if model_path not in self.ser_models:
            print(f"[perception_worker] loading SER model: {model_path}", flush=True)
            self.ser_models[model_path] = AutoModel(model=model_path, disable_update=True)
        return self.ser_models[model_path]

    def _run_asr(self, wav_path: Path, model_name: str, language: str) -> dict[str, Any]:
        model = self._get_asr_model(model_name)
        with self.lock:
            result = model.transcribe(str(wav_path), language=language)
        return {
            "utterance_id": wav_path.stem,
            "wav_path": str(wav_path.resolve()),
            "text": result["text"].strip(),
            "asr_source": f"whisper-{model_name}",
            "language": language,
        }

    def _run_ser(self, wav_path: Path, model_name: str) -> dict[str, Any]:
        model_path = model_name or resolve_default_model_path()
        model = self._get_ser_model(model_path)
        with self.lock:
            res = model.generate(
                input=str(wav_path),
                granularity="utterance",
                extract_embedding=False,
            )
        raw_emotion, emotion_score = extract_label_score(res)
        emotion = normalize_label(raw_emotion)
        return {
            "utterance_id": wav_path.stem,
            "wav_path": str(wav_path.resolve()),
            "emotion": emotion,
            "emotion_score": emotion_score,
            "raw_emotion": raw_emotion,
            "emotion_source": str(model_path),
        }

    def infer_file(
        self,
        wav: str,
        perception_out: str,
        task1_out: str,
        model: str,
        language: str,
        speaker_id: str,
        ser_model: str,
        no_llm: bool,
        llm_model: str,
        llm_base_url: str,
        llm_api_key: str,
    ) -> dict[str, Any]:
        wav_path = Path(wav).resolve()
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV not found: {wav_path}")

        perception_out_path = resolve_perception_output_path(perception_out, wav_path)
        asr_ret = self._run_asr(wav_path, model or self.default_asr_model, language)
        ser_ret = self._run_ser(wav_path, ser_model or self.default_ser_model)

        perception_obj = {
            "utterance_id": wav_path.stem,
            "wav_path": str(wav_path),
            "text": asr_ret["text"],
            "emotion": ser_ret["emotion"],
            "emotion_score": ser_ret["emotion_score"],
            "raw_emotion": ser_ret.get("raw_emotion", ""),
            "asr_source": asr_ret["asr_source"],
            "emotion_source": ser_ret["emotion_source"],
            "speaker_id": speaker_id,
            "language": "zh",
        }
        save_json(perception_obj, perception_out_path)

        provisional_task1_path = get_output_path(task1_out, perception_out_path, perception_obj)
        if no_llm:
            task1_obj = build_fallback_task1(perception_obj=perception_obj)
        else:
            if not llm_api_key:
                raise ValueError("OPENAI_API_KEY is empty. Please export it or pass llm_api_key.")
            try:
                task1_obj = call_llm(
                    perception_obj=perception_obj,
                    model=llm_model,
                    base_url=llm_base_url,
                    api_key=llm_api_key,
                )
                task1_obj = finalize_task1_result(task1_obj)
                validate_task1_schema(task1_obj)
            except Exception as exc:
                print(f"[perception_worker] WARN: LLM conversion failed, fallback will be used. Error: {exc}", flush=True)
                task1_obj = build_fallback_task1(perception_obj=perception_obj)

        save_json(task1_obj, provisional_task1_path)
        task1_obj = normalize_task1_file(provisional_task1_path)
        final_task1_path = resolve_task1_output_path(task1_out, wav_path, task1_obj)

        return {
            "wav": str(wav_path),
            "perception_json": str(perception_out_path),
            "task1_input_json": str(provisional_task1_path),
            "reported_task1_json": str(final_task1_path),
            "asr_model": model or self.default_asr_model,
            "ser_model": ser_model or self.default_ser_model,
        }


class PerceptionRequestHandler(BaseHTTPRequestHandler):
    engine: PerceptionEngine

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("[perception_worker] " + format % args + "\n")
        sys.stdout.flush()

    def do_GET(self) -> None:
        if self.path != "/health":
            _json_response(self, 404, {"ok": False, "error": "not found"})
            return
        _json_response(
            self,
            200,
            {
                "ok": True,
                "asr_models": list(self.engine.asr_models.keys()),
                "ser_models": list(self.engine.ser_models.keys()),
            },
        )

    def do_POST(self) -> None:
        if self.path != "/run":
            _json_response(self, 404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.engine.infer_file(
                wav=str(payload["wav"]),
                perception_out=str(payload["perception_out"]),
                task1_out=str(payload["task1_out"]),
                model=str(payload.get("model", "")),
                language=str(payload.get("language", "Chinese")),
                speaker_id=str(payload.get("speaker_id", "user")),
                ser_model=str(payload.get("ser_model", "")),
                no_llm=bool(payload.get("no_llm", False)),
                llm_model=str(payload.get("llm_model", "openai/gpt-oss-120b:free")),
                llm_base_url=str(payload.get("llm_base_url", "https://openrouter.ai/api/v1")),
                llm_api_key=str(payload.get("llm_api_key", "")),
            )
            _json_response(self, 200, {"ok": True, **result})
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-lived perception worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--model", default="small")
    parser.add_argument("--ser_model", default="")
    args = parser.parse_args()

    print("[perception_worker] loading perception models...", flush=True)
    PerceptionRequestHandler.engine = PerceptionEngine(
        asr_model=args.model,
        ser_model=args.ser_model,
    )
    print(
        f"[perception_worker] ready on http://{args.host}:{args.port} "
        f"asr={list(PerceptionRequestHandler.engine.asr_models.keys())} "
        f"ser={list(PerceptionRequestHandler.engine.ser_models.keys())}",
        flush=True,
    )
    server = ThreadingHTTPServer((args.host, args.port), PerceptionRequestHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
