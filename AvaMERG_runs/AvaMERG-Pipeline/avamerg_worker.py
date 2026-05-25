from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
MERG_CODE = ROOT / "merg_code"
sys.path.insert(0, str(MERG_CODE))
sys.path.insert(0, str(ROOT))

from run_task1_infer import (  # noqa: E402
    attach_batch_to_args,
    build_args,
    build_single_batch,
    load_json,
    normalize_output,
    save_json,
    to_jsonable,
    try_infer,
)
from model import load_model  # noqa: E402


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class AvaMERGEngine:
    def __init__(self, save_path: str | None = None, log_path: str | None = None) -> None:
        self.lock = threading.Lock()
        self.cfg = build_args(save_path=save_path, log_path=log_path)
        self.agent = load_model(self.cfg)
        self.device = str(getattr(self.agent, "device", "unknown"))

    def infer_file(self, input_json: str, out_json: str) -> dict[str, Any]:
        sample = load_json(input_json)
        batch = build_single_batch(sample)

        with self.lock:
            attach_batch_to_args(self.cfg, batch)
            # Keep agent.args pointed at the same mutable config, but update defensively.
            attach_batch_to_args(self.agent.args, batch)
            with torch.no_grad():
                ret = try_infer(self.agent, batch)

        out = {
            "input_json": str(Path(input_json).resolve()),
            "batch_preview": to_jsonable(batch),
        }
        out.update(normalize_output(ret))
        save_json(out, out_json)
        return out


class AvaMERGRequestHandler(BaseHTTPRequestHandler):
    engine: AvaMERGEngine

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("[avamerg_worker] " + format % args + "\n")
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
                "device": self.engine.device,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/infer":
            _json_response(self, 404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            input_json = str(payload["input_json"])
            out_json = str(payload["out_json"])
            result = self.engine.infer_file(input_json, out_json)
            _json_response(self, 200, {"ok": True, "out_json": out_json, "result": result})
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-lived AvaMERG Task1 worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8789)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    print("[avamerg_worker] loading AvaMERG model...", flush=True)
    AvaMERGRequestHandler.engine = AvaMERGEngine(
        save_path=args.save_path,
        log_path=args.log_path,
    )
    print(
        f"[avamerg_worker] ready on http://{args.host}:{args.port} "
        f"device={AvaMERGRequestHandler.engine.device}",
        flush=True,
    )
    server = ThreadingHTTPServer((args.host, args.port), AvaMERGRequestHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
