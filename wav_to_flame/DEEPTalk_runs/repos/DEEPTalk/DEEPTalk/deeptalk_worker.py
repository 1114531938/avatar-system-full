from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

from demo import MEAD_ACTOR_DICT, pad_audio_to_match_quantfactor
from DEE.get_DEE import get_DEE_from_json
from DEE.utils.utils import compare_checkpoint_model
from FER.get_model import init_affectnet_feature_extractor
from models import DEMOTE_VQ
from utils.extra import seed_everything


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class DEEPTalkEngine:
    def __init__(
        self,
        deeptalk_config_path: str,
        dee_ckpt_path: str,
        use_sampling: bool = False,
        control_logvar: float | None = None,
        tau: float = 0.0001,
    ) -> None:
        self.deeptalk_config_path = deeptalk_config_path
        self.dee_ckpt_path = dee_ckpt_path
        self.use_sampling = use_sampling
        self.control_logvar = control_logvar
        self.tau = tau
        self.lock = threading.Lock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed_everything(42)
        demote_config_path = os.path.join(os.path.dirname(deeptalk_config_path), "config.json")
        with open(demote_config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        flint_config_path = self.config["motionprior_config"]["config_path"]
        with open(flint_config_path, "r", encoding="utf-8") as f:
            self.flint_config = json.load(f)
        flint_ckpt = self.config["motionprior_config"]["checkpoint_path"]

        dee_config_path = self._find_dee_config(dee_ckpt_path)
        print(f"[deeptalk_worker] DEE config loaded: {dee_config_path}", flush=True)
        dee_model, dee_config = get_DEE_from_json(dee_config_path)
        dee_checkpoint = torch.load(dee_ckpt_path, map_location="cpu")
        dee_model.load_state_dict(dee_checkpoint)
        dee_model.eval()
        compare_checkpoint_model(dee_checkpoint, dee_model.to("cpu"))

        self.affectnet_feature_extractor = None
        if dee_config.affectnet_model_path:
            model_path = dee_config.affectnet_model_path
            config_path = os.path.dirname(model_path) + "/config.yaml"
            _, self.affectnet_feature_extractor = init_affectnet_feature_extractor(config_path, model_path)
            self.affectnet_feature_extractor.to(self.device)
            self.affectnet_feature_extractor.eval()
            self.affectnet_feature_extractor.requires_grad_(False)

        self.model = DEMOTE_VQ.DEMOTE_VQVAE_condition(
            self.config,
            self.flint_config,
            dee_config,
            flint_ckpt,
            dee_model,
            load_motion_prior=False,
        )
        demote_ckpt = torch.load(deeptalk_config_path, map_location="cpu")
        self.model.load_state_dict(demote_ckpt)
        self._verify_loaded_weights(demote_ckpt, dee_checkpoint, flint_ckpt)
        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def _find_dee_config(dee_ckpt_path: str) -> str:
        ckpt_dir = os.path.dirname(dee_ckpt_path)
        for path in sorted(Path(ckpt_dir).glob("*.json")):
            return str(path)
        raise FileNotFoundError(f"No DEE config json found beside {dee_ckpt_path}")

    def _verify_loaded_weights(self, demote_ckpt: dict[str, Any], dee_checkpoint: dict[str, Any], flint_ckpt: str) -> None:
        state_dict = self.model.state_dict()
        flint_checkpoint = torch.load(flint_ckpt, map_location="cpu")

        for key in dee_checkpoint.keys():
            if key.startswith("audio_encoder"):
                original_weights = dee_checkpoint[key]
                loaded_weights = state_dict[f"sequence_decoder.DEE.{key}"]
                if not torch.allclose(original_weights, loaded_weights):
                    raise ValueError(f"{key} is different")

        for key in flint_checkpoint.keys():
            if key.startswith("motion_decoder"):
                original_weights = flint_checkpoint[key]
                new_key = key.replace("motion_decoder", "sequence_decoder.motion_prior")
                loaded_weights = state_dict[new_key]
                if not torch.allclose(original_weights, loaded_weights):
                    raise ValueError(f"{key} is different")

        for name, param in state_dict.items():
            original_weights = demote_ckpt[name]
            if not torch.allclose(param, original_weights):
                raise ValueError(f"{name} is different")

    def infer_file(self, audio_path: str, output_npy: str) -> dict[str, Any]:
        if audio_path.endswith(".wav"):
            wavdata, _ = librosa.load(audio_path, sr=16000)
        elif audio_path.endswith(".npy"):
            wavdata = np.load(audio_path)
        else:
            raise ValueError("audio file must be either .wav or .npy")

        audio = torch.tensor(wavdata, dtype=torch.float32)
        audio = pad_audio_to_match_quantfactor(
            audio,
            fps=self.config["audio_config"]["target_fps"],
            quant_factor=self.config["sequence_decoder_config"]["quant_factor"],
        )

        emotion = 0
        intensity = 0
        actor_id = MEAD_ACTOR_DICT["M003"]
        n_emotions = self.config["sequence_decoder_config"]["style_embedding"]["n_expression"]
        n_intensities = self.config["sequence_decoder_config"]["style_embedding"]["n_intensities"]
        n_identities = self.config["sequence_decoder_config"]["style_embedding"]["n_identities"]
        condition_size = n_emotions + n_intensities + n_identities
        input_style = torch.eye(condition_size)[
            [
                emotion,
                n_emotions + intensity,
                n_emotions + n_intensities + actor_id,
            ]
        ]
        input_style = torch.sum(input_style, dim=0).unsqueeze(0)

        with self.lock:
            audio = audio.unsqueeze(0).to(self.device)
            input_style = input_style.to(self.device)
            with torch.no_grad():
                output = self.model(
                    audio,
                    input_style,
                    sample=self.use_sampling,
                    control_logvar=self.control_logvar,
                    tau=self.tau,
                )

        output_array = output.squeeze(0).detach().cpu().numpy()
        out_path = Path(output_npy)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), output_array)
        return {
            "audio_path": audio_path,
            "output_npy": str(out_path),
            "shape": list(output_array.shape),
            "device": str(self.device),
        }


class DEEPTalkRequestHandler(BaseHTTPRequestHandler):
    engine: DEEPTalkEngine

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("[deeptalk_worker] " + format % args + "\n")
        sys.stdout.flush()

    def do_GET(self) -> None:
        if self.path != "/health":
            _json_response(self, 404, {"ok": False, "error": "not found"})
            return
        _json_response(self, 200, {"ok": True, "device": str(self.engine.device)})

    def do_POST(self) -> None:
        if self.path != "/infer":
            _json_response(self, 404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            audio_path = str(payload["audio_path"])
            output_npy = str(payload["output_npy"])
            result = self.engine.infer_file(audio_path, output_npy)
            _json_response(self, 200, {"ok": True, **result})
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-lived DEEPTalk motion worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--DEEPTalk_config_path", default="./checkpoint/DEEPTalk/DEEPTalk.pth")
    parser.add_argument("--DEE_ckpt_path", default="../DEE/checkpoint/DEE.pt")
    parser.add_argument("--use_sampling", action="store_true")
    parser.add_argument("--control_logvar", type=float, default=None)
    parser.add_argument("--tau", type=float, default=0.0001)
    args = parser.parse_args()

    torch.set_num_threads(1)
    print("[deeptalk_worker] loading DEEPTalk models...", flush=True)
    DEEPTalkRequestHandler.engine = DEEPTalkEngine(
        deeptalk_config_path=args.DEEPTalk_config_path,
        dee_ckpt_path=args.DEE_ckpt_path,
        use_sampling=args.use_sampling,
        control_logvar=args.control_logvar,
        tau=args.tau,
    )
    print(
        f"[deeptalk_worker] ready on http://{args.host}:{args.port} "
        f"device={DEEPTalkRequestHandler.engine.device}",
        flush=True,
    )
    server = ThreadingHTTPServer((args.host, args.port), DEEPTalkRequestHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
