from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer
from yacs import config as CONFIG

from models.hifigan.get_vocoder import MAX_WAV_VALUE
from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class EmotiVoiceEngine:
    def __init__(self, logdir: str, config_folder: str, checkpoint: str) -> None:
        self.logdir = logdir
        self.config_folder = config_folder
        self.checkpoint = checkpoint
        self.lock = threading.Lock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sys.path.insert(0, os.path.abspath(config_folder))
        from config import Config

        self.config = Config()
        root_path = os.path.join(self.config.output_directory, logdir)
        ckpt_dir = os.path.join(root_path, "ckpt")
        checkpoint_name = checkpoint or self._first_checkpoint(ckpt_dir)
        self.checkpoint_path = os.path.join(ckpt_dir, checkpoint_name)
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")

        with open(self.config.model_config_path, "r", encoding="utf-8") as fin:
            conf = CONFIG.load_cfg(fin)
        conf.n_vocab = self.config.n_symbols
        conf.n_speaker = self.config.speaker_n_labels

        self.style_encoder = StyleEncoder(self.config)
        style_ckpt = torch.load(self.config.style_encoder_ckpt, map_location="cpu")
        model_ckpt = {}
        for key, value in style_ckpt["model"].items():
            model_ckpt[key[7:]] = value
        self.style_encoder.load_state_dict(model_ckpt, strict=False)
        self.style_encoder.eval()

        self.generator = JETSGenerator(conf).to(self.device)
        generator_ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(generator_ckpt["generator"])
        self.generator.eval()

        with open(self.config.token_list_path, "r", encoding="utf-8") as f:
            self.token2id = {t.strip(): idx for idx, t in enumerate(f.readlines())}
        with open(self.config.speaker2id_path, "r", encoding="utf-8") as f:
            self.speaker2id = {t.strip(): idx for idx, t in enumerate(f.readlines())}

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_path)

    @staticmethod
    def _first_checkpoint(ckpt_dir: str) -> str:
        files = sorted(os.listdir(ckpt_dir))
        if not files:
            raise FileNotFoundError(f"no checkpoints found in {ckpt_dir}")
        return files[0]

    def _style_embedding(self, prompt: str) -> np.ndarray:
        prompt_tokens = self.tokenizer([prompt], return_tensors="pt")
        with torch.no_grad():
            output = self.style_encoder(
                input_ids=prompt_tokens["input_ids"],
                token_type_ids=prompt_tokens["token_type_ids"],
                attention_mask=prompt_tokens["attention_mask"],
            )
        return output["pooled_output"].cpu().squeeze().numpy()

    def synthesize_file(self, test_file: str, output_wav: str) -> dict[str, Any]:
        test_path = Path(test_file)
        if not test_path.exists():
            raise FileNotFoundError(f"test_file not found: {test_file}")

        lines = [line.strip() for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"test_file has no usable lines: {test_file}")

        speaker, prompt, text, content = lines[0].split("|", 3)
        phonemes = text.split()
        if speaker not in self.speaker2id:
            raise ValueError(f"unknown speaker id: {speaker}")

        with self.lock:
            style_embedding = self._style_embedding(prompt)
            content_embedding = self._style_embedding(content)
            text_int = [self.token2id[ph] for ph in phonemes]

            sequence = torch.from_numpy(np.array(text_int)).to(self.device).long().unsqueeze(0)
            sequence_len = torch.from_numpy(np.array([len(text_int)])).to(self.device)
            style_tensor = torch.from_numpy(style_embedding).to(self.device).unsqueeze(0)
            content_tensor = torch.from_numpy(content_embedding).to(self.device).unsqueeze(0)
            speaker_tensor = torch.from_numpy(np.array([self.speaker2id[speaker]])).to(self.device)

            with torch.no_grad():
                infer_output = self.generator(
                    inputs_ling=sequence,
                    inputs_style_embedding=style_tensor,
                    input_lengths=sequence_len,
                    inputs_content_embedding=content_tensor,
                    inputs_speaker=speaker_tensor,
                    alpha=1.0,
                )
            audio = infer_output["wav_predictions"].squeeze() * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")

        out_path = Path(output_wav)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(file=str(out_path), data=audio, samplerate=self.config.sampling_rate)
        return {
            "output_wav": str(out_path),
            "sampling_rate": self.config.sampling_rate,
            "samples": int(audio.shape[0]),
            "device": str(self.device),
            "checkpoint": self.checkpoint_path,
        }


class TTSRequestHandler(BaseHTTPRequestHandler):
    engine: EmotiVoiceEngine

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("[tts_worker] " + format % args + "\n")
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
                "device": str(self.engine.device),
                "checkpoint": self.engine.checkpoint_path,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/synthesize":
            _json_response(self, 404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            test_file = str(payload["test_file"])
            output_wav = str(payload["output_wav"])
            result = self.engine.synthesize_file(test_file, output_wav)
            _json_response(self, 200, {"ok": True, **result})
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-lived EmotiVoice TTS worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--logdir", default="prompt_tts_open_source_joint")
    parser.add_argument("--config_folder", default="config/joint")
    parser.add_argument("--checkpoint", default="g_00140000")
    args = parser.parse_args()

    torch.set_num_threads(1)
    print("[tts_worker] loading EmotiVoice models...", flush=True)
    TTSRequestHandler.engine = EmotiVoiceEngine(args.logdir, args.config_folder, args.checkpoint)
    print(
        f"[tts_worker] ready on http://{args.host}:{args.port} "
        f"device={TTSRequestHandler.engine.device}",
        flush=True,
    )
    server = ThreadingHTTPServer((args.host, args.port), TTSRequestHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
