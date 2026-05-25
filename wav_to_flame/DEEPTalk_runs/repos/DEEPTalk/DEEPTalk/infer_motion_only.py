from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from deeptalk_worker import DEEPTalkEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DEEPTalk and save motion npy only")
    parser.add_argument("--audio_path", required=True, help="Path to input wav or npy")
    parser.add_argument("--output_npy", required=True, help="Destination .npy path")
    parser.add_argument("--DEEPTalk_config_path", default="./checkpoint/DEEPTalk/DEEPTalk.pth")
    parser.add_argument("--DEE_ckpt_path", default="../DEE/checkpoint/DEE.pt")
    parser.add_argument("--use_sampling", action="store_true")
    parser.add_argument("--control_logvar", type=float, default=None)
    parser.add_argument("--tau", type=float, default=0.0001)
    args = parser.parse_args()

    out_path = Path(args.output_npy)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    engine = DEEPTalkEngine(
        deeptalk_config_path=args.DEEPTalk_config_path,
        dee_ckpt_path=args.DEE_ckpt_path,
        use_sampling=args.use_sampling,
        control_logvar=args.control_logvar,
        tau=args.tau,
    )
    result = engine.infer_file(args.audio_path, str(out_path))

    motion = np.load(str(out_path), allow_pickle=True)
    print(f"saved in {out_path}")
    print(f"shape = {motion.shape}")
    print(f"device = {result['device']}")


if __name__ == "__main__":
    main()
