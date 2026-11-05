"""Train the LoRA adapter.

    python scripts/train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --output /path/to/adapter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finetune.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--validation", default="data/validation.jsonl")
    parser.add_argument("--output", default="outputs/adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=3072)
    args = parser.parse_args()

    report = train(args.model, args.train, args.validation, args.output, epochs=args.epochs,
                   batch_size=args.batch_size, accumulation=args.accumulation, peak_lr=args.lr,
                   lora_rank=args.rank, max_length=args.max_length)
    Path(args.output, "training_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=1))


if __name__ == "__main__":
    main()
