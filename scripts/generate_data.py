"""Generate training and validation conversations.

    python scripts/generate_data.py
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from finetune.traces import build_examples, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as work_dir:
        train, validation, stats = build_examples(work_dir, tuple(args.seeds))
    out = Path(args.out)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(validation, out / "validation.jsonl")
    summary = {
        "train": len(train), "validation": len(validation), **stats,
        "train_domains": dict(Counter(e["domain"] for e in train)),
        "train_kinds": dict(Counter(e["kind"] for e in train)),
        "validation_kinds": dict(Counter(e["kind"] for e in validation)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
