"""Build a comparison table from benchmark results.

    python scripts/compare.py --run "Base=results/base/benchmark.json" --run "LoRA=results/lora/benchmark.json"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CATEGORIES = ("aggregation", "join", "multi_step", "python", "refusal")


def summarize(summary: dict) -> dict:
    rows = summary["rows"]
    n = len(rows)
    return {
        "model": summary["model"],
        "correct": sum(r["correct"] for r in rows),
        "n": n,
        "by_category": {c: summary["by_category"].get(c, {"correct": 0, "n": 0}) for c in CATEGORIES},
        "finished": sum(r["status"] == "answered" for r in rows),
        "native_final_answers": sum(r["status"] == "answered" and not r.get("answer_recovered_from_text") for r in rows),
        "recovered_from_text": sum(bool(r.get("answer_recovered_from_text")) for r in rows),
        "statuses": summary.get("statuses", {}),
        "tool_errors": sum(r["tool_errors"] for r in rows),
        "mean_steps": round(sum(r["steps"] for r in rows) / n, 2),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 1),
    }


def to_markdown(runs: dict[str, dict]) -> str:
    names = list(runs)
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]

    def row(label: str, value) -> None:
        lines.append(f"| {label} | " + " | ".join(str(value(runs[name])) for name in names) + " |")

    row("Model", lambda s: f"`{s['model']}`")
    row("**Correct**", lambda s: f"**{s['correct']}/{s['n']}**")
    for category in CATEGORIES:
        row(category, lambda s, c=category: f"{s['by_category'][c]['correct']}/{s['by_category'][c]['n']}")
    row("Finished with an answer", lambda s: f"{s['finished']}/{s['n']}")
    row("Answers sent as native tool calls", lambda s: f"{s['native_final_answers']}/{s['n']}")
    row("Answers recovered from plain text", lambda s: s["recovered_from_text"])
    row("Tool errors", lambda s: s["tool_errors"])
    row("Mean steps", lambda s: s["mean_steps"])
    row("Mean seconds per question", lambda s: s["mean_seconds"])
    row("How runs ended", lambda s: ", ".join(f"{k}: {v}" for k, v in sorted(s["statuses"].items())))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="Name=path/to/benchmark.json")
    parser.add_argument("--out", default="results/comparison.md")
    args = parser.parse_args()

    runs = {}
    for item in args.run:
        name, path = item.split("=", 1)
        runs[name] = summarize(json.loads(Path(path).read_text()))
    table = to_markdown(runs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(table)
    Path(args.out).with_suffix(".json").write_text(json.dumps(runs, indent=1))
    print(table)


if __name__ == "__main__":
    main()
