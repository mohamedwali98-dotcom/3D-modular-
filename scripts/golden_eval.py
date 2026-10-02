"""Score the shipped reading path on the golden set and hold it to the committed baseline (audit C3).
Usage:
  uv run python scripts/golden_eval.py                    # table of every sample and the pooled numbers
  uv run python scripts/golden_eval.py --check            # exit 1 when a number fell past the tolerance
  uv run python scripts/golden_eval.py --write-baseline   # record today's numbers (after adding samples)
Needs the trocr extra: uv sync --extra trocr."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from s2c.multiview import golden

ROOT = Path(__file__).resolve().parents[1] / "tests" / "golden_sketch"


def table(rows: list[dict], summary: dict) -> str:
    lines = ["| sample | faces | values read | sizes right | silently wrong | builds | note |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {r['name']} | {r['named']}/{r['views']} | {r['matched']}/{r['values']} | {r['correct']}/{r['sizes']} "
              f"| {r['wrong']} | {'yes' if r['built'] else 'no'} | {r['note']} |" for r in rows]
    s = summary
    lines.append(f"| **all {s['samples']}** | {s['faces']:.2f} | {s['values']:.2f} | {s['sizes']:.2f} "
                 f"| {s['silent_errors']} | {s['builds']:.2f} | |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--check", action="store_true", help="fail when a number fell below the baseline")
    ap.add_argument("--write-baseline", action="store_true", help="record these numbers as the baseline")
    ap.add_argument("--out", type=Path, help="also write the rows and summary as JSON here")
    args = ap.parse_args(argv)
    result = golden.evaluate(args.root)
    print(table(result["rows"], result["summary"]))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    baseline_path = args.root / "baseline.json"
    if args.write_baseline:
        baseline_path.write_text(json.dumps(result["summary"], indent=2) + "\n", encoding="utf-8")
        print(f"baseline written to {baseline_path}")
    if args.check:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline.get("samples") != result["summary"]["samples"]:
            print(f"note: {result['summary']['samples']} samples, the baseline has {baseline.get('samples')}: "
                  "run --write-baseline after adding samples")
        failed = golden.regressions(result["summary"], baseline)
        for line in failed:
            print(f"REGRESSION: {line}")
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
