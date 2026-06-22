#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from fraclog.baseline import BaselineImportError, standardize_baseline_predictions
from fraclog.data import load_jsonl, write_jsonl


def read_prediction_rows(path: str | Path) -> list[dict[str, str]]:
    input_path = Path(path)
    if input_path.suffix.lower() != ".csv":
        return load_jsonl(input_path)

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: value for key, value in row.items() if key is not None and value is not None} for row in reader]


def main() -> int:
    parser = argparse.ArgumentParser(description="Import existing E0 baseline predictions into the standard evaluator format.")
    parser.add_argument("--input", required=True, help="JSONL or CSV with one prediction per eval row, keyed by id or image.")
    parser.add_argument("--eval", default="outputs/data/eval_test.jsonl")
    parser.add_argument("--output", default="outputs/predictions/e0_predictions.jsonl")
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--metrics-out", default="outputs/e0/metrics.json")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--refresh-summary", action="store_true")
    parser.add_argument("--refresh-audit", action="store_true")
    args = parser.parse_args()

    try:
        rows = standardize_baseline_predictions(
            load_jsonl(args.eval),
            read_prediction_rows(args.input),
            prediction_field=args.prediction_field,
        )
    except BaselineImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} standardized E0 predictions to {output_path}")

    if args.evaluate:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/evaluate_predictions.py",
                "--predictions",
                str(output_path),
                "--out",
                args.metrics_out,
                "--experiment",
                "E0",
                "--model",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "--method",
                "DAPT + LoRA-SFT + DPO",
            ],
            check=False,
        )
        if completed.returncode:
            return completed.returncode
    if args.refresh_summary:
        completed = subprocess.run([sys.executable, "scripts/summarize_results.py"], check=False)
        if completed.returncode:
            return completed.returncode
    if args.refresh_audit:
        completed = subprocess.run([sys.executable, "scripts/audit_completion.py"], check=False)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
