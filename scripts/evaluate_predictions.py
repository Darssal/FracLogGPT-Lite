#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from fraclog.data import load_jsonl, write_json
from fraclog.metrics import (
    average_text_metrics,
    compute_count_metrics,
    compute_hallucination_metrics,
    compute_structured_text_metrics,
    parse_fracture_count,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate FracLog prediction JSONL files.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--experiment")
    parser.add_argument("--model")
    parser.add_argument("--method")
    parser.add_argument("--peak-vram-mib", type=float)
    parser.add_argument("--training-time-s", type=float)
    args = parser.parse_args()

    rows = load_jsonl(args.predictions)
    refs = [row.get("reference", "") for row in rows]
    preds = [row.get("prediction", "") for row in rows]
    ref_counts = [row.get("count") if row.get("count") is not None else parse_fracture_count(ref) for row, ref in zip(rows, refs)]
    pred_counts = [parse_fracture_count(pred) for pred in preds]
    latencies = [float(row["latency_s"]) for row in rows if row.get("latency_s") is not None]

    metrics = {}
    metrics.update(compute_count_metrics(ref_counts, pred_counts))
    metrics.update(compute_hallucination_metrics(ref_counts, pred_counts))
    metrics.update(average_text_metrics(preds, refs))
    metrics.update(compute_structured_text_metrics(refs, preds))
    metrics["samples"] = len(rows)
    metrics["avg_latency_s"] = sum(latencies) / len(latencies) if latencies else None
    if args.experiment:
        metrics["experiment"] = args.experiment
    if args.model:
        metrics["model"] = args.model
    if args.method:
        metrics["method"] = args.method
    if args.peak_vram_mib is not None:
        metrics["peak_vram_mib"] = args.peak_vram_mib
    if args.training_time_s is not None:
        metrics["training_time_s"] = args.training_time_s
    metrics["notes"] = [
        "loc_acc and morph_acc are automatic text-label consistency metrics parsed from reference and prediction text; they are not a replacement for manual expert review labels.",
    ]
    write_json(args.out, metrics)
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
