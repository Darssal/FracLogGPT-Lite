#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean

from fraclog.data import load_jsonl, write_json
from fraclog.metrics import (
    average_text_metrics,
    count_category,
    compute_count_metrics,
    compute_hallucination_metrics,
    compute_structured_text_metrics,
    extract_location_labels,
    extract_morphology_labels,
    parse_fracture_count,
    rouge_l_score,
    simple_bleu,
)


DEFAULT_METRICS = [
    "count_precision",
    "count_recall",
    "count_f1",
    "count_accuracy",
    "loc_acc",
    "morph_acc",
    "avg_bleu",
    "avg_rouge_l",
    "zero_fracture_hallucination_rate",
    "over_count_rate",
    "under_count_rate",
]


def default_prediction_specs() -> list[tuple[str, str]]:
    return [
        ("E1", "outputs/predictions/e1_predictions.jsonl"),
        ("E2", "outputs/predictions/e2_predictions.jsonl"),
        ("E3", "outputs/predictions/e3_frac_vcd_predictions.jsonl"),
    ]


def compute_metrics_for_rows(rows: list[dict]) -> dict[str, float | int | None]:
    refs = [row.get("reference", "") for row in rows]
    preds = [row.get("prediction", "") for row in rows]
    ref_counts = [
        row.get("count") if row.get("count") is not None else parse_fracture_count(ref)
        for row, ref in zip(rows, refs)
    ]
    pred_counts = [parse_fracture_count(pred) for pred in preds]
    metrics: dict[str, float | int | None] = {}
    metrics.update(compute_count_metrics(ref_counts, pred_counts))
    metrics.update(compute_hallucination_metrics(ref_counts, pred_counts))
    metrics.update(average_text_metrics(preds, refs))
    metrics.update(compute_structured_text_metrics(refs, preds))
    metrics["samples"] = len(rows)
    return metrics


def _rank(label: str | None) -> int | None:
    if label is None:
        return None
    return {"0": 0, "1": 1, "2": 2, ">=3": 3}[label]


def _macro_count_metrics(ref_labels: list[str | None], pred_labels: list[str | None]) -> dict[str, float]:
    labels = ["0", "1", "2", ">=3"]
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for label in labels:
        tp = sum(ref == label and pred == label for ref, pred in zip(ref_labels, pred_labels))
        fp = sum(ref != label and pred == label for ref, pred in zip(ref_labels, pred_labels))
        fn = sum(ref == label and pred != label for ref, pred in zip(ref_labels, pred_labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    return {
        "count_precision": sum(precision_values) / len(labels),
        "count_recall": sum(recall_values) / len(labels),
        "count_f1": sum(f1_values) / len(labels),
        "count_accuracy": sum(ref == pred for ref, pred in zip(ref_labels, pred_labels)) / len(ref_labels) if ref_labels else 0.0,
    }


def precompute_rows(rows: list[dict]) -> list[dict[str, float | int | str | bool | None]]:
    prepared = []
    for row in rows:
        ref = row.get("reference", "")
        pred = row.get("prediction", "")
        ref_count = row.get("count") if row.get("count") is not None else parse_fracture_count(ref)
        pred_count = parse_fracture_count(pred)
        ref_label = count_category(ref_count)
        pred_label = count_category(pred_count)
        ref_loc = extract_location_labels(ref)
        pred_loc = extract_location_labels(pred)
        ref_morph = extract_morphology_labels(ref)
        pred_morph = extract_morphology_labels(pred)
        prepared.append(
            {
                "ref_label": ref_label,
                "pred_label": pred_label,
                "ref_rank": _rank(ref_label),
                "pred_rank": _rank(pred_label),
                "bleu": simple_bleu(pred, ref),
                "rouge_l": rouge_l_score(pred, ref),
                "loc_eval": bool(ref_loc),
                "loc_ok": bool(ref_loc) and ref_loc.issubset(pred_loc),
                "morph_eval": bool(ref_morph),
                "morph_ok": bool(ref_morph) and ref_morph.issubset(pred_morph),
            }
        )
    return prepared


def aggregate_precomputed(prepared: list[dict[str, float | int | str | bool | None]], indices: list[int]) -> dict[str, float | int | None]:
    selected = [prepared[index] for index in indices]
    ref_labels = [row["ref_label"] for row in selected]
    pred_labels = [row["pred_label"] for row in selected]
    metrics: dict[str, float | int | None] = {}
    metrics.update(_macro_count_metrics(ref_labels, pred_labels))
    ranked_pairs = [
        (row["ref_rank"], row["pred_rank"])
        for row in selected
        if row["ref_rank"] is not None and row["pred_rank"] is not None
    ]
    zero_refs = [(ref, pred) for ref, pred in ranked_pairs if ref == 0]
    metrics["zero_fracture_hallucination_rate"] = (
        sum(pred > 0 for _, pred in zero_refs) / len(zero_refs) if zero_refs else 0.0
    )
    metrics["over_count_rate"] = sum(pred > ref for ref, pred in ranked_pairs) / len(ranked_pairs) if ranked_pairs else 0.0
    metrics["under_count_rate"] = sum(pred < ref for ref, pred in ranked_pairs) / len(ranked_pairs) if ranked_pairs else 0.0
    metrics["avg_bleu"] = sum(float(row["bleu"]) for row in selected) / len(selected) if selected else 0.0
    metrics["avg_rouge_l"] = sum(float(row["rouge_l"]) for row in selected) / len(selected) if selected else 0.0
    loc_rows = [row for row in selected if row["loc_eval"]]
    morph_rows = [row for row in selected if row["morph_eval"]]
    metrics["loc_acc"] = sum(bool(row["loc_ok"]) for row in loc_rows) / len(loc_rows) if loc_rows else None
    metrics["morph_acc"] = sum(bool(row["morph_ok"]) for row in morph_rows) / len(morph_rows) if morph_rows else None
    metrics["samples"] = len(selected)
    return metrics


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def bootstrap_rows(
    rows: list[dict],
    *,
    iterations: int = 1000,
    seed: int = 42,
    metrics: list[str] | None = None,
) -> dict[str, dict[str, float | int]]:
    if not rows:
        raise ValueError("Cannot bootstrap an empty prediction set")
    metrics = metrics or DEFAULT_METRICS
    rng = random.Random(seed)
    point = compute_metrics_for_rows(rows)
    prepared = precompute_rows(rows)
    samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(iterations):
        sampled_indices = [rng.randrange(len(rows)) for _ in rows]
        current = aggregate_precomputed(prepared, sampled_indices)
        for metric in metrics:
            value = current.get(metric)
            if value is not None:
                samples[metric].append(float(value))
    summary: dict[str, dict[str, float | int]] = {}
    for metric in metrics:
        values = samples[metric]
        if not values:
            continue
        summary[metric] = {
            "point": float(point.get(metric) or 0.0),
            "mean": mean(values),
            "ci_low": percentile(values, 0.025),
            "ci_high": percentile(values, 0.975),
            "iterations": iterations,
            "samples": len(rows),
        }
    return summary


def write_summary_csv(path: Path, experiment: str, summary: dict[str, dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["experiment", "metric", "point", "mean", "ci_low", "ci_high", "iterations", "samples"],
        )
        writer.writeheader()
        for metric, values in summary.items():
            writer.writerow({"experiment": experiment, "metric": metric, **values})


def process_prediction_file(
    experiment: str,
    predictions: Path,
    out_dir: Path,
    *,
    iterations: int = 1000,
    seed: int = 42,
) -> dict:
    rows = load_jsonl(predictions)
    summary = bootstrap_rows(rows, iterations=iterations, seed=seed)
    payload = {
        "experiment": experiment,
        "predictions": str(predictions),
        "seed": seed,
        "iterations": iterations,
        "metrics": summary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = experiment.lower().replace("/", "_")
    write_json(out_dir / f"bootstrap_metrics_{safe_name}.json", payload)
    write_summary_csv(out_dir / f"bootstrap_metrics_{safe_name}.csv", experiment, summary)
    return payload


def write_combined_markdown(path: Path, results: list[dict]) -> None:
    lines = [
        "# Bootstrap Confidence Intervals",
        "",
        "These intervals use sample-level nonparametric bootstrap over the existing prediction rows. They do not replace same-split E0 reproduction or manual expert review.",
        "",
        "|Experiment|Metric|Point|95% CI|Iterations|Samples|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        experiment = result["experiment"]
        for metric, values in result["metrics"].items():
            lines.append(
                f"|{experiment}|{metric}|{values['point']:.4f}|[{values['ci_low']:.4f}, {values['ci_high']:.4f}]|{values['iterations']}|{values['samples']}|"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for FracLog prediction JSONL files.")
    parser.add_argument("--out-dir", default="reports/bootstrap")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prediction",
        action="append",
        nargs=2,
        metavar=("EXPERIMENT", "PATH"),
        help="Experiment name and prediction JSONL path. Can be repeated.",
    )
    args = parser.parse_args()

    predictions = args.prediction or default_prediction_specs()
    out_dir = Path(args.out_dir)
    results = [
        process_prediction_file(
            experiment,
            Path(prediction_path),
            out_dir,
            iterations=args.iterations,
            seed=args.seed,
        )
        for experiment, prediction_path in predictions
    ]
    write_json(out_dir / "bootstrap_metrics_all.json", {"results": results})
    write_combined_markdown(out_dir / "bootstrap_metrics_summary.md", results)
    print(f"Wrote bootstrap summaries for {len(results)} experiments to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
