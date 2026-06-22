#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MAIN_COLUMNS = [
    "experiment",
    "model",
    "method",
    "count_precision",
    "count_recall",
    "count_f1",
    "loc_acc",
    "morph_acc",
    "avg_bleu",
    "avg_rouge_l",
    "samples",
    "zero_fracture_hallucination_rate",
    "over_count_rate",
    "under_count_rate",
]

EFF_COLUMNS = ["experiment", "peak_vram_mib", "training_time_s", "avg_latency_s"]

EXPERIMENT_TEMPLATES = {
    "E0": {"experiment": "E0", "model": "Qwen/Qwen2.5-VL-7B-Instruct", "method": "DAPT + LoRA-SFT + DPO"},
    "E1": {"experiment": "E1", "model": "Qwen/Qwen3.5-4B", "method": "DAPT + LoRA-SFT + DPO"},
    "E2": {"experiment": "E2", "model": "Qwen/Qwen3.5-4B", "method": "DAPT + LoRA-SFT + SimPO"},
    "E3": {"experiment": "E3", "model": "Qwen/Qwen3.5-4B", "method": "DAPT + LoRA-SFT + SimPO + Frac-VCD"},
}

EXPERIMENT_ORDER = [
    "E0",
    "E1",
    "E2",
    "E3",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize E0-E3 metrics into report CSV files.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    rows_by_experiment = {
        name: {column: template.get(column) for column in MAIN_COLUMNS}
        for name, template in EXPERIMENT_TEMPLATES.items()
    }
    efficiency_by_experiment = {
        name: {"experiment": name, "peak_vram_mib": None, "training_time_s": None, "avg_latency_s": None}
        for name in EXPERIMENT_TEMPLATES
    }
    for metrics_path in sorted(Path(args.outputs_dir).glob("**/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        experiment = str(metrics.get("experiment") or metrics_path.parent.name).upper()
        if experiment not in EXPERIMENT_TEMPLATES:
            continue
        row = {column: metrics.get(column) for column in MAIN_COLUMNS}
        row["experiment"] = experiment
        row["model"] = metrics.get("model") or EXPERIMENT_TEMPLATES[experiment]["model"]
        row["method"] = metrics.get("method") or EXPERIMENT_TEMPLATES[experiment]["method"]
        rows_by_experiment[experiment].update({key: value for key, value in row.items() if value is not None})
        efficiency_by_experiment[experiment].update(
            {
                "peak_vram_mib": metrics.get("peak_vram_mib"),
                "training_time_s": metrics.get("training_time_s"),
                "avg_latency_s": metrics.get("avg_latency_s"),
            }
        )

    rows = [rows_by_experiment[name] for name in EXPERIMENT_ORDER]
    efficiency = [efficiency_by_experiment[name] for name in EXPERIMENT_ORDER]

    with (reports / "results_main.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAIN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (reports / "results_efficiency.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EFF_COLUMNS)
        writer.writeheader()
        writer.writerows(efficiency)
    print(f"Wrote {len(rows)} result rows to {reports}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
