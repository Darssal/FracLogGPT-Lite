from __future__ import annotations

from typing import Any


PAPER_BASELINE_SOURCE = "MDPI Electronics 2025, Table 5"
PAPER_BASELINE_URL = "https://www.mdpi.com/2079-9292/15/1/127"


def build_paper_baseline_metrics() -> dict[str, Any]:
    return {
        "experiment": "E0",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "method": "DAPT + LoRA-SFT + DPO (paper Full Model)",
        "count_precision": 0.708,
        "count_recall": 0.700,
        "count_f1": 0.697,
        "count_accuracy": None,
        "loc_acc": 0.488,
        "morph_acc": 0.431,
        "avg_bleu": 0.592,
        "avg_rouge_l": 0.670,
        "samples": 160,
        "zero_fracture_hallucination_rate": None,
        "over_count_rate": None,
        "under_count_rate": None,
        "peak_vram_mib": None,
        "training_time_s": None,
        "avg_latency_s": None,
        "source": PAPER_BASELINE_SOURCE,
        "source_url": PAPER_BASELINE_URL,
        "same_test_set": False,
        "comparison_scope": "paper reported independent 160-image test set; directional reference only",
        "notes": [
            "This E0 row uses the paper-reported Full Model metrics directly because no local E0 checkpoint or prediction file is available.",
            "It is a directional paper baseline reference, not a same-test-set local reevaluation.",
        ],
    }
