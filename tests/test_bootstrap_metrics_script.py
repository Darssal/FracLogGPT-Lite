from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

from fraclog.data import write_jsonl


class BootstrapMetricsScriptTest(unittest.TestCase):
    def test_bootstrap_ci_returns_metric_bounds(self) -> None:
        module = importlib.import_module("scripts.bootstrap_metrics")
        rows = [
            {
                "reference": "one fracture; location: upper third; morphology: white hyperbola s-shaped symmetric",
                "prediction": "one fracture; location: upper third; morphology: white hyperbola s-shaped symmetric",
                "count": 1,
            },
            {
                "reference": "two fractures; location: lower third; morphology: white hyperbola",
                "prediction": "two fractures; location: lower third; morphology: white hyperbola",
                "count": 2,
            },
            {
                "reference": "one fracture; location: upper third; morphology: white hyperbola",
                "prediction": "two fractures; location: lower third; morphology: white",
                "count": 1,
            },
            {
                "reference": "two fractures; location: lower third; morphology: white hyperbola",
                "prediction": "one fracture; location: upper third; morphology: hyperbola",
                "count": 2,
            },
        ]

        summary = module.bootstrap_rows(rows, iterations=50, seed=123)

        for metric in ["count_f1", "avg_bleu", "avg_rouge_l", "loc_acc", "morph_acc"]:
            self.assertIn(metric, summary)
            self.assertLessEqual(summary[metric]["ci_low"], summary[metric]["mean"])
            self.assertLessEqual(summary[metric]["mean"], summary[metric]["ci_high"])

    def test_process_prediction_file_writes_outputs(self) -> None:
        module = importlib.import_module("scripts.bootstrap_metrics")
        rows = [
            {"id": "a", "reference": "one fracture", "prediction": "one fracture", "count": 1},
            {"id": "b", "reference": "two fractures", "prediction": "two fractures", "count": 2},
            {"id": "c", "reference": "one fracture", "prediction": "two fractures", "count": 1},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pred_path = tmp_path / "pred.jsonl"
            write_jsonl(pred_path, rows)
            out_dir = tmp_path / "out"

            result = module.process_prediction_file(
                "toy",
                pred_path,
                out_dir,
                iterations=25,
                seed=7,
            )

            self.assertEqual("toy", result["experiment"])
            self.assertTrue((out_dir / "bootstrap_metrics_toy.json").exists())
            self.assertTrue((out_dir / "bootstrap_metrics_toy.csv").exists())

    def test_default_prediction_specs_cover_public_matrix(self) -> None:
        module = importlib.import_module("scripts.bootstrap_metrics")
        specs = module.default_prediction_specs()

        experiments = [experiment for experiment, _ in specs]
        self.assertEqual(["E1", "E2", "E3"], experiments)


if __name__ == "__main__":
    unittest.main()
