from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

from fraclog.baseline import BaselineImportError, standardize_baseline_predictions


class BaselineImportTests(unittest.TestCase):
    def test_standardize_baseline_predictions_merges_eval_references_by_id(self) -> None:
        eval_rows = [
            {"id": "a", "images": ["pic/a.png"], "reference": "ref a", "count": 1},
            {"id": "b", "images": ["pic/b.png"], "reference": "ref b", "count": 2},
        ]
        prediction_rows = [
            {"id": "b", "prediction": "pred b", "latency_s": 1.5},
            {"id": "a", "prediction": "pred a"},
        ]

        rows = standardize_baseline_predictions(eval_rows, prediction_rows, prediction_field="prediction")

        self.assertEqual([row["id"] for row in rows], ["a", "b"])
        self.assertEqual(rows[0]["image"], "pic/a.png")
        self.assertEqual(rows[0]["reference"], "ref a")
        self.assertEqual(rows[0]["prediction"], "pred a")
        self.assertEqual(rows[1]["latency_s"], 1.5)

    def test_standardize_baseline_predictions_rejects_missing_rows(self) -> None:
        eval_rows = [
            {"id": "a", "images": ["pic/a.png"], "reference": "ref a"},
            {"id": "b", "images": ["pic/b.png"], "reference": "ref b"},
        ]
        prediction_rows = [{"id": "a", "prediction": "pred a"}]

        with self.assertRaisesRegex(BaselineImportError, "Missing baseline predictions"):
            standardize_baseline_predictions(eval_rows, prediction_rows, prediction_field="prediction")

    def test_standardize_baseline_predictions_rejects_duplicate_keys(self) -> None:
        eval_rows = [{"id": "a", "images": ["pic/a.png"], "reference": "ref a"}]
        prediction_rows = [
            {"id": "a", "prediction": "first"},
            {"id": "a", "prediction": "second"},
        ]

        with self.assertRaisesRegex(BaselineImportError, "Duplicate baseline prediction"):
            standardize_baseline_predictions(eval_rows, prediction_rows, prediction_field="prediction")

    def test_import_script_reads_csv_prediction_rows(self) -> None:
        module = importlib.import_module("scripts.import_baseline_predictions")

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "predictions.csv"
            csv_path.write_text(
                "id,image,prediction,latency_s\n"
                "a,pic/a.png,pred a,1.2\n"
                "b,pic/b.png,pred b,\n",
                encoding="utf-8-sig",
            )

            rows = module.read_prediction_rows(csv_path)

        self.assertEqual("a", rows[0]["id"])
        self.assertEqual("pred a", rows[0]["prediction"])
        self.assertEqual("1.2", rows[0]["latency_s"])
        self.assertEqual("pred b", rows[1]["prediction"])


if __name__ == "__main__":
    unittest.main()
