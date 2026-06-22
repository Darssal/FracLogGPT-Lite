from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fraclog.data import write_jsonl
from fraclog.predictions import (
    dedupe_prediction_file,
    dedupe_prediction_rows,
    prediction_file_lock,
    select_rows_for_prediction,
)


class PredictionResumeTests(unittest.TestCase):
    def test_select_rows_for_prediction_skips_existing_ids_when_resuming(self) -> None:
        rows = [
            {"id": "a", "images": ["pic/a.png"]},
            {"id": "b", "images": ["pic/b.png"]},
            {"id": "c", "images": ["pic/c.png"]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "predictions.jsonl"
            write_jsonl(existing, [{"id": "b", "prediction": "already done"}])

            selected = select_rows_for_prediction(rows, existing, resume=True)

        self.assertEqual([row["id"] for row in selected], ["a", "c"])

    def test_select_rows_for_prediction_overwrites_by_default(self) -> None:
        rows = [{"id": "a"}, {"id": "b"}]
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "predictions.jsonl"
            write_jsonl(existing, [{"id": "a", "prediction": "already done"}])

            selected = select_rows_for_prediction(rows, existing, resume=False)

        self.assertEqual([row["id"] for row in selected], ["a", "b"])

    def test_dedupe_prediction_rows_keeps_latest_row_per_key(self) -> None:
        rows = [
            {"id": "a", "prediction": "old"},
            {"id": "b", "prediction": "only"},
            {"id": "a", "prediction": "new"},
        ]

        deduped = dedupe_prediction_rows(rows)

        self.assertEqual(
            deduped,
            [
                {"id": "a", "prediction": "new"},
                {"id": "b", "prediction": "only"},
            ],
        )

    def test_dedupe_prediction_file_rewrites_jsonl_and_returns_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            write_jsonl(
                path,
                [
                    {"id": "a", "prediction": "old"},
                    {"id": "b", "prediction": "only"},
                    {"id": "a", "prediction": "new"},
                ],
            )

            counts = dedupe_prediction_file(path)

            self.assertEqual(counts, (3, 2))
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                [
                    '{"id": "a", "prediction": "new"}',
                    '{"id": "b", "prediction": "only"}',
                ],
            )

    def test_prediction_file_lock_prevents_concurrent_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"

            with prediction_file_lock(path):
                with self.assertRaises(RuntimeError):
                    with prediction_file_lock(path):
                        pass

            with prediction_file_lock(path):
                self.assertTrue(Path(f"{path}.lock").exists())


if __name__ == "__main__":
    unittest.main()
