from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fraclog.data import write_jsonl
from scripts.prepare_dapt_corpus import rows_from_sft_jsonl


class PrepareDaptCorpusTests(unittest.TestCase):
    def test_rows_from_sft_jsonl_uses_assistant_text_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sft.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "messages": [
                            {"role": "user", "content": "<image>question"},
                            {"role": "assistant", "content": "assistant answer"},
                        ]
                    }
                ],
            )

            rows = rows_from_sft_jsonl(path, max_chars=100)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "assistant answer")

    def test_rows_from_sft_jsonl_can_include_user_text_without_image_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sft.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "messages": [
                            {"role": "user", "content": "<image>question"},
                            {"role": "assistant", "content": "assistant answer"},
                        ]
                    }
                ],
            )

            rows = rows_from_sft_jsonl(path, max_chars=100, include_user_text=True)

        self.assertEqual(rows[0]["text"], "question\nassistant answer")


if __name__ == "__main__":
    unittest.main()
