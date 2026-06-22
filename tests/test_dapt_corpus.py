from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fraclog.dapt import check_dapt_corpus_for_formal, metadata_path, write_dapt_corpus


class DaptCorpusMetadataTests(unittest.TestCase):
    def test_write_dapt_corpus_records_source_type_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dapt_train.jsonl"

            write_dapt_corpus(
                out,
                [{"text": "fracture geology text", "source": "data/dapt_corpus/a.txt"}],
                source_type="real_corpus",
                source="data/dapt_corpus",
                max_chars=4000,
                include_user_text=False,
            )

            meta = metadata_path(out).read_text(encoding="utf-8")

        self.assertIn('"source_type": "real_corpus"', meta)
        self.assertIn('"rows": 1', meta)

    def test_check_dapt_corpus_rejects_sft_derived_for_formal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dapt_train.jsonl"
            write_dapt_corpus(
                out,
                [{"text": "assistant answer", "source": "outputs/data/sft_train.jsonl"}],
                source_type="from_sft_jsonl",
                source="outputs/data/sft_train.jsonl",
                max_chars=4000,
                include_user_text=False,
            )

            status = check_dapt_corpus_for_formal(out)

        self.assertFalse(status["ok"])
        self.assertEqual(status["source_type"], "from_sft_jsonl")
        self.assertIn("paper-strict reproduction requires a real geological DAPT corpus", status["message"])

    def test_check_dapt_corpus_accepts_real_corpus_for_formal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dapt_train.jsonl"
            write_dapt_corpus(
                out,
                [{"text": "fracture geology text", "source": "data/dapt_corpus/a.txt"}],
                source_type="real_corpus",
                source="data/dapt_corpus",
                max_chars=4000,
                include_user_text=False,
            )

            status = check_dapt_corpus_for_formal(out)

        self.assertTrue(status["ok"])
        self.assertEqual(status["source_type"], "real_corpus")


if __name__ == "__main__":
    unittest.main()
