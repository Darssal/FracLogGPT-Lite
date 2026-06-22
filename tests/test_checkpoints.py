from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fraclog.checkpoints import find_latest_checkpoint, resolve_placeholders


class CheckpointResolutionTests(unittest.TestCase):
    def test_find_latest_checkpoint_prefers_newest_checkpoint_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "v0" / "checkpoint-1"
            new = root / "v1" / "checkpoint-2"
            old.mkdir(parents=True)
            new.mkdir(parents=True)
            (old / "adapter_model.safetensors").write_text("old", encoding="utf-8")
            (new / "adapter_model.safetensors").write_text("new", encoding="utf-8")

            latest = find_latest_checkpoint(root)

        self.assertEqual(latest.name, "checkpoint-2")

    def test_resolve_stage_checkpoint_placeholder(self) -> None:
        resolved = resolve_placeholders(
            {"adapters": "${checkpoint:sft}", "nested": ["${checkpoint:sft}"]},
            {"sft": "outputs/checkpoints/e2/sft/v1/checkpoint-100"},
        )
        self.assertEqual(resolved["adapters"], "outputs/checkpoints/e2/sft/v1/checkpoint-100")
        self.assertEqual(resolved["nested"][0], "outputs/checkpoints/e2/sft/v1/checkpoint-100")

    def test_unresolved_placeholder_can_be_preserved_for_dry_run(self) -> None:
        resolved = resolve_placeholders("${checkpoint:dapt}", {}, allow_unresolved=True)
        self.assertEqual(resolved, "${checkpoint:dapt}")


if __name__ == "__main__":
    unittest.main()
