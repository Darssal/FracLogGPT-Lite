from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_pipeline import load_config_with_resolved_args, seed_stage_checkpoints_from_existing


class RunPipelineResumeTests(unittest.TestCase):
    def test_seed_stage_checkpoints_from_existing_output_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dapt_checkpoint = root / "outputs" / "checkpoints" / "e1" / "dapt" / "v0" / "checkpoint-100"
            dapt_checkpoint.mkdir(parents=True)
            (dapt_checkpoint / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
            config = {
                "stages": {
                    "dapt": {
                        "command": "pt",
                        "args": {"output_dir": "outputs/checkpoints/e1/dapt"},
                    },
                    "sft": {
                        "command": "sft",
                        "args": {
                            "adapters": "${checkpoint:dapt}",
                            "output_dir": "outputs/checkpoints/e1/sft",
                        },
                    },
                }
            }

            stage_checkpoints = seed_stage_checkpoints_from_existing(config, root=root)
            resolved = load_config_with_resolved_args(
                config,
                "sft",
                stage_checkpoints,
                allow_unresolved=False,
            )

        self.assertEqual(stage_checkpoints["dapt"], str(dapt_checkpoint))
        self.assertEqual(resolved["stages"]["sft"]["args"]["adapters"], str(dapt_checkpoint))


if __name__ == "__main__":
    unittest.main()
