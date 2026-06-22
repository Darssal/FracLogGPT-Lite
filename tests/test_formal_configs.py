from __future__ import annotations

import json
import unittest
from pathlib import Path


class FormalConfigTests(unittest.TestCase):
    def test_formal_dapt_and_preference_stages_save_resumable_checkpoints(self) -> None:
        for config_path in [
            Path("configs/e0_qwen25_vl7b_lora_dpo.json"),
            Path("configs/e1_qwen35_4b_lora_dpo.json"),
            Path("configs/e2_qwen35_4b_lora_simpo.json"),
        ]:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for stage_name, stage in config["stages"].items():
                if stage.get("command") not in {"pt", "rlhf"}:
                    continue
                save_steps = int(stage["args"].get("save_steps", 0))
                self.assertGreater(save_steps, 0, f"{config_path}:{stage_name} must save checkpoints")
                self.assertLessEqual(save_steps, 100, f"{config_path}:{stage_name} should be resumable on local hardware")


if __name__ == "__main__":
    unittest.main()
