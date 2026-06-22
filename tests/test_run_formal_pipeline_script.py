from __future__ import annotations

import unittest
from pathlib import Path


class RunFormalPipelineScriptTests(unittest.TestCase):
    def test_formal_pipeline_allows_derived_dapt_unless_strict(self) -> None:
        script = Path("scripts/run_formal_pipeline.ps1").read_text(encoding="utf-8")

        self.assertIn("[switch]$StrictDapt", script)
        self.assertIn("if (-not $StrictDapt)", script)
        self.assertIn("$daptCheckArgs += '--allow-derived'", script)

    def test_formal_pipeline_exposes_stage_selection(self) -> None:
        script = Path("scripts/run_formal_pipeline.ps1").read_text(encoding="utf-8")

        self.assertIn("[string]$Stages = ''", script)
        self.assertIn("Invoke-FracLogPipeline 'configs\\e1_qwen35_4b_lora_dpo.json' $Stages", script)
        self.assertIn("Invoke-FracLogPipeline 'configs\\e2_qwen35_4b_lora_simpo.json' $Stages", script)

    def test_training_only_stage_does_not_reevaluate_existing_predictions(self) -> None:
        script = Path("scripts/run_formal_pipeline.ps1").read_text(encoding="utf-8")

        self.assertIn("$shouldEvaluate = (-not $Stages) -or ($Stages -match '(^|,)infer(,|$)')", script)
        self.assertIn("if ($Execute -and $shouldEvaluate)", script)
        self.assertIn("if ($Execute -and -not $shouldEvaluate)", script)


if __name__ == "__main__":
    unittest.main()
