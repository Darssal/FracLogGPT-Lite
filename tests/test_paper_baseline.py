from __future__ import annotations

import unittest

from fraclog.paper_baseline import build_paper_baseline_metrics


class PaperBaselineTests(unittest.TestCase):
    def test_build_paper_baseline_metrics_contains_public_e0_values(self) -> None:
        metrics = build_paper_baseline_metrics()

        self.assertEqual(metrics["experiment"], "E0")
        self.assertEqual(metrics["model"], "Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(metrics["samples"], 160)
        self.assertAlmostEqual(metrics["count_precision"], 0.708)
        self.assertAlmostEqual(metrics["count_recall"], 0.700)
        self.assertAlmostEqual(metrics["count_f1"], 0.697)
        self.assertAlmostEqual(metrics["loc_acc"], 0.488)
        self.assertAlmostEqual(metrics["morph_acc"], 0.431)
        self.assertAlmostEqual(metrics["avg_bleu"], 0.592)
        self.assertAlmostEqual(metrics["avg_rouge_l"], 0.670)
        self.assertIn("MDPI Electronics 2025, Table 5", metrics["source"])
        self.assertIn("directional reference", metrics["comparison_scope"])
        self.assertFalse(metrics["same_test_set"])


if __name__ == "__main__":
    unittest.main()
