from __future__ import annotations

import unittest

from scripts import summarize_results


class SummarizeResultsScriptTests(unittest.TestCase):
    def test_public_report_order_is_e0_to_e3(self) -> None:
        self.assertEqual(["E0", "E1", "E2", "E3"], summarize_results.EXPERIMENT_ORDER)
        self.assertEqual(set(summarize_results.EXPERIMENT_ORDER), set(summarize_results.EXPERIMENT_TEMPLATES))


if __name__ == "__main__":
    unittest.main()
