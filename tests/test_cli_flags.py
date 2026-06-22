from __future__ import annotations

import unittest

from scripts.infer_frac_vcd import parse_bool as parse_vcd_bool
from scripts.infer_transformers import parse_bool as parse_transformers_bool


class CliFlagTests(unittest.TestCase):
    def test_resume_bool_parser_accepts_config_style_values(self) -> None:
        self.assertTrue(parse_transformers_bool("true"))
        self.assertFalse(parse_transformers_bool("false"))
        self.assertTrue(parse_vcd_bool("true"))
        self.assertFalse(parse_vcd_bool("false"))


if __name__ == "__main__":
    unittest.main()
