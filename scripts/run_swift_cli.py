#!/usr/bin/env python
from __future__ import annotations

import argparse
import runpy
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an ms-swift CLI subcommand after importing pandas first.")
    parser.add_argument("command", choices=["pt", "sft", "rlhf", "infer"])
    args, remaining = parser.parse_known_args()

    # Workaround for a Windows + CUDA torch import-order failure where importing
    # swift first can fail in pandas -> pyarrow with WinError 6714.
    import pandas  # noqa: F401

    sys.argv = [f"swift {args.command}", *remaining]
    runpy.run_module(f"swift.cli.{args.command}", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
