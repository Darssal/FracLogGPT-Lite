#!/usr/bin/env python
from __future__ import annotations

import argparse

from fraclog.dapt import check_dapt_corpus_for_formal


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate whether dapt_train.jsonl is suitable for formal DAPT.")
    parser.add_argument("--path", default="outputs/data/dapt_train.jsonl")
    parser.add_argument(
        "--allow-derived",
        action="store_true",
        help="Allow SFT-derived or unverified DAPT for the current available SFT/DPO annotated-data scope.",
    )
    args = parser.parse_args()

    status = check_dapt_corpus_for_formal(args.path, allow_derived=args.allow_derived)
    print(status["message"])
    print(f"path={status['path']}")
    print(f"source_type={status['source_type']}")
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
