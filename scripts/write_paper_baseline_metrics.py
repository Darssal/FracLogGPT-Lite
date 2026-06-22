#!/usr/bin/env python
from __future__ import annotations

import argparse

from fraclog.data import write_json
from fraclog.paper_baseline import build_paper_baseline_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Write paper-reported E0 Full Model metrics in standard FracLog format.")
    parser.add_argument("--out", default="outputs/e0/metrics.json")
    args = parser.parse_args()

    write_json(args.out, build_paper_baseline_metrics())
    print(f"Wrote paper E0 baseline metrics to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
