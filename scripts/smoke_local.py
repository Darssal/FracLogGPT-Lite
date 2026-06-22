#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(command))
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local smoke checks without full training.")
    parser.add_argument("--include-model-smoke", action="store_true", help="Attempt Qwen3.5-4B image inference; may download weights and can fail on 11GB GPUs.")
    args = parser.parse_args()

    root = Path(".").resolve()
    logs = root / "outputs" / "smoke" / "logs"
    swift_bin = Path(sys.executable).with_name("swift.exe")
    swift_args = ["--swift-bin", str(swift_bin)] if swift_bin.exists() else []
    swift_wrapper_args = ["--swift-wrapper", "scripts/run_swift_cli.py"]
    steps = [
        ([sys.executable, "scripts/inspect_data.py"], logs / "inspect_data.log"),
        ([sys.executable, "scripts/convert_data.py"], logs / "convert_data.log"),
        ([sys.executable, "scripts/prepare_dapt_corpus.py", "--allow-smoke-from-sft", "--out", "outputs/data/dapt_train.jsonl"], logs / "prepare_dapt_corpus.log"),
        ([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], logs / "unit_tests.log"),
        ([sys.executable, "scripts/smoke_ms_swift.py", *swift_args, *swift_wrapper_args], logs / "ms_swift_smoke.log"),
        ([sys.executable, "scripts/run_pipeline.py", "--config", "configs/e1_qwen35_4b_lora_dpo.json", *swift_args, *swift_wrapper_args], logs / "dryrun_e1_pipeline.log"),
        ([sys.executable, "scripts/run_pipeline.py", "--config", "configs/e2_qwen35_4b_lora_simpo.json", *swift_args, *swift_wrapper_args], logs / "dryrun_e2_pipeline.log"),
        ([sys.executable, "scripts/run_pipeline.py", "--config", "configs/e3_qwen35_4b_simpo_frac_vcd.json", "--stages", "infer"], logs / "dryrun_e3_infer.log"),
    ]
    if args.include_model_smoke:
        steps.append(
            (
                [
                    sys.executable,
                    "scripts/infer_transformers.py",
                    "--model",
                    "Qwen/Qwen3.5-4B",
                    "--input",
                    "outputs/data/eval_smoke.jsonl",
                    "--output",
                    "outputs/smoke/qwen35_image_predictions.jsonl",
                    "--limit",
                    "1",
                    "--max-new-tokens",
                    "32",
                ],
                logs / "qwen35_image_smoke.log",
            )
        )

    failures = []
    for command, log_path in steps:
        code = run_step(command, log_path)
        if code:
            failures.append({"command": command, "log": str(log_path), "returncode": code})
    if failures:
        print("Smoke failures:")
        for failure in failures:
            print(failure)
        return 1
    print("Local smoke checks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
