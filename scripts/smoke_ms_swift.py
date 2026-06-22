#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SMOKE_STAGES = [
    {"stage": "dapt", "label": "LoRA-DAPT smoke"},
    {"stage": "sft", "label": "LoRA-SFT smoke"},
    {"stage": "rlhf", "label": "DPO smoke"},
    {"stage": "rlhf_simpo", "label": "SimPO smoke"},
]


def check_import(module: str) -> dict[str, Any]:
    if module == "swift":
        code = "import pandas; import swift; print(getattr(swift, '__file__', 'built-in'))"
    else:
        code = f"import {module}; print(getattr({module}, '__file__', 'built-in'))"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "module": module,
        "ok": completed.returncode == 0,
        "output": completed.stdout.strip(),
    }


def command_text(command: list[str]) -> str:
    return " ".join(command)


def kill_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:  # pragma: no cover - Windows is the current local target.
        subprocess.run(["pkill", "-TERM", "-P", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_stage(config: str, stage: str, swift_bin: str, swift_wrapper: str | None, execute: bool, timeout_s: int) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/run_ms_swift.py",
        "--config",
        config,
        "--stage",
        stage,
        "--swift-bin",
        swift_bin,
    ]
    if swift_wrapper:
        command.extend(["--swift-wrapper", swift_wrapper])
    if execute:
        command.append("--execute")
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        stdout, _ = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        kill_process_tree(process.pid)
        stdout, _ = process.communicate()
        return {
            "stage": stage,
            "wrapper_command": command,
            "returncode": None,
            "status": "timeout",
            "timeout_s": timeout_s,
            "stdout": (stdout or "").strip(),
        }
    completed_returncode = process.returncode
    return {
        "stage": stage,
        "wrapper_command": command,
        "returncode": completed_returncode,
        "status": "ok" if completed_returncode == 0 else "failed",
        "stdout": (stdout or "").strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check and optionally execute ms-swift DAPT/SFT/DPO/SimPO smoke stages.")
    parser.add_argument("--config", default="configs/smoke_qwen35_4b.json")
    parser.add_argument("--swift-bin", default="swift")
    parser.add_argument("--swift-wrapper", default=None)
    parser.add_argument("--execute", action="store_true", help="Run max_steps=1 smoke training. Default only dry-runs.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if dependencies are missing in dry-run mode.")
    parser.add_argument("--stage-timeout-s", type=int, default=600, help="Per-stage timeout when executing smoke training.")
    parser.add_argument("--log", default="outputs/smoke/logs/ms_swift_smoke.json")
    args = parser.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    swift_path = shutil.which(args.swift_bin)
    dependency_checks = {
        "swift_cli": {"ok": swift_path is not None, "path": swift_path},
        "python_imports": [check_import("torch"), check_import("swift")],
    }
    missing = []
    if not dependency_checks["swift_cli"]["ok"]:
        missing.append(f"{args.swift_bin} CLI")
    for check in dependency_checks["python_imports"]:
        if not check["ok"]:
            missing.append(f"python import {check['module']}")

    stages = []
    if args.execute and missing:
        status = "blocked"
        for item in SMOKE_STAGES:
            stages.append(
                {
                    **item,
                    "status": "not_run",
                    "minimal_repro": command_text(
                        [
                            sys.executable,
                            "scripts/run_ms_swift.py",
                            "--config",
                            args.config,
                            "--stage",
                            item["stage"],
                            "--swift-bin",
                            args.swift_bin,
                            *([] if args.swift_wrapper is None else ["--swift-wrapper", args.swift_wrapper]),
                            "--execute",
                        ]
                    ),
                }
            )
    else:
        status = "dry_run_completed"
        if args.execute:
            status = "executed"
        for item in SMOKE_STAGES:
            result = run_stage(args.config, item["stage"], args.swift_bin, args.swift_wrapper, args.execute, args.stage_timeout_s)
            stages.append({**item, **result})
            if result.get("status") == "timeout":
                status = "timeout"
                if args.execute:
                    break
            elif result["returncode"] != 0:
                status = "failed"
                if args.execute:
                    break

    payload = {
        "status": status,
        "execute": args.execute,
        "config": args.config,
        "dependency_checks": dependency_checks,
        "missing": missing,
        "stages": stages,
    }
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if status in {"failed", "blocked", "timeout"} and (args.execute or args.strict):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
