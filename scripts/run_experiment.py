#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or dry-run a configured experiment stage sequence.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stages", default="dapt,sft,rlhf,infer")
    parser.add_argument("--swift-bin", default="swift")
    parser.add_argument("--swift-wrapper", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--commands-out", default="outputs/run_commands")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured = config.get("stages", {})
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    selected = [stage for stage in stages if stage in configured]
    if not selected:
        raise SystemExit(f"No selected stages are configured. Selected={stages}, configured={sorted(configured)}")

    commands_dir = Path(args.commands_out)
    commands_dir.mkdir(parents=True, exist_ok=True)
    command_file = commands_dir / f"{config.get('experiment', config_path.stem).lower()}_commands.ps1"
    lines = []
    exit_code = 0
    for stage in selected:
        command = [
            sys.executable,
            "scripts/run_ms_swift.py",
            "--config",
            str(config_path),
            "--stage",
            stage,
            "--swift-bin",
            args.swift_bin,
        ]
        if args.swift_wrapper:
            command.extend(["--swift-wrapper", args.swift_wrapper])
        if args.execute:
            command.append("--execute")
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(completed.stdout)
        lines.append(completed.stdout.strip())
        if completed.returncode:
            exit_code = completed.returncode
            if args.execute:
                break
    command_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote command log to {command_file}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
