#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from fraclog.runtime import write_run_metadata


def flatten_args(args: dict[str, Any]) -> list[str]:
    flat: list[str] = []
    for key, value in args.items():
        if value is None:
            continue
        flat.append(f"--{key}")
        if isinstance(value, bool):
            flat.append("true" if value else "false")
        elif isinstance(value, list):
            flat.append(",".join(str(item) for item in value))
        else:
            flat.append(str(value))
    return flat


def build_command(config: dict[str, Any], stage: str, swift_bin: str, swift_wrapper: str | None = None) -> list[str]:
    stages = config.get("stages", {})
    if stage not in stages:
        raise KeyError(f"Stage {stage!r} is not configured. Available: {sorted(stages)}")
    stage_cfg = stages[stage]
    if stage_cfg["command"] == "script":
        command = [stage_cfg.get("runner", sys.executable), stage_cfg["script"]]
    elif swift_wrapper:
        command = [sys.executable, swift_wrapper, stage_cfg["command"]]
    else:
        command = [swift_bin, stage_cfg["command"]]
    command.extend(flatten_args(stage_cfg.get("args", {})))
    return command


def quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or run ms-swift commands from an experiment config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--swift-bin", default="swift")
    parser.add_argument("--swift-wrapper", default=None, help="Optional Python wrapper for ms-swift CLI subcommands.")
    parser.add_argument("--execute", action="store_true", help="Run the command. Default is dry-run.")
    parser.add_argument("--metadata-out", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    command = build_command(config, args.stage, args.swift_bin, args.swift_wrapper)
    command_text = quote_command(command)
    meta = {"config": str(config_path), "experiment": config.get("experiment"), "stage": args.stage, "command": command}
    print(command_text)

    if args.execute:
        completed = subprocess.run(command, check=False)
        meta["returncode"] = completed.returncode
        if args.metadata_out:
            write_run_metadata(args.metadata_out, meta)
        return completed.returncode

    meta["dry_run"] = True
    if args.metadata_out:
        write_run_metadata(args.metadata_out, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
