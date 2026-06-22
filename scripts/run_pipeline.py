#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from fraclog.checkpoints import find_latest_checkpoint, resolve_placeholders
from fraclog.runtime import timed, write_run_metadata
from run_ms_swift import build_command, quote_command


TRAINING_COMMANDS = {"pt", "sft", "rlhf"}


def configured_stage_order(config: dict[str, Any], selected: str | None) -> list[str]:
    stages = config.get("stages", {})
    if selected:
        requested = [stage.strip() for stage in selected.split(",") if stage.strip()]
    else:
        requested = list(stages)
    resolved = [stage for stage in requested if stage in stages]
    if not resolved:
        raise SystemExit(f"No requested stages are configured. requested={requested}, configured={sorted(stages)}")
    return resolved


def load_config_with_resolved_args(
    config: dict[str, Any],
    stage: str,
    stage_checkpoints: dict[str, str],
    *,
    allow_unresolved: bool,
) -> dict[str, Any]:
    resolved = json.loads(json.dumps(config))
    stage_args = resolved["stages"][stage].get("args", {})
    resolved["stages"][stage]["args"] = resolve_placeholders(
        stage_args,
        stage_checkpoints,
        allow_unresolved=allow_unresolved,
    )
    return resolved


def _rooted_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def seed_stage_checkpoints_from_existing(config: dict[str, Any], *, root: str | Path = ".") -> dict[str, str]:
    root_path = Path(root)
    checkpoints: dict[str, str] = {}
    for stage, stage_cfg in config.get("stages", {}).items():
        if stage_cfg.get("command") not in TRAINING_COMMANDS:
            continue
        output_dir = stage_cfg.get("args", {}).get("output_dir")
        if not output_dir:
            continue
        try:
            checkpoints[stage] = str(find_latest_checkpoint(_rooted_path(root_path, output_dir)))
        except FileNotFoundError:
            continue
    return checkpoints


def stage_checkpoint_after_success(config: dict[str, Any], stage: str) -> str | None:
    stage_cfg = config["stages"][stage]
    if stage_cfg.get("command") not in TRAINING_COMMANDS:
        return None
    output_dir = stage_cfg.get("args", {}).get("output_dir")
    if not output_dir:
        return None
    return str(find_latest_checkpoint(output_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run chained FracLog experiment stages with checkpoint resolution.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stages", default=None, help="Comma-separated stages. Defaults to configured order.")
    parser.add_argument("--swift-bin", default="swift")
    parser.add_argument("--swift-wrapper", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--commands-out", default="outputs/run_commands")
    parser.add_argument("--state-out", default=None)
    parser.add_argument(
        "--resume-existing-checkpoints",
        action="store_true",
        help="Preload latest checkpoints from configured output_dir values so downstream stages can be resumed.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiment = config.get("experiment", config_path.stem).lower()
    selected_stages = configured_stage_order(config, args.stages)

    commands_dir = Path(args.commands_out)
    commands_dir.mkdir(parents=True, exist_ok=True)
    command_file = commands_dir / f"{experiment}_pipeline_commands.ps1"
    state_file = Path(args.state_out) if args.state_out else Path("outputs/run_state") / f"{experiment}.json"

    stage_checkpoints: dict[str, str] = (
        seed_stage_checkpoints_from_existing(config) if args.resume_existing_checkpoints else {}
    )
    stage_records: list[dict[str, Any]] = []
    command_lines: list[str] = []
    exit_code = 0

    for stage in selected_stages:
        resolved_config = load_config_with_resolved_args(
            config,
            stage,
            stage_checkpoints,
            allow_unresolved=not args.execute,
        )
        command = build_command(resolved_config, stage, args.swift_bin, args.swift_wrapper)
        command_text = quote_command(command)
        print(command_text)
        command_lines.append(command_text)

        record: dict[str, Any] = {
            "stage": stage,
            "command": command,
            "dry_run": not args.execute,
        }
        if args.execute:
            with timed() as timing:
                completed = subprocess.run(command, check=False)
            record.update(timing)
            record["returncode"] = completed.returncode
            if completed.returncode:
                exit_code = completed.returncode
                stage_records.append(record)
                break
            checkpoint = stage_checkpoint_after_success(resolved_config, stage)
            if checkpoint:
                stage_checkpoints[stage] = checkpoint
                record["checkpoint"] = checkpoint

        stage_records.append(record)

    command_file.write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    state = {
        "config": str(config_path),
        "experiment": config.get("experiment"),
        "stages": stage_records,
        "stage_checkpoints": stage_checkpoints,
        "commands_file": str(command_file),
    }
    write_run_metadata(state_file, state)
    print(f"Wrote command log to {command_file}")
    print(f"Wrote pipeline state to {state_file}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
