from __future__ import annotations

import re
from pathlib import Path
from typing import Any

CHECKPOINT_PATTERN = re.compile(r"^\$\{(?P<kind>checkpoint|latest):(?P<value>[^}]+)\}$")


def is_checkpoint_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    marker_names = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "trainer_state.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    if any((path / name).exists() for name in marker_names):
        return True
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def find_latest_checkpoint(root: str | Path) -> Path:
    root_path = Path(root)
    if is_checkpoint_dir(root_path):
        return root_path
    if not root_path.exists():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root_path}")

    candidates = [path for path in root_path.rglob("checkpoint-*") if is_checkpoint_dir(path)]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directory found under: {root_path}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def resolve_placeholders(
    value: Any,
    stage_checkpoints: dict[str, str],
    *,
    allow_unresolved: bool = False,
) -> Any:
    if isinstance(value, dict):
        return {
            key: resolve_placeholders(item, stage_checkpoints, allow_unresolved=allow_unresolved)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_placeholders(item, stage_checkpoints, allow_unresolved=allow_unresolved) for item in value]
    if not isinstance(value, str):
        return value

    match = CHECKPOINT_PATTERN.match(value)
    if not match:
        return value

    kind = match.group("kind")
    payload = match.group("value")
    if kind == "checkpoint":
        checkpoint = stage_checkpoints.get(payload)
        if checkpoint is None:
            if allow_unresolved:
                return value
            raise KeyError(f"Stage checkpoint is not available yet: {payload}")
        return checkpoint

    try:
        return str(find_latest_checkpoint(payload))
    except FileNotFoundError:
        if allow_unresolved:
            return value
        raise
