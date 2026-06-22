from __future__ import annotations

import json
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    meta: dict[str, float] = {"start_time": time.time()}
    try:
        yield meta
    finally:
        meta["end_time"] = time.time()
        meta["elapsed_seconds"] = meta["end_time"] - meta["start_time"]


def query_gpu_memory_mib() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    values = [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]
    return max(values) if values else None


def write_run_metadata(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
