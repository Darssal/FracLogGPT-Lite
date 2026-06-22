from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fraclog.data import load_jsonl, write_jsonl


def prediction_key(row: dict[str, Any]) -> str:
    if row.get("id") is not None:
        return f"id:{row['id']}"
    if row.get("image") is not None:
        return f"image:{row['image']}"
    images = row.get("images")
    if isinstance(images, list) and images:
        return f"image:{images[0]}"
    raise ValueError(f"Cannot derive stable prediction key from row: {row}")


def existing_prediction_keys(path: str | Path) -> set[str]:
    prediction_path = Path(path)
    if not prediction_path.exists() or prediction_path.stat().st_size == 0:
        return set()
    return {prediction_key(row) for row in load_jsonl(prediction_path)}


def dedupe_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for row in rows:
        keyed[prediction_key(row)] = row
    return list(keyed.values())


def dedupe_prediction_file(path: str | Path) -> tuple[int, int]:
    prediction_path = Path(path)
    rows = load_jsonl(prediction_path)
    deduped = dedupe_prediction_rows(rows)
    write_jsonl(prediction_path, deduped)
    return len(rows), len(deduped)


@contextmanager
def prediction_file_lock(path: str | Path):
    lock_path = Path(f"{Path(path)}.lock")
    try:
        lock_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(f"Prediction output is already locked: {lock_path}") from exc
    try:
        yield lock_path
    finally:
        lock_path.rmdir()


def select_rows_for_prediction(rows: list[dict[str, Any]], output_path: str | Path, *, resume: bool = False) -> list[dict[str, Any]]:
    if not resume:
        return list(rows)
    completed = existing_prediction_keys(output_path)
    return [row for row in rows if prediction_key(row) not in completed]
