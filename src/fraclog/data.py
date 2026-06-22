from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_IMAGE_DIR = "pic"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_image_ref(ref: str, image_dir: str = DEFAULT_IMAGE_DIR) -> str:
    name = Path(ref).name
    if not name:
        raise ValueError(f"Image reference has no filename: {ref!r}")
    return str(Path(image_dir) / name).replace("\\", "/")


def resolve_image_path(ref: str, root: str | Path, image_dir: str = DEFAULT_IMAGE_DIR) -> Path:
    root_path = Path(root)
    candidates = [
        root_path / ref,
        root_path / normalize_image_ref(ref, image_dir=image_dir),
        root_path / Path(ref).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve image reference {ref!r}; checked: {checked}")


def _copy_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Record must contain a non-empty messages list")
    copied: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"messages[{index}] must contain string role and content")
        copied.append({"role": role, "content": content})
    return copied


def _normalize_images(record: dict[str, Any], image_dir: str = DEFAULT_IMAGE_DIR) -> list[str]:
    images = record.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Record must contain a non-empty images list")
    normalized: list[str] = []
    for ref in images:
        if not isinstance(ref, str):
            raise ValueError(f"Image reference must be a string: {ref!r}")
        normalized.append(normalize_image_ref(ref, image_dir=image_dir))
    return normalized


def convert_sft_record(record: dict[str, Any], image_dir: str = DEFAULT_IMAGE_DIR) -> dict[str, Any]:
    return {
        "messages": _copy_messages(record),
        "images": _normalize_images(record, image_dir=image_dir),
    }


def convert_preference_record(record: dict[str, Any], image_dir: str = DEFAULT_IMAGE_DIR) -> dict[str, Any]:
    chosen = record.get("chosen", record.get("ok_reply"))
    rejected = record.get("rejected", record.get("reply"))
    if not isinstance(chosen, str) or not chosen.strip():
        raise ValueError("Preference record must contain non-empty chosen or ok_reply")
    if not isinstance(rejected, str) or not rejected.strip():
        raise ValueError("Preference record must contain non-empty rejected or reply")

    messages = _copy_messages(record)
    if messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    messages.append({"role": "assistant", "content": chosen})

    return {
        "messages": messages,
        "rejected_response": rejected,
        "images": _normalize_images(record, image_dir=image_dir),
    }


def prompt_without_final_assistant(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if not messages:
        return []
    prompt = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
    if prompt[-1]["role"] == "assistant":
        return prompt[:-1]
    return prompt


def validate_image_refs(rows: Iterable[dict[str, Any]], root: str | Path, image_dir: str = DEFAULT_IMAGE_DIR) -> dict[str, Any]:
    total = 0
    missing: list[dict[str, str]] = []
    repaired = 0
    for row_index, row in enumerate(rows):
        for ref in row.get("images", []):
            total += 1
            original = Path(root) / ref
            try:
                resolved = resolve_image_path(ref, root, image_dir=image_dir)
            except FileNotFoundError:
                missing.append({"row": str(row_index), "image": str(ref)})
                continue
            if resolved != original:
                repaired += 1
    return {"total": total, "missing": missing, "repaired": repaired}


def stratified_split(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = row.get(label_key)
        if label is None:
            raise ValueError(f"Row is missing label key {label_key!r}: {row}")
        grouped[str(label)].append(row)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for label in sorted(grouped):
        bucket = list(grouped[label])
        rng.shuffle(bucket)
        n = len(bucket)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)
        if n >= 3 and val_ratio > 0:
            train_n = max(1, train_n)
            val_n = max(1, val_n)
            if train_n + val_n >= n:
                val_n = max(0, n - train_n - 1)
        splits["train"].extend(bucket[:train_n])
        splits["val"].extend(bucket[train_n : train_n + val_n])
        splits["test"].extend(bucket[train_n + val_n :])

    for split_rows in splits.values():
        split_rows.sort(key=lambda row: str(row.get("id", row.get("images", row))))
    return splits
