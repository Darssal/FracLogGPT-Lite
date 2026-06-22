#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraclog.data import load_json, load_jsonl, validate_image_refs, write_json
from fraclog.metrics import count_category, parse_fracture_count


def summarize_sft(root: Path) -> dict:
    path = root / "sft" / "conversations.json"
    rows = load_json(path) if path.exists() else []
    labels = {}
    parsed = 0
    for row in rows:
        assistants = [msg["content"] for msg in row.get("messages", []) if msg.get("role") == "assistant"]
        count = parse_fracture_count(assistants[-1]) if assistants else None
        label = count_category(count) or "unknown"
        labels[label] = labels.get(label, 0) + 1
        parsed += int(count is not None)
    return {
        "path": str(path),
        "exists": path.exists(),
        "records": len(rows),
        "count_labels": labels,
        "parsed_count_records": parsed,
        "image_refs": validate_image_refs(rows, root) if rows else {"total": 0, "missing": [], "repaired": 0},
    }


def summarize_preferences(root: Path) -> dict:
    path = root / "rlhf_data" / "pref_triplet.jsonl"
    rows = load_jsonl(path) if path.exists() else []
    return {
        "path": str(path),
        "exists": path.exists(),
        "records": len(rows),
        "keys": sorted(rows[0].keys()) if rows else [],
        "image_refs": validate_image_refs(rows, root) if rows else {"total": 0, "missing": [], "repaired": 0},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect current FracLog data files.")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--out", default="reports/data_audit.json", help="Output audit JSON")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    image_dir = root / "pic"
    audit = {
        "root": str(root),
        "image_dir": str(image_dir),
        "image_dir_exists": image_dir.exists(),
        "image_count": len(list(image_dir.glob("*.png"))) if image_dir.exists() else 0,
        "sft": summarize_sft(root),
        "preference": summarize_preferences(root),
        "notes": [
            "Existing JSON files reference pic_4/*.png; this audit counts those as repairable when pic/*.png exists.",
            "Original paper split files are not present in the current folder.",
        ],
    }
    write_json(root / args.out, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
