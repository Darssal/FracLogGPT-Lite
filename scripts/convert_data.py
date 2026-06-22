#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from fraclog.data import (
    convert_preference_record,
    convert_sft_record,
    load_json,
    load_jsonl,
    prompt_without_final_assistant,
    stratified_split,
    write_json,
    write_jsonl,
)
from fraclog.metrics import count_category, parse_fracture_count


def sft_manifest_row(row: dict, index: int) -> dict:
    assistants = [msg["content"] for msg in row.get("messages", []) if msg.get("role") == "assistant"]
    reference = assistants[-1] if assistants else ""
    count = parse_fracture_count(reference)
    image = row.get("images", [""])[0]
    return {
        "id": f"sft-{index:05d}",
        "image": image,
        "reference": reference,
        "count": count,
        "count_label": count_category(count) or "unknown",
    }


def query_record_from_sft(row: dict, manifest: dict) -> dict:
    return {
        "id": manifest["id"],
        "messages": prompt_without_final_assistant(row["messages"]),
        "images": row["images"],
        "reference": manifest["reference"],
        "count": manifest["count"],
        "count_label": manifest["count_label"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert FracLog data to ms-swift-ready JSONL files.")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument("--out-dir", default="outputs/data", help="Converted data output directory")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument("--smoke-size", type=int, default=8, help="Rows to write into smoke files")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    source_sft = load_json(root / "sft" / "conversations.json")
    converted_sft = [convert_sft_record(row) for row in source_sft]
    manifest = [sft_manifest_row(row, idx) for idx, row in enumerate(converted_sft)]
    query_rows = [query_record_from_sft(row, meta) for row, meta in zip(converted_sft, manifest)]

    split_input = [{**row, **meta} for row, meta in zip(converted_sft, manifest)]
    splits = stratified_split(split_input, label_key="count_label", seed=args.seed)

    write_jsonl(out_dir / "sft_all.jsonl", converted_sft)
    write_jsonl(out_dir / "sft_smoke.jsonl", converted_sft[: args.smoke_size])
    write_jsonl(out_dir / "sft_manifest.jsonl", manifest)
    write_jsonl(out_dir / "eval_all.jsonl", query_rows)
    for split_name, split_rows in splits.items():
        train_rows = [{"messages": row["messages"], "images": row["images"]} for row in split_rows]
        eval_rows = [
            {
                "id": row["id"],
                "messages": prompt_without_final_assistant(row["messages"]),
                "images": row["images"],
                "reference": row["reference"],
                "count": row["count"],
                "count_label": row["count_label"],
            }
            for row in split_rows
        ]
        write_jsonl(out_dir / f"sft_{split_name}.jsonl", train_rows)
        write_jsonl(out_dir / f"eval_{split_name}.jsonl", eval_rows)
    write_jsonl(out_dir / "eval_smoke.jsonl", query_rows[: args.smoke_size])

    pref_path = root / "rlhf_data" / "pref_triplet.jsonl"
    pref_rows = load_jsonl(pref_path)
    converted_pref = [convert_preference_record(row) for row in pref_rows]
    write_jsonl(out_dir / "pref_all.jsonl", converted_pref)
    write_jsonl(out_dir / "pref_smoke.jsonl", converted_pref[: args.smoke_size])

    dapt_smoke = [{"text": row["reference"]} for row in manifest[: max(args.smoke_size, 1)] if row["reference"]]
    write_jsonl(out_dir / "dapt_smoke.jsonl", dapt_smoke)

    summary = {
        "seed": args.seed,
        "sft_records": len(converted_sft),
        "preference_records": len(converted_pref),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "count_labels": {label: sum(row["count_label"] == label for row in manifest) for label in sorted({row["count_label"] for row in manifest})},
        "notes": [
            "Split is regenerated because explicit original split files are absent.",
            "DAPT smoke corpus is derived from SFT references and is not a replacement for the original geological DAPT corpus.",
        ],
    }
    write_json(out_dir / "conversion_summary.json", summary)
    print(f"Wrote converted data to {out_dir}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
