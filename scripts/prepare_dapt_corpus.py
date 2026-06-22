#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from fraclog.data import load_jsonl
from fraclog.dapt import write_dapt_corpus


def chunk_text(text: str, max_chars: int) -> list[str]:
    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 > max_chars and current:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def rows_from_sft_jsonl(path: Path, max_chars: int, include_user_text: bool = False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row_index, row in enumerate(load_jsonl(path)):
        messages = row.get("messages", [])
        if not isinstance(messages, list):
            continue
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if role == "assistant" or (include_user_text and role == "user"):
                parts.append(content.replace("<image>", "").strip())
        text = "\n".join(part for part in parts if part)
        for chunk_index, chunk in enumerate(chunk_text(text, max_chars)):
            rows.append({"text": chunk, "source": str(path), "row": str(row_index), "chunk": str(chunk_index)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare text JSONL for DAPT.")
    parser.add_argument("--corpus-dir", help="Directory containing .txt or .md DAPT corpus files")
    parser.add_argument("--from-sft-jsonl", help="Build an available-data DAPT corpus from assistant text in an SFT JSONL file")
    parser.add_argument("--out", default="outputs/data/dapt_train.jsonl")
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--include-user-text", action="store_true", help="When using --from-sft-jsonl, include user prompts as well as assistant responses")
    parser.add_argument("--allow-smoke-from-sft", action="store_true", help="Use outputs/data/dapt_smoke.jsonl when no corpus is available")
    args = parser.parse_args()

    out = Path(args.out)
    rows: list[dict[str, str]] = []
    source_type = "unverified"
    source: str | None = None
    if args.corpus_dir:
        corpus_dir = Path(args.corpus_dir)
        if not corpus_dir.exists():
            raise FileNotFoundError(f"DAPT corpus directory not found: {corpus_dir}")
        for path in sorted(list(corpus_dir.glob("*.txt")) + list(corpus_dir.glob("*.md"))):
            text = path.read_text(encoding="utf-8", errors="replace")
            for idx, chunk in enumerate(chunk_text(text, args.max_chars)):
                rows.append({"text": chunk, "source": str(path), "chunk": str(idx)})
        source_type = "real_corpus"
        source = str(corpus_dir)
    elif args.from_sft_jsonl:
        sft_path = Path(args.from_sft_jsonl)
        if not sft_path.exists():
            raise FileNotFoundError(f"SFT JSONL not found: {sft_path}")
        rows = rows_from_sft_jsonl(sft_path, args.max_chars, include_user_text=args.include_user_text)
        source_type = "from_sft_jsonl"
        source = str(sft_path)
    elif args.allow_smoke_from_sft:
        smoke = Path("outputs/data/dapt_smoke.jsonl")
        rows = load_jsonl(smoke) if smoke.exists() else []
        source_type = "smoke_from_sft"
        source = str(smoke)
    else:
        raise SystemExit("No DAPT corpus provided. Pass --corpus-dir or explicit --allow-smoke-from-sft for local smoke only.")

    if not rows:
        raise SystemExit("No DAPT rows were produced.")
    write_dapt_corpus(
        out,
        rows,
        source_type=source_type,
        source=source,
        max_chars=args.max_chars,
        include_user_text=args.include_user_text,
    )
    print(f"Wrote {len(rows)} DAPT rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
