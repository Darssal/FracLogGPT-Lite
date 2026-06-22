from __future__ import annotations

from pathlib import Path
from typing import Any

from fraclog.data import load_json, write_json, write_jsonl


REAL_DAPT_SOURCE_TYPE = "real_corpus"
DERIVED_DAPT_SOURCE_TYPES = {"from_sft_jsonl", "smoke_from_sft", "unverified"}


def metadata_path(corpus_path: str | Path) -> Path:
    path = Path(corpus_path)
    return path.with_suffix(".meta.json")


def build_dapt_metadata(
    *,
    source_type: str,
    source: str | None,
    rows: int,
    max_chars: int,
    include_user_text: bool,
) -> dict[str, Any]:
    return {
        "source_type": source_type,
        "source": source,
        "rows": rows,
        "max_chars": max_chars,
        "include_user_text": include_user_text,
    }


def write_dapt_corpus(
    out: str | Path,
    rows: list[dict[str, Any]],
    *,
    source_type: str,
    source: str | None,
    max_chars: int,
    include_user_text: bool,
) -> None:
    out_path = Path(out)
    write_jsonl(out_path, rows)
    write_json(
        metadata_path(out_path),
        build_dapt_metadata(
            source_type=source_type,
            source=source,
            rows=len(rows),
            max_chars=max_chars,
            include_user_text=include_user_text,
        ),
    )


def read_dapt_metadata(corpus_path: str | Path) -> dict[str, Any] | None:
    path = metadata_path(corpus_path)
    if not path.exists():
        return None
    payload = load_json(path)
    return payload if isinstance(payload, dict) else None


def check_dapt_corpus_for_formal(corpus_path: str | Path, *, allow_derived: bool = False) -> dict[str, Any]:
    path = Path(corpus_path)
    if not path.exists():
        return {
            "ok": False,
            "path": str(path),
            "source_type": None,
            "message": f"DAPT corpus is missing: {path}",
        }
    metadata = read_dapt_metadata(path)
    source_type = metadata.get("source_type") if metadata else "unverified"
    if source_type == REAL_DAPT_SOURCE_TYPE or allow_derived:
        return {
            "ok": True,
            "path": str(path),
            "source_type": source_type,
            "message": "DAPT corpus accepted for formal execution.",
        }
    return {
        "ok": False,
        "path": str(path),
        "source_type": source_type,
        "message": (
            "Current annotated-data runs may allow this SFT-derived DAPT corpus as a caveat; "
            "paper-strict reproduction requires a real geological DAPT corpus. "
            "Rebuild with scripts/prepare_dapt_corpus.py --corpus-dir data/dapt_corpus "
            "or pass --allow-derived when the intended scope is the available SFT/DPO annotated data."
        ),
    }
