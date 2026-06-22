#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Qwen3.5-4B hub metadata and multimodal processor without downloading model weights.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out", default="outputs/smoke/logs/qwen35_hf_metadata_smoke.json")
    args = parser.parse_args()

    payload: dict[str, Any] = {"model": args.model, "status": "started"}
    try:
        from huggingface_hub import model_info
        from transformers import AutoConfig, AutoProcessor

        info = model_info(args.model)
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        payload.update(
            {
                "status": "ok",
                "model_id": info.modelId,
                "pipeline_tag": info.pipeline_tag,
                "tags": info.tags or [],
                "siblings": [s.rfilename for s in (info.siblings or [])],
                "config_class": config.__class__.__name__,
                "model_type": getattr(config, "model_type", None),
                "architectures": getattr(config, "architectures", None),
                "processor_class": processor.__class__.__name__,
                "has_image_processor": hasattr(processor, "image_processor"),
                "has_chat_template": hasattr(processor, "chat_template") or hasattr(processor, "tokenizer"),
            }
        )
    except Exception as exc:  # pragma: no cover - smoke script records external failures.
        payload.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
