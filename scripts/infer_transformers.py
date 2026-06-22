#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fraclog.data import load_jsonl, resolve_image_path
from fraclog.inference import append_final_user_instruction, build_mm_messages
from fraclog.predictions import prediction_file_lock, select_rows_for_prediction


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run normal Transformers multimodal inference.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", default="outputs/data/eval_test.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--adapter")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--final-instruction", help="Optional instruction appended to the last user turn at inference time.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Append to an existing prediction file and skip completed rows.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_file_lock(output_path):
        rows = select_rows_for_prediction(rows, output_path, resume=args.resume)
        if not rows:
            print(f"No pending rows for {args.output}")
            return 0

        import torch
        from PIL import Image
        from transformers import AutoProcessor
        import transformers

        model_cls = (
            getattr(transformers, "AutoModelForImageTextToText", None)
            or getattr(transformers, "AutoModelForMultimodalLM", None)
            or getattr(transformers, "AutoModelForCausalLM")
        )
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        dtype = "auto" if args.torch_dtype == "auto" else getattr(torch, args.torch_dtype)
        model = model_cls.from_pretrained(args.model, torch_dtype=dtype, device_map=args.device, trust_remote_code=True)
        if args.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

        mode = "a" if args.resume else "w"
        with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(rows, start=1):
                image_path = resolve_image_path(row["images"][0], root)
                image = Image.open(image_path).convert("RGB")
                messages = append_final_user_instruction(build_mm_messages(row, image_path), args.final_instruction)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
                start = time.perf_counter()
                with torch.no_grad():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                latency = time.perf_counter() - start
                new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
                prediction = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
                payload = {
                    "id": row.get("id"),
                    "image": row["images"][0],
                    "reference": row.get("reference", ""),
                    "prediction": prediction,
                    "latency_s": latency,
                }
                handle.write(json.dumps(payload, ensure_ascii=False))
                handle.write("\n")
                handle.flush()
                print(f"Processed {index}/{len(rows)} -> {args.output} ({latency:.2f}s)")
        print(f"Wrote {len(rows)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
