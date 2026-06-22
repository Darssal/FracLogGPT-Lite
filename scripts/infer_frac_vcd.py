#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from fraclog.data import load_jsonl, resolve_image_path
from fraclog.inference import append_final_user_instruction, build_mm_messages
from fraclog.predictions import prediction_file_lock, select_rows_for_prediction
from fraclog.vcd import VCDConfig, make_perturbed_image, visual_contrastive_generate_batched, visual_contrastive_generate_hf


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
    parser = argparse.ArgumentParser(description="Run Frac-VCD visual contrastive decoding inference.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", default="outputs/data/eval_test.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--adapter")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--final-instruction", help="Optional instruction appended to the last user turn at inference time.")
    parser.add_argument("--perturbation", default="blur", choices=["blur", "gray", "downsample"])
    parser.add_argument("--decoder", default="cache", choices=["cache", "hf-generate"])
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
    cfg = VCDConfig(alpha=args.alpha, temperature=args.temperature, max_new_tokens=args.max_new_tokens, perturbation=args.perturbation)
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
        model = model_cls.from_pretrained(args.model, torch_dtype="auto", device_map=args.device, trust_remote_code=True)
        if args.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

        mode = "a" if args.resume else "w"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
                for index, row in enumerate(rows, start=1):
                    image_path = resolve_image_path(row["images"][0], root)
                    perturbed_path = make_perturbed_image(image_path, tmp_path / Path(image_path).name, mode=args.perturbation)
                    original_image = Image.open(image_path).convert("RGB")
                    perturbed_image = Image.open(perturbed_path).convert("RGB")
                    messages = append_final_user_instruction(build_mm_messages(row, image_path), args.final_instruction)
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    paired_inputs = processor(text=[text, text], images=[original_image, perturbed_image], return_tensors="pt").to(model.device)
                    start = time.perf_counter()
                    if args.decoder == "hf-generate":
                        token_ids = visual_contrastive_generate_hf(model, processor, paired_inputs, config=cfg)
                    else:
                        token_ids = visual_contrastive_generate_batched(model, processor, paired_inputs, config=cfg)
                    latency = time.perf_counter() - start
                    prediction = processor.batch_decode([token_ids], skip_special_tokens=True)[0]
                    payload = {
                        "id": row.get("id"),
                        "image": row["images"][0],
                        "reference": row.get("reference", ""),
                        "prediction": prediction,
                        "latency_s": latency,
                        "frac_vcd": {
                            "alpha": args.alpha,
                            "perturbation": args.perturbation,
                            "temperature": args.temperature,
                            "decoder": args.decoder,
                        },
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False))
                    handle.write("\n")
                    handle.flush()
                    print(f"Processed {index}/{len(rows)} -> {args.output} ({latency:.2f}s)")
        print(f"Wrote {len(rows)} Frac-VCD predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
