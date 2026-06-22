from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


VISUAL_INPUT_KEYS = {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}


@dataclass
class VCDConfig:
    alpha: float = 1.0
    max_new_tokens: int = 256
    temperature: float = 0.2
    perturbation: str = "blur"
    blur_radius: float = 6.0


def make_perturbed_image(image_path: str | Path, output_path: str | Path, *, mode: str = "blur", blur_radius: float = 6.0) -> Path:
    try:
        from PIL import Image, ImageFilter, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow is required for Frac-VCD image perturbation") from exc

    source = Image.open(image_path).convert("RGB")
    if mode == "blur":
        perturbed = source.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    elif mode == "gray":
        perturbed = ImageOps.grayscale(source).convert("RGB")
    elif mode == "downsample":
        small = source.resize((max(1, source.width // 4), max(1, source.height // 4)))
        perturbed = small.resize(source.size)
    else:
        raise ValueError(f"Unsupported VCD perturbation mode: {mode}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    perturbed.save(out)
    return out


def fuse_logits(original_logits: Any, perturbed_logits: Any, alpha: float = 1.0) -> Any:
    return original_logits + alpha * (original_logits - perturbed_logits)


@dataclass
class BatchedVCDLogitsProcessor:
    alpha: float = 1.0

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        batch_size = int(scores.shape[0])
        if batch_size != 2:
            raise ValueError(f"Frac-VCD logits processor expects batch size 2, got {batch_size}")
        fused = fuse_logits(scores[0:1], scores[1:2], alpha=self.alpha)
        return fused.expand_as(scores).clone()


def sample_next_token(logits: Any, *, temperature: float = 0.2) -> Any:
    import torch

    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def append_generated_text_token(inputs: dict[str, Any], next_token: Any) -> None:
    """Append one generated text token to Qwen-style multimodal inputs in-place."""
    import torch

    inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=-1)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(next_token)], dim=-1)
    if "mm_token_type_ids" in inputs:
        # Qwen3.5 uses 0 for text tokens; generated continuations are text.
        inputs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros_like(next_token)], dim=-1)


def _detach_visual_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in inputs.items() if key not in VISUAL_INPUT_KEYS}


def _prepare_cached_step_inputs(model: Any, state: dict[str, Any], past_key_values: Any) -> dict[str, Any]:
    prepared = {"input_ids": state["input_ids"][:, -1:]}
    prepared["past_key_values"] = past_key_values
    prepared["use_cache"] = True
    return prepared


def visual_contrastive_generate(
    model: Any,
    processor: Any,
    original_inputs: dict[str, Any],
    perturbed_inputs: dict[str, Any],
    *,
    config: VCDConfig | None = None,
    eos_token_id: int | None = None,
) -> list[int]:
    """Generic reference Frac-VCD decoder for Qwen-style multimodal models.

    The caller is responsible for building model inputs for the original and
    perturbed images with the same prompt. This helper keeps both branches in
    sync and fuses the next-token logits at every decoding step.
    """
    import torch

    cfg = config or VCDConfig()
    generated: list[int] = []
    orig = {key: value for key, value in original_inputs.items()}
    pert = {key: value for key, value in perturbed_inputs.items()}
    if eos_token_id is None:
        eos_token_id = getattr(getattr(processor, "tokenizer", processor), "eos_token_id", None)

    with torch.no_grad():
        for _ in range(cfg.max_new_tokens):
            out_orig = model(**orig)
            out_pert = model(**pert)
            fused = fuse_logits(out_orig.logits[:, -1, :], out_pert.logits[:, -1, :], alpha=cfg.alpha)
            next_token = sample_next_token(fused, temperature=cfg.temperature)
            token_id = int(next_token[0, 0].item())
            generated.append(token_id)
            if eos_token_id is not None and token_id == eos_token_id:
                break
            append_generated_text_token(orig, next_token)
            append_generated_text_token(pert, next_token)
    return generated


def visual_contrastive_generate_batched(
    model: Any,
    processor: Any,
    paired_inputs: dict[str, Any],
    *,
    config: VCDConfig | None = None,
    eos_token_id: int | None = None,
) -> list[int]:
    """Cached Frac-VCD decoder using a batch of [original, perturbed].

    Qwen3.5's linear-attention cache is not safe to alternate as two separate
    cache objects on one model instance on this CUDA setup. A single batch cache
    keeps the two visual branches synchronized and avoids that failure mode.
    """
    import torch

    cfg = config or VCDConfig()
    state = {key: value for key, value in paired_inputs.items()}
    batch_size = int(state["input_ids"].shape[0])
    if batch_size != 2:
        raise ValueError(f"Frac-VCD batched decoding expects batch size 2, got {batch_size}")
    if eos_token_id is None:
        eos_token_id = getattr(getattr(processor, "tokenizer", processor), "eos_token_id", None)

    generated: list[int] = []
    past_key_values = None
    with torch.no_grad():
        for _ in range(cfg.max_new_tokens):
            if past_key_values is None:
                outputs = model(**state, use_cache=True)
            else:
                outputs = model(**_prepare_cached_step_inputs(model, state, past_key_values))
            past_key_values = getattr(outputs, "past_key_values", None)
            logits = outputs.logits[:, -1, :]
            fused = fuse_logits(logits[0:1], logits[1:2], alpha=cfg.alpha)
            next_token = sample_next_token(fused, temperature=cfg.temperature)
            token_id = int(next_token[0, 0].item())
            generated.append(token_id)
            if eos_token_id is not None and token_id == eos_token_id:
                break
            append_generated_text_token(state, next_token.expand(batch_size, -1))
    return generated


def visual_contrastive_generate_hf(
    model: Any,
    processor: Any,
    paired_inputs: dict[str, Any],
    *,
    config: VCDConfig | None = None,
    eos_token_id: int | None = None,
) -> list[int]:
    """Frac-VCD decoder that delegates cache and position handling to HF generate."""

    cfg = config or VCDConfig()
    if cfg.temperature > 0:
        raise ValueError("HF generate Frac-VCD path currently requires temperature <= 0 to keep both branches synchronized")
    state = {key: value for key, value in paired_inputs.items()}
    batch_size = int(state["input_ids"].shape[0])
    if batch_size != 2:
        raise ValueError(f"Frac-VCD HF generate expects batch size 2, got {batch_size}")
    if eos_token_id is None:
        eos_token_id = getattr(getattr(processor, "tokenizer", processor), "eos_token_id", None)
    pad_token_id = getattr(getattr(processor, "tokenizer", processor), "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    from transformers import LogitsProcessorList

    prompt_len = int(state["input_ids"].shape[-1])
    generated = model.generate(
        **state,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
        logits_processor=LogitsProcessorList([BatchedVCDLogitsProcessor(alpha=cfg.alpha)]),
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    return generated[0, prompt_len:].tolist()
