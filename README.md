# FracLogGPT-Lite

FracLogGPT-Lite is a reproducible experiment harness for lightweight fracture interpretation from imaging-log images. It keeps the original data split and prompt style configurable, uses `ms-swift` for DAPT/SFT/preference optimization, and adds an inference-only Frac-VCD decoder for visual-grounding checks.

## Experiment Matrix

| ID | Model | Training / Inference Path |
|---|---|---|
| E0 | `Qwen/Qwen2.5-VL-7B-Instruct` | DAPT + LoRA-SFT + DPO baseline reproduction or imported baseline predictions |
| E1 | `Qwen/Qwen3.5-4B` | DAPT + LoRA-SFT + DPO |
| E2 | `Qwen/Qwen3.5-4B` | DAPT + LoRA-SFT + SimPO |
| E3 | `Qwen/Qwen3.5-4B` | E2 checkpoint + inference-only Frac-VCD visual contrastive decoding |

Frac-VCD does not retrain model parameters. It compares logits from the original image and a perturbed image during decoding to reduce unsupported fracture counts and descriptions.

## Public Repository Scope

This repository is safe to publish as code. It intentionally excludes:

- Raw well-log images and annotation files.
- Local train/validation/test splits when they contain private samples.
- Model weights, LoRA adapters, checkpoints, generated predictions, metrics, logs, and reports.
- Manuscript drafts, blind-review packets, author information, acceptance checklists, and internal experiment notes.
- Credentials, tokens, local account names, and machine-specific absolute paths.

Place private data and generated outputs only in ignored local directories. See [docs/REPOSITORY_SCOPE.md](docs/REPOSITORY_SCOPE.md) and [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md).

## Repository Layout

```text
configs/       Public experiment configs for E0-E3 and smoke checks.
docs/          Public setup, data schema, and release-scope notes.
examples/      Synthetic examples showing expected file formats.
scripts/       Data conversion, training, inference, evaluation, and utility entrypoints.
src/fraclog/   Shared Python utilities for data IO, metrics, inference, checkpoints, and VCD.
tests/         Unit and script-level regression tests for the public code path.
pic/           Ignored local image pool placeholder.
sft/           Ignored local SFT annotation placeholder.
rlhf_data/     Ignored local preference annotation placeholder.
outputs/       Ignored checkpoints, predictions, metrics, and logs.
reports/       Ignored generated summaries.
third_party/   Ignored external project clones.
```

## Setup

```powershell
py -3.10 -m venv .venv310
.\.venv310\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

Install the CUDA-specific PyTorch build that matches your driver before GPU training or inference.

## Data Placement

Use the same local layout as the original project, but keep these files out of Git:

```text
sft/conversations.json
sft/sft.xlsx
rlhf_data/pref_triplet.jsonl
rlhf_data/pref_query.jsonl
pic/*.png
```

The conversion script inspects available source files before producing normalized files under `outputs/data/`.

```powershell
python scripts/inspect_data.py
python scripts/convert_data.py
```

For real DAPT, provide a domain text corpus outside Git:

```powershell
python scripts/prepare_dapt_corpus.py --corpus-dir data/dapt_corpus --out outputs/data/dapt_train.jsonl
python scripts/check_dapt_corpus.py --path outputs/data/dapt_train.jsonl
```

For a local smoke-only run, a small derived DAPT corpus can be created from SFT text:

```powershell
python scripts/prepare_dapt_corpus.py --allow-smoke-from-sft --out outputs/data/dapt_train.jsonl
```

## Third-Party Code

External projects are downloaded locally and are not vendored into this repository:

```powershell
python scripts/download_third_party.py
```

Model weights must be downloaded through Hugging Face, ModelScope, or the relevant Python libraries. Do not clone model weights into this Git repository.

## Smoke Checks

```powershell
python scripts/smoke_local.py
python scripts/smoke_qwen35_metadata.py
python scripts/smoke_ms_swift.py
```

Optional image-inference smoke test:

```powershell
python scripts/infer_transformers.py --model Qwen/Qwen3.5-4B --input outputs/data/eval_smoke.jsonl --output outputs/smoke/qwen35_image_predictions.jsonl --limit 1 --max-new-tokens 8 --device cuda --torch-dtype float16
```

If `Qwen/Qwen3.5-4B` cannot process image input in your environment, record the minimal reproduction and do not silently replace it with another model.

## Running E0-E3

Generate ms-swift commands without launching training:

```powershell
python scripts/run_pipeline.py --config configs/e1_qwen35_4b_lora_dpo.json --swift-wrapper scripts/run_swift_cli.py
python scripts/run_pipeline.py --config configs/e2_qwen35_4b_lora_simpo.json --swift-wrapper scripts/run_swift_cli.py
python scripts/run_pipeline.py --config configs/e3_qwen35_4b_simpo_frac_vcd.json --stages infer
```

Execute the configured stages after data conversion and DAPT-corpus preparation:

```powershell
python scripts/run_pipeline.py --config configs/e1_qwen35_4b_lora_dpo.json --swift-wrapper scripts/run_swift_cli.py --execute
python scripts/run_pipeline.py --config configs/e2_qwen35_4b_lora_simpo.json --swift-wrapper scripts/run_swift_cli.py --execute
python scripts/run_pipeline.py --config configs/e3_qwen35_4b_simpo_frac_vcd.json --stages infer --execute
```

Or use the staged PowerShell entrypoint:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_formal_pipeline.ps1 -Experiment all -Execute
```

## Evaluation

```powershell
python scripts/evaluate_predictions.py --predictions outputs/predictions/e1_predictions.jsonl --out outputs/e1/metrics.json
python scripts/evaluate_predictions.py --predictions outputs/predictions/e2_predictions.jsonl --out outputs/e2/metrics.json
python scripts/evaluate_predictions.py --predictions outputs/predictions/e3_frac_vcd_predictions.jsonl --out outputs/e3/metrics.json
python scripts/summarize_results.py
```

Metrics include Count-Precision, Count-Recall, Count-F1, LocAcc, MorphAcc, BLEU, ROUGE-L, peak VRAM, training time, inference latency, zero-fracture hallucination rate, over-count rate, and under-count rate when those values are available.

## Validation

```powershell
python -m compileall scripts src tests
$env:PYTHONPATH = (Resolve-Path src).Path
python -m unittest discover -s tests
```

## References

- [ms-swift](https://github.com/modelscope/ms-swift)
- [SimPO](https://github.com/princeton-nlp/SimPO)
- [VCD](https://github.com/DAMO-NLP-SG/VCD)
- [Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
