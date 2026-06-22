# Public Repository Scope

This repository contains the reusable FracLogGPT-Lite experiment code, not the private experiment workspace.

## Tracked

- E0-E3 experiment configs.
- Data inspection and conversion scripts.
- DAPT/SFT/DPO/SimPO orchestration wrappers for `ms-swift`.
- Transformers inference and Frac-VCD inference scripts.
- Metric computation and result-summary utilities.
- Synthetic examples and public documentation.
- Tests for the public code path.

## Not Tracked

- Raw image pools under `pic/`.
- SFT annotation files under `sft/`.
- Preference annotation files under `rlhf_data/`.
- External DAPT corpora under `data/`.
- Checkpoints, adapters, predictions, metrics, logs, and generated reports under `outputs/` or `reports/`.
- Third-party source clones under `third_party/`.
- Model weights or model-cache folders.
- Credentials, tokens, personal account names, machine-specific paths, manuscript drafts, review packets, and internal acceptance materials.

The excluded folders are represented by `.gitkeep` placeholders only where that helps users reconstruct the local layout after cloning.
