# FracLogGPT-Lite

![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-blue)
![Framework](https://img.shields.io/badge/Framework-ms--swift-5C6BC0)
![Models](https://img.shields.io/badge/Models-Qwen%202.5--VL%20%7C%20Qwen%203.5-00897B)
![Data](https://img.shields.io/badge/Data-not%20included-critical)
![Status](https://img.shields.io/badge/Status-reproducible%20research%20code-brightgreen)

**FracLogGPT-Lite** 是一个面向成像测井裂缝解释任务的轻量化多模态实验框架。项目围绕 `Qwen/Qwen3.5-4B` 构建 DAPT、LoRA-SFT、DPO/SimPO 偏好优化与 Frac-VCD 推理端视觉可信解码流程，并保留对 `Qwen/Qwen2.5-VL-7B-Instruct` 原始基线的复现或预测接入能力。

本仓库默认以中文说明为主，面向希望复现、扩展或审查 FracLogGPT 轻量化实验流程的研究者和工程使用者。仓库仅发布可复用代码与合成格式示例，不包含任何真实井资料、图像、标注、模型权重、checkpoint、预测文本或内部报告。

## 核心特性

- **完整实验矩阵**：覆盖 E0 原文基线、E1 轻量 4B+DPO、E2 轻量 4B+SimPO、E3 Frac-VCD 推理增强。
- **数据格式自检优先**：先检查本地 SFT、偏好数据和图像引用，再生成训练框架可读的 JSONL 文件。
- **主训练框架统一**：通过 `ms-swift` 执行 DAPT、LoRA-SFT、DPO 与 SimPO，命令由配置文件驱动。
- **推理端视觉校验**：Frac-VCD 在不改模型参数的前提下，对原图与扰动图 logits 进行对比融合。
- **公开仓库安全边界**：默认忽略数据、模型、输出、报告、第三方源码与本机缓存，降低误提交风险。
- **可测试代码路径**：提供数据处理、指标计算、checkpoint 解析、推理消息构造和 Frac-VCD 逻辑的单元测试。

## 方法概览

```text
本地私有数据
  ├─ SFT 对话标注
  ├─ DPO/SimPO chosen-rejected 偏好三元组
  └─ 成像测井图像池
        │
        ▼
数据检查与转换
        │
        ├─ DAPT 语料
        ├─ SFT train/val/test
        ├─ preference JSONL
        └─ eval JSONL
        │
        ▼
E0: Qwen2.5-VL-7B + DAPT + LoRA-SFT + DPO
E1: Qwen3.5-4B    + DAPT + LoRA-SFT + DPO
E2: Qwen3.5-4B    + DAPT + LoRA-SFT + SimPO
E3: E2 checkpoint + Frac-VCD inference
        │
        ▼
预测文本、结构化解析、指标汇总与效率统计
```

## 实验矩阵

| 实验 | 基座模型 | 训练与推理路径 | 目标 |
|---|---|---|---|
| E0 | `Qwen/Qwen2.5-VL-7B-Instruct` | DAPT + LoRA-SFT + DPO | 接入或复现原始 FracLogGPT 基线 |
| E1 | `Qwen/Qwen3.5-4B` | DAPT + LoRA-SFT + DPO | 验证 4B 模型替代 7B 基线的可行性 |
| E2 | `Qwen/Qwen3.5-4B` | DAPT + LoRA-SFT + SimPO | 比较 SimPO 与 DPO 在专家偏好数据上的表现 |
| E3 | `Qwen/Qwen3.5-4B` | E2 checkpoint + Frac-VCD | 在推理阶段降低过度计数、零裂缝幻觉与形态偏离 |

## 仓库结构

```text
configs/       E0-E3 与 smoke test 配置文件
docs/          数据格式、公开发布范围和安全边界说明
examples/      合成格式样例，不是真实训练数据
scripts/       数据转换、训练调度、推理、评测和发布工具
src/fraclog/   通用 Python 包：数据、指标、推理、checkpoint、VCD
tests/         公共代码路径的单元测试

pic/           本地图像池占位目录，默认不纳入 Git
sft/           本地 SFT 标注占位目录，默认不纳入 Git
rlhf_data/     本地偏好标注占位目录，默认不纳入 Git
outputs/       checkpoint、预测、指标和日志目录，默认不纳入 Git
reports/       结果表和报告目录，默认不纳入 Git
third_party/   外部 GitHub 项目下载目录，默认不纳入 Git
```

## 安装环境

```powershell
py -3.10 -m venv .venv310
.\.venv310\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

GPU 训练或推理前，请按本机 CUDA 驱动安装对应版本的 PyTorch。模型权重应通过 Hugging Face、ModelScope 或相关 Python 库下载，不应使用 `git clone` 下载到本仓库。

## 数据放置

本仓库不提供真实数据。请在本地按如下布局放置私有数据，并确保这些文件不进入 Git：

```text
sft/conversations.json
sft/sft.xlsx
rlhf_data/pref_triplet.jsonl
rlhf_data/pref_query.jsonl
pic/*.png
```

格式示例见 [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) 与 [examples/](examples/)。

执行数据检查与转换：

```powershell
python scripts/inspect_data.py
python scripts/convert_data.py
```

构建正式 DAPT 语料：

```powershell
python scripts/prepare_dapt_corpus.py --corpus-dir data/dapt_corpus --out outputs/data/dapt_train.jsonl
python scripts/check_dapt_corpus.py --path outputs/data/dapt_train.jsonl
```

仅用于本地 smoke test 时，可从 SFT 文本派生一个小型 DAPT 语料：

```powershell
python scripts/prepare_dapt_corpus.py --allow-smoke-from-sft --out outputs/data/dapt_train.jsonl
```

## 下载外部项目

外部项目不 vendoring 到 Git。需要时执行：

```powershell
python scripts/download_third_party.py
```

默认下载并记录以下项目版本：

- `modelscope/ms-swift`
- `princeton-nlp/SimPO`
- `DAMO-NLP-SG/VCD`
- `huggingface/transformers`
- `huggingface/peft`
- `vllm-project/vllm`

## Smoke Test

基础检查：

```powershell
python scripts/smoke_local.py
python scripts/smoke_qwen35_metadata.py
python scripts/smoke_ms_swift.py
```

可选图像推理检查：

```powershell
python scripts/infer_transformers.py --model Qwen/Qwen3.5-4B --input outputs/data/eval_smoke.jsonl --output outputs/smoke/qwen35_image_predictions.jsonl --limit 1 --max-new-tokens 8 --device cuda --torch-dtype float16
```

如果 `Qwen/Qwen3.5-4B` 在当前环境无法正常处理图像输入，请记录最小复现和错误日志，不要静默替换为其他模型。

## 运行实验

生成 `ms-swift` 命令但不启动训练：

```powershell
python scripts/run_pipeline.py --config configs/e1_qwen35_4b_lora_dpo.json --swift-wrapper scripts/run_swift_cli.py
python scripts/run_pipeline.py --config configs/e2_qwen35_4b_lora_simpo.json --swift-wrapper scripts/run_swift_cli.py
python scripts/run_pipeline.py --config configs/e3_qwen35_4b_simpo_frac_vcd.json --stages infer
```

执行配置中的阶段：

```powershell
python scripts/run_pipeline.py --config configs/e1_qwen35_4b_lora_dpo.json --swift-wrapper scripts/run_swift_cli.py --execute
python scripts/run_pipeline.py --config configs/e2_qwen35_4b_lora_simpo.json --swift-wrapper scripts/run_swift_cli.py --execute
python scripts/run_pipeline.py --config configs/e3_qwen35_4b_simpo_frac_vcd.json --stages infer --execute
```

也可以使用分阶段入口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_formal_pipeline.ps1 -Experiment all -Execute
```

## 评测指标

```powershell
python scripts/evaluate_predictions.py --predictions outputs/predictions/e1_predictions.jsonl --out outputs/e1/metrics.json
python scripts/evaluate_predictions.py --predictions outputs/predictions/e2_predictions.jsonl --out outputs/e2/metrics.json
python scripts/evaluate_predictions.py --predictions outputs/predictions/e3_frac_vcd_predictions.jsonl --out outputs/e3/metrics.json
python scripts/summarize_results.py
```

默认指标包括：

- Count-Precision、Count-Recall、Count-F1
- LocAcc、MorphAcc
- BLEU、ROUGE-L
- Peak VRAM、Training Time、Inference Latency
- Zero-Fracture Hallucination Rate
- Over-count Rate、Under-count Rate

## 公开发布与隐私边界

公开仓库仅保留复现实验框架。以下内容不得提交：

- 原始成像测井图像、井号、区块、井段或任何可识别生产资料的信息。
- SFT、DPO、SimPO 标注原文和专家评审记录。
- checkpoint、LoRA adapter、模型缓存、预测文本和完整结果报告。
- 本机绝对路径、账号名、token、API key、代理配置和内部工作记录。

发布前建议执行：

```powershell
rg -n -i "token|secret|password|api[_-]?key|ghp_|gho_|github_pat_|hf_|sk-|C:\\\\Users|D:\\\\program" .
git status -sb
git ls-files
```

更多说明见 [SECURITY.md](SECURITY.md) 与 [docs/REPOSITORY_SCOPE.md](docs/REPOSITORY_SCOPE.md)。

## 测试

```powershell
python -m compileall scripts src tests
$env:PYTHONPATH = (Resolve-Path src).Path
python -m unittest discover -s tests
```

如果当前环境未安装 `torch`，Frac-VCD 相关单元测试会跳过；安装完整训练/推理依赖后会自动执行。

## 参考项目

- [ms-swift](https://github.com/modelscope/ms-swift)
- [SimPO](https://github.com/princeton-nlp/SimPO)
- [VCD](https://github.com/DAMO-NLP-SG/VCD)
- [Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [vLLM](https://github.com/vllm-project/vllm)

## English Abstract

FracLogGPT-Lite is a reproducible research-code repository for lightweight multimodal fracture interpretation from imaging-log data. It provides E0-E3 experiment orchestration, data conversion, ms-swift training wrappers, Transformers inference, Frac-VCD decoding, and evaluation utilities. Real images, annotations, model weights, checkpoints, predictions, logs, reports, and private research materials are intentionally excluded.
