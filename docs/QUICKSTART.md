# FracLogGPT-Lite 使用指南

[返回项目首页](../README.md) · [数据格式](DATA_SCHEMA.md) · [公开发布范围](REPOSITORY_SCOPE.md)

本文保留完整安装、数据准备、训练、推理、评测与发布检查命令。除特别说明外，所有命令都从仓库根目录执行；下面使用 PowerShell 语法。

## 无 GPU、无私有数据的公开代码验证

准备 Python 3.10 或以上版本后，在仓库根目录运行：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m unittest discover -s tests

python scripts/run_pipeline.py --config configs/e1_qwen35_4b_lora_dpo.json --swift-wrapper scripts/run_swift_cli.py
python scripts/run_pipeline.py --config configs/e2_qwen35_4b_lora_simpo.json --swift-wrapper scripts/run_swift_cli.py
python scripts/run_pipeline.py --config configs/e3_qwen35_4b_simpo_frac_vcd.json --stages infer
```

这条验证路径不需要安装完整训练依赖，不会下载模型或启动训练。单元测试覆盖数据处理、评测计算、checkpoint 解析和推理辅助逻辑；若未安装 `torch`，张量相关 VCD 测试会自动跳过。它不验证真实模型训练或图像解释效果。

不带 `--execute` 时，`run_pipeline.py` 只打印阶段命令，并写入被 Git 忽略的 `outputs/run_commands/` 与 `outputs/run_state/`。尚未存在的 checkpoint 保留为 `${checkpoint:...}` 或 `${latest:...}` 占位符，这是 dry-run 的预期行为。实际训练成功后，编排器才会解析并传递阶段 checkpoint。

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

格式示例见 [DATA_SCHEMA.md](DATA_SCHEMA.md) 与 [examples/](../examples/)。

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

以下检查面向已准备本地数据和依赖的实验环境。`smoke_local.py` 会调用数据检查、转换和 DAPT 语料准备，依赖本地 `sft/conversations.json`、偏好数据与图像；它不是干净克隆后的无数据演示。`smoke_qwen35_metadata.py` 会访问外部模型服务，模型推理检查可能下载权重并需要 GPU。公开代码验证请使用上方的单元测试和命令 dry-run。

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

也可以使用分阶段入口（该入口会先检查、转换本地私有数据；默认允许带来源说明的 SFT 派生 DAPT，严格论文复现时须额外传入 `-StrictDapt` 并准备真实领域语料）：

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

评测工具支持以下指标字段；有评测数据或外部测量输入时才会产生相应数值：

- Count-Precision、Count-Recall、Count-F1
- LocAcc、MorphAcc
- BLEU、ROUGE-L
- Peak VRAM、Training Time、Inference Latency
- Zero-Fracture Hallucination Rate
- Over-count Rate、Under-count Rate


这些字段的解释边界：

- Count 指标按 `0 / 1 / 2 / >=3` 四类计算宏平均，具体口径见 [`metrics.py`](../src/fraclog/metrics.py)。
- `LocAcc` 和 `MorphAcc` 是从参考文本与预测文本解析标签后得到的自动一致性指标，不等同于专家视觉定位或形态判读准确率。BLEU 使用仓库内的轻量实现；跨论文比较前需要核对分词与计算口径。
- 推理脚本记录逐样本生成耗时；`evaluate_predictions.py` 计算平均耗时。Peak VRAM 和 Training Time 由 `--peak-vram-mib`、`--training-time-s` 参数传入，不能仅凭配置声称已测得。
- [`paper_baseline.py`](../src/fraclog/paper_baseline.py) 中的 E0 数值来自引用论文，标记为 `same_test_set=False`，仅作方向参考。不能作为本仓库的本地复现结果，也不能与不同测试集的 E1–E3 数值直接比较提升幅度。
- 公开仓库没有实验预测、checkpoint 或结果表；`examples/` 仅供理解格式，不应用于报告模型效果。

对已有预测进行样本级 Bootstrap 置信区间分析，可参考 [`bootstrap_metrics.py`](../scripts/bootstrap_metrics.py)。这类分析不会补齐缺失的同测试集基线或专家评审。

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

更多说明见 [SECURITY.md](../SECURITY.md) 与 [REPOSITORY_SCOPE.md](REPOSITORY_SCOPE.md)。

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
