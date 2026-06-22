param(
  [ValidateSet('e0', 'e1', 'e2', 'e3', 'all')]
  [string]$Experiment = 'all',
  [string]$Stages = '',
  [switch]$Execute,
  [switch]$StrictDapt,
  [string]$Python = '.\.venv310\Scripts\python.exe',
  [string]$SwiftBin = '.\.venv310\Scripts\swift.exe',
  [string]$SwiftWrapper = 'scripts\run_swift_cli.py'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $Python)) {
  $Python = 'python'
}

$commonArgs = @()
if (Test-Path $SwiftBin) {
  $commonArgs += @('--swift-bin', $SwiftBin)
}
if (Test-Path $SwiftWrapper) {
  $commonArgs += @('--swift-wrapper', $SwiftWrapper)
}
if ($Execute) {
  $commonArgs += '--execute'
}

& $Python scripts\inspect_data.py
& $Python scripts\convert_data.py

if ($Execute -and -not (Test-Path 'outputs\data\dapt_train.jsonl')) {
  throw 'outputs\data\dapt_train.jsonl is missing. Build it with scripts\prepare_dapt_corpus.py before execution.'
}
if ($Execute) {
  $daptCheckArgs = @('scripts\check_dapt_corpus.py', '--path', 'outputs\data\dapt_train.jsonl')
  if (-not $StrictDapt) {
    $daptCheckArgs += '--allow-derived'
  }
  & $Python @daptCheckArgs
  if ($LASTEXITCODE -ne 0) {
    throw 'Formal execution stopped because outputs\data\dapt_train.jsonl is not accepted by the selected DAPT corpus policy.'
  }
}

function Invoke-FracLogPipeline {
  param([string]$ConfigPath, [string]$Stages = '')
  $pipelineArgs = @('scripts\run_pipeline.py', '--config', $ConfigPath, '--resume-existing-checkpoints') + $commonArgs
  if ($Stages) {
    $pipelineArgs += @('--stages', $Stages)
  }
  & $Python @pipelineArgs
}

if ($Experiment -eq 'e0' -or $Experiment -eq 'all') {
  Invoke-FracLogPipeline 'configs\e0_qwen25_vl7b_lora_dpo.json' $Stages
}
if ($Experiment -eq 'e1' -or $Experiment -eq 'all') {
  Invoke-FracLogPipeline 'configs\e1_qwen35_4b_lora_dpo.json' $Stages
}
if ($Experiment -eq 'e2' -or $Experiment -eq 'all') {
  Invoke-FracLogPipeline 'configs\e2_qwen35_4b_lora_simpo.json' $Stages
}
if ($Experiment -eq 'e3' -or $Experiment -eq 'all') {
  Invoke-FracLogPipeline 'configs\e3_qwen35_4b_simpo_frac_vcd.json' 'infer'
}

if ($Execute) {
  $shouldEvaluate = (-not $Stages) -or ($Stages -match '(^|,)infer(,|$)')
}

if ($Execute -and $shouldEvaluate) {
  $evals = @(
    @('outputs\predictions\e0_predictions.jsonl', 'outputs\e0\metrics.json'),
    @('outputs\predictions\e1_predictions.jsonl', 'outputs\e1\metrics.json'),
    @('outputs\predictions\e2_predictions.jsonl', 'outputs\e2\metrics.json'),
    @('outputs\predictions\e3_frac_vcd_predictions.jsonl', 'outputs\e3\metrics.json')
  )
  foreach ($pair in $evals) {
    if (Test-Path $pair[0]) {
      & $Python scripts\evaluate_predictions.py --predictions $pair[0] --out $pair[1]
    }
  }
  & $Python scripts\summarize_results.py
}
if ($Execute -and -not $shouldEvaluate) {
  Write-Host "Skipping prediction evaluation because selected stages do not include infer."
}
