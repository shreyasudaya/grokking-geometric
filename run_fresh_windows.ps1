<#
.SYNOPSIS
  Fresh-clone Windows launcher for storage-safe grokking-geometry runs.

.USAGE
  .\run_fresh_windows.ps1 -Mode quick
  .\run_fresh_windows.ps1 -Mode pilot
  .\run_fresh_windows.ps1 -Mode full
  .\run_fresh_windows.ps1 -Mode ablation

  Add -Cpu for CPU-only PyTorch. Add -SkipInstall if the environment is ready.
#>

param(
  [ValidateSet("quick", "pilot", "full", "ablation")]
  [string]$Mode = "pilot",
  [switch]$Cpu,
  [switch]$SkipInstall,
  [string]$RunRoot = "",
  [string]$CudaIndexUrl = "https://download.pytorch.org/whl/cu128",
  [string]$WeightDecays = "0.0,0.1,1.0",
  [string]$TrainFractions = "0.25,0.5,1.0"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath ".\run.py")) {
  throw "Run this script from the repository root, next to run.py."
}

$venvPython = Join-Path ".venv" "Scripts\python.exe"
$repoRoot = (Resolve-Path ".").Path
$venvScripts = Join-Path $repoRoot ".venv\Scripts"

if (-not (Test-Path -LiteralPath $venvPython)) {
  Write-Host "Creating .venv..." -ForegroundColor Cyan
  python -m venv .venv
}

$env:PATH = "$venvScripts;$env:PATH"

if (-not $SkipInstall) {
  Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
  & $venvPython -m pip install --upgrade pip
  if ($Cpu) {
    & $venvPython -m pip install torch torchvision torchaudio
  } else {
    & $venvPython -m pip install torch torchvision torchaudio --index-url $CudaIndexUrl
  }
  & $venvPython -m pip install -r requirements.txt
}

Write-Host "Preparing datasets..." -ForegroundColor Cyan
& $venvPython data/modular_addition/prepare.py
& $venvPython data/modular_subtraction/prepare.py
& $venvPython data/modular_multiplication/prepare.py
& $venvPython data/symmetric_group/prepare.py
& $venvPython data/permutation_composition/prepare.py

if ([string]::IsNullOrWhiteSpace($RunRoot)) {
  $RunRoot = "runs_${Mode}_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$extra = "analysis.interventions.enabled=true"
if ($Cpu) {
  $extra = "$extra device=cpu"
}

Write-Host "Running mode=$Mode into $RunRoot" -ForegroundColor Cyan
if ($Mode -eq "quick") {
  .\run_sweeps.ps1 -Quick -RunRoot $RunRoot -ExtraOverrides $extra
} elseif ($Mode -eq "pilot") {
  .\run_sweeps.ps1 -Pilot -RunRoot $RunRoot -ExtraOverrides $extra
} elseif ($Mode -eq "full") {
  .\run_sweeps.ps1 -RunRoot $RunRoot -ExtraOverrides $extra
} elseif ($Mode -eq "ablation") {
  .\run_sweeps.ps1 -RunRoot $RunRoot -WeightDecays $WeightDecays -TrainFractions $TrainFractions -ExtraOverrides $extra
}

Write-Host "Done." -ForegroundColor Green
Write-Host "Run root: $RunRoot"
Write-Host "Aggregates: results\$RunRoot"
