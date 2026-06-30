<#
.SYNOPSIS
  Phase 1 sweep: Cross-architecture and cross-width universality.

.GRID
  - Tasks:  modular_addition, modular_multiplication, symmetric_group
  - Depths: 1, 2, 4, 6 layers
  - Widths: 64, 128, 256, 512
  - Seeds:  0..4 (5 seeds)
  => 3 x 4 x 4 x 5 = 240 runs

.USAGE
  .\run_sweeps.ps1                         # full sweep (240 runs, ~days)
  .\run_sweeps.ps1 -Quick                   # 1 config only, for testing
  .\run_sweeps.ps1 -Dataset modadd          # single dataset
  .\run_sweeps.ps1 -Resume                  # resume from existing checkpoints
#>

param(
  [switch]$Quick,
  [string]$Dataset = "all",
  [switch]$Resume,
  [double]$MaxRunGB = 9.5,
  [switch]$KeepCheckpoints,
  [string]$RunRoot = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
  $RunRoot = "runs_fresh_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$storagePolicy = "run_root=$RunRoot save_every=200 eval_interval=200 analysis.stride=1 storage.save_optimizer=false storage.save_rng=false storage.max_checkpoints=60 storage.max_run_gb=$MaxRunGB storage.keep_last_checkpoints=0 wandb.mode=disabled"
if ($KeepCheckpoints) {
  $storagePolicy = "run_root=$RunRoot save_every=200 eval_interval=200 analysis.stride=1 storage.save_optimizer=false storage.save_rng=false storage.max_checkpoints=60 storage.max_run_gb=$MaxRunGB storage.keep_checkpoints_after_analysis=true storage.keep_last_checkpoints=1 wandb.mode=disabled"
}
$BASE = "python run.py $storagePolicy"

if ($Quick) {
  Write-Host "=== QUICK TEST: single config ===" -ForegroundColor Yellow
  $cmd = "$BASE dataset=modular_addition model.n_layer=1 model.n_embd=128 model.n_head=4 max_iters=500 seed=42"
  Write-Host "Running: $cmd"
  Invoke-Expression $cmd
  exit 0
}

# ---- Build sweep grid ----
$datasets = if ($Dataset -eq "all") { @("modular_addition","modular_multiplication","symmetric_group") } else { @($Dataset) }
$layers = @(1, 2, 4, 6)
$widths = @(64, 128, 256, 512)
$seeds = @(0, 1, 2, 3, 4)

$total = $datasets.Count * $layers.Count * $widths.Count * $seeds.Count
$count = 0

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  PHASE 1 SWEEP: $total configurations"    -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Datasets : $($datasets -join ', ')"
Write-Host "Layers   : $($layers -join ', ')"
Write-Host "Widths   : $($widths -join ', ')"
Write-Host "Seeds    : $($seeds -join ', ')"
Write-Host "Run root : $RunRoot"
Write-Host ""

foreach ($ds in $datasets) {
  foreach ($L in $layers) {
    foreach ($d in $widths) {
      # Compute n_head from width (maintain head_dim=32)
      $nhead = [Math]::Max(1, $d / 32)
      foreach ($seed in $seeds) {
        $count++
        $exp_name = "${ds}_L${L}_d${d}_seed${seed}"
        $out_dir = "runs/$exp_name"

        if ($Resume) {
          $signals_path = Join-Path $out_dir "signals.csv"
          if (Test-Path -LiteralPath $signals_path) {
            Write-Host "[$count/$total] SKIP $exp_name (signals.csv found)" -ForegroundColor DarkGray
            continue
          } elseif (Test-Path -LiteralPath $out_dir) {
            Get-ChildItem -LiteralPath $out_dir -Filter "ckpt_*.pt" -File -ErrorAction SilentlyContinue | Remove-Item -Force
            Get-ChildItem -LiteralPath $out_dir -Filter "signals.png" -File -ErrorAction SilentlyContinue | Remove-Item -Force
          }
        }

        Write-Host "[$count/$total] RUN $exp_name" -ForegroundColor Green
        $cmd = "$BASE dataset=$ds model.n_layer=$L model.n_embd=$d model.n_head=$nhead seed=$seed"

        if ($ds -eq "symmetric_group") {
          $cmd += " max_iters=20000 analysis.stride=1"
        } else {
          $cmd += " max_iters=10000 analysis.stride=1"
        }

        Write-Host "  $cmd"

        $start = Get-Date
        try {
          Invoke-Expression $cmd
          if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAILED (exit code $LASTEXITCODE)" -ForegroundColor Red
          }
        } catch {
          Write-Host "  EXCEPTION: $_" -ForegroundColor Red
        }
        $elapsed = (Get-Date) - $start
        Write-Host "  Completed in $($elapsed.TotalMinutes.ToString('F1')) min" -ForegroundColor Gray
        Write-Host ""
      }
    }
  }
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  SWEEP COMPLETE: $count / $total runs"      -ForegroundColor Cyan
Write-Host "  Aggregating results into results/$RunRoot" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

python summarize_runs.py --runs-dir $RunRoot --output-dir "results/$RunRoot"
