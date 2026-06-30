<#
.SYNOPSIS
  Storage-safe sweeps for grokking geometry experiments.

.GRID
  Default full sweep:
    - Tasks:  modular_addition, modular_multiplication, symmetric_group
    - Depths: 1, 2, 4, 6 layers
    - Widths: 64, 128, 256, 512
    - Seeds:  0..4
    - Weight decay: 1.0 unless -WeightDecays is supplied
    - Train fraction: 1.0 unless -TrainFractions is supplied
    => 240 runs by default

.USAGE
  .\run_sweeps.ps1                         # full sweep (240 runs, ~days)
  .\run_sweeps.ps1 -Quick                   # 1 config only, for testing
  .\run_sweeps.ps1 -Pilot                   # small ablation pilot
  .\run_sweeps.ps1 -Dataset modular_addition # single dataset
  .\run_sweeps.ps1 -Resume                  # resume from existing checkpoints
  .\run_sweeps.ps1 -WeightDecays "0.0,0.1,1.0" -TrainFractions "0.25,0.5,1.0"
#>

param(
  [switch]$Quick,
  [switch]$Pilot,
  [string]$Dataset = "all",
  [switch]$Resume,
  [double]$MaxRunGB = 9.5,
  [switch]$KeepCheckpoints,
  [string]$RunRoot = "",
  [string]$WeightDecays = "1.0",
  [string]$TrainFractions = "1.0"
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

function Parse-Grid([string]$value) {
  return @($value.Split(",") | ForEach-Object { [double]::Parse($_.Trim(), [Globalization.CultureInfo]::InvariantCulture) })
}

function Format-Tag([double]$value) {
  return ($value.ToString("G", [Globalization.CultureInfo]::InvariantCulture) -replace "-", "m" -replace "\.", "p")
}

if ($Quick) {
  Write-Host "=== QUICK TEST: single config ===" -ForegroundColor Yellow
  $cmd = "$BASE dataset=modular_addition model.n_layer=1 model.n_embd=128 model.n_head=4 max_iters=500 seed=42 weight_decay=1.0 train_fraction=1.0 analysis.hutchinson_samples=1 analysis.power_iters=5 analysis.num_examples=64 analysis.sinkhorn_iters=10"
  Write-Host "Running: $cmd"
  Invoke-Expression $cmd
  exit 0
}

# ---- Build sweep grid ----
$weightDecayGrid = Parse-Grid $WeightDecays
$trainFractionGrid = Parse-Grid $TrainFractions

if ($Pilot) {
  $datasets = if ($Dataset -eq "all") { @("modular_addition") } else { @($Dataset) }
  $layers = @(1)
  $widths = @(64, 128)
  $seeds = @(0)
  if ($WeightDecays -eq "1.0") { $weightDecayGrid = @(0.0, 1.0) }
  if ($TrainFractions -eq "1.0") { $trainFractionGrid = @(0.25, 1.0) }
  $BASE = "$BASE max_iters=800 eval_interval=400 save_every=400 analysis.hutchinson_samples=1 analysis.power_iters=5 analysis.num_examples=64 analysis.sinkhorn_iters=10"
} else {
  $datasets = if ($Dataset -eq "all") { @("modular_addition","modular_multiplication","symmetric_group") } else { @($Dataset) }
  $layers = @(1, 2, 4, 6)
  $widths = @(64, 128, 256, 512)
  $seeds = @(0, 1, 2, 3, 4)
}

$total = $datasets.Count * $layers.Count * $widths.Count * $seeds.Count * $weightDecayGrid.Count * $trainFractionGrid.Count
$count = 0

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  SWEEP: $total configurations"            -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Datasets : $($datasets -join ', ')"
Write-Host "Layers   : $($layers -join ', ')"
Write-Host "Widths   : $($widths -join ', ')"
Write-Host "Seeds    : $($seeds -join ', ')"
Write-Host "WD       : $($weightDecayGrid -join ', ')"
Write-Host "Train Fr.: $($trainFractionGrid -join ', ')"
Write-Host "Run root : $RunRoot"
Write-Host ""

foreach ($ds in $datasets) {
  foreach ($L in $layers) {
    foreach ($d in $widths) {
      # Compute n_head from width (maintain head_dim=32)
      $nhead = [Math]::Max(1, $d / 32)
      foreach ($seed in $seeds) {
        foreach ($wd in $weightDecayGrid) {
          foreach ($tf in $trainFractionGrid) {
            $count++
            $wdTag = Format-Tag $wd
            $tfTag = Format-Tag $tf
            $exp_name = "${ds}_L${L}_d${d}_wd${wdTag}_tf${tfTag}_seed${seed}"
            $out_dir = Join-Path $RunRoot $exp_name

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
            $cmd = "$BASE dataset=$ds model.n_layer=$L model.n_embd=$d model.n_head=$nhead seed=$seed weight_decay=$($wd.ToString("G", [Globalization.CultureInfo]::InvariantCulture)) train_fraction=$($tf.ToString("G", [Globalization.CultureInfo]::InvariantCulture))"

            if (-not $Pilot) {
              if ($ds -eq "symmetric_group") {
                $cmd += " max_iters=20000 analysis.stride=1"
              } else {
                $cmd += " max_iters=10000 analysis.stride=1"
              }
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
  }
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  SWEEP COMPLETE: $count / $total runs"      -ForegroundColor Cyan
Write-Host "  Aggregating results into results/$RunRoot" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

python summarize_runs.py --runs-dir $RunRoot --output-dir "results/$RunRoot"
