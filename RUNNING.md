# Running Grokking-Geometric

This guide is the recommended path for a fresh Windows clone. It keeps
checkpoints storage-safe, generates per-run plots, writes aggregate result
tables/figures, and includes the current ablation controls.

For a fresh Windows machine, the one-command launcher is:

```powershell
.\run_fresh_windows.ps1 -Mode pilot
```

Use `-Mode quick` for a smoke test, `-Mode full` for the default 240-run sweep,
or `-Mode ablation` for the expanded weight-decay/train-fraction grid. The
launcher creates `.venv`, installs dependencies, generates datasets, enables
intervention metrics, and then calls `run_sweeps.ps1`.

## 1. Clone And Enter The Repo

```powershell
git clone <repo-url>
cd grokking-geometric
```

## 2. Create A Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, keep the venv and run commands with:

```powershell
.\.venv\Scripts\python.exe <script-or-command>
```

## 3. Install Dependencies

Install PyTorch first. Use the CUDA wheel if the machine has an NVIDIA GPU:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

CPU-only works, but the full sweep will be very slow:

```powershell
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## 4. Generate Datasets

```powershell
python data/modular_addition/prepare.py
python data/modular_subtraction/prepare.py
python data/modular_multiplication/prepare.py
python data/symmetric_group/prepare.py
python data/permutation_composition/prepare.py
```

The generated `.bin` files are ignored by git.

## 5. Run A Quick Smoke Test

Run this before the full sweep:

```powershell
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_sweeps.ps1 -Quick -RunRoot runs_quick_test
```

Expected output directory:

```text
runs_quick_test/
```

The run should produce `signals.csv` and `signals.png`, then delete temporary
checkpoint files.

`signals.csv` includes loss, Hessian/OT geometry, CKA, activation norms,
SVCCA, parameter norms, gradient norms, entropy, probability margins, logit
margins, and true-class logit margins. If interventions are enabled, it also
includes layerwise intervention loss/accuracy deltas.

## 6. Run A Small Pilot Sweep

Run this before spending time on a full sweep:

```powershell
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_sweeps.ps1 -Pilot -RunRoot runs_pilot_baselines
```

The built-in pilot runs 8 short modular-addition configurations:

```text
widths: 64, 128
weight_decay: 0.0, 1.0
train_fraction: 0.25, 1.0
seed: 0
```

It uses cheap analysis settings so it validates the full train/analyze/plot/
aggregate pipeline quickly. Aggregate outputs are written to:

```text
results/runs_pilot_baselines/
```

To include intervention metrics when calling `run_sweeps.ps1` directly, add:

```powershell
-ExtraOverrides "analysis.interventions.enabled=true"
```

## 7. Run The Full Storage-Safe Sweep

```powershell
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_sweeps.ps1
```

The script creates a timestamped run root:

```text
runs_fresh_<YYYYMMDD_HHMMSS>/
```

When the sweep finishes, aggregate outputs are written to:

```text
results/runs_fresh_<YYYYMMDD_HHMMSS>/
```

The default full sweep is still 240 runs. It uses:

```text
weight_decay: 1.0
train_fraction: 1.0
```

To run ablations, pass comma-separated grids:

```powershell
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_sweeps.ps1 -RunRoot runs_wd_tf_ablation -WeightDecays "0.0,0.1,1.0" -TrainFractions "0.25,0.5,1.0"
```

That multiplies the run count by `len(WeightDecays) * len(TrainFractions)`, so
use a dedicated run root and expect much longer runtime.

The fresh-clone launcher exposes this as:

```powershell
.\run_fresh_windows.ps1 -Mode ablation -WeightDecays "0.0,0.1,1.0" -TrainFractions "0.25,0.5,1.0"
```

## 8. Resume An Interrupted Sweep

Use the same run root and pass `-Resume`:

```powershell
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_sweeps.ps1 -RunRoot runs_fresh_YYYYMMDD_HHMMSS -Resume
```

Resume skips any configuration that already has `signals.csv`. If a run folder
exists without `signals.csv`, stale temporary checkpoints are removed and that
configuration restarts cleanly.

## 9. Re-Aggregate Results Manually

```powershell
python summarize_runs.py --runs-dir runs_fresh_YYYYMMDD_HHMMSS --output-dir results\runs_fresh_YYYYMMDD_HHMMSS
```

This writes:

```text
summary.csv
aggregate_summary.png
```

## 10. Important Output Columns

Each finished run writes `signals.csv` in its run folder. Useful columns:

```text
train_loss, val_loss
weight_decay, train_fraction
trace, trace_normalized, lambda_max
sinkhorn_mean, sinkhorn_L*_to_L*
cka_mean, cka_L*_to_L*
svcca_mean, svcca_L*_to_L*, svcca_top_L*_to_L*
activation_rms_L*, activation_std_L*
param_l2, param_rms
grad_l2, grad_rms
train_entropy, val_entropy
train_prob_margin, val_prob_margin
train_logit_margin, val_logit_margin
train_true_logit_margin, val_true_logit_margin
intervention_clean_loss, intervention_clean_accuracy
intervene_<mode>_<layer>_loss_delta
intervene_<mode>_<layer>_accuracy_delta
```

Run folder names include the ablation settings:

```text
<dataset>_L<layers>_d<width>_wd<weight_decay>_tf<train_fraction>_seed<seed>
```

For example:

```text
modular_addition_L1_d128_wd1_tf0p25_seed0
```

Intervention modes currently supported:

```text
mean_ablate
shuffle
noise
zero
```

The default intervention override uses `mean_ablate`, `shuffle`, and `noise`.

## 11. Storage Policy

The default storage policy is designed to stay below 10 GB:

- checkpoint every 200 steps;
- model weights only, no optimizer state;
- no RNG state;
- analyze every saved checkpoint;
- delete checkpoints after `signals.csv` and `signals.png` are produced;
- cap each run root at `storage.max_run_gb=9.5`.

Expected storage:

```text
Peak during largest run: about 4.5-5 GB
Final full sweep output: usually under 200 MB
```

Do not use this unless debugging:

```powershell
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_sweeps.ps1 -KeepCheckpoints
```

Keeping checkpoints can push storage into the hundreds of GB.

## 12. What To Commit

Commit source code, configs, and documentation:

```text
README.md
RUNNING.md
requirements.txt
conf/
run.py
run_sweeps.ps1
run_fresh_windows.ps1
summarize_runs.py
analysis and plotting scripts
geometry_utils/
```

Do not commit generated artifacts:

```text
data/*.bin
runs/
runs_fresh_*/
runs_*/
outputs/
logs/
results/
plots/*.png
plots/*.csv
*.pt
__pycache__/
```

These are ignored by `.gitignore`.

## 13. CPU Notes

CPU execution is supported because `run.py` falls back to CPU when CUDA is not
available. For CPU, use a smaller run:

```powershell
python run.py device=cpu dataset=modular_addition model.n_layer=1 model.n_embd=64 model.n_head=2 max_iters=2000 eval_interval=400 save_every=400 weight_decay=1.0 train_fraction=1.0 analysis.hutchinson_samples=1 analysis.power_iters=5 analysis.num_examples=64 analysis.sinkhorn_iters=10
```

Or use the fresh-clone launcher:

```powershell
.\run_fresh_windows.ps1 -Mode quick -Cpu
.\run_fresh_windows.ps1 -Mode pilot -Cpu
```

Do not run the full 240-run sweep on CPU unless you are prepared for a very
long runtime.

## 14. Progress And Logs

The sweep script prints progress for each configuration:

```text
[17/240] RUN modular_addition_L2_d64_wd1_tf1_seed3
step   200: train loss ..., val loss ...
Analyzing: 40%|...
Completed in 8.3 min
```

To save terminal output while still seeing it:

```powershell
New-Item -ItemType Directory -Force logs
.\run_fresh_windows.ps1 -Mode full *>&1 | Tee-Object -FilePath logs\full_sweep.log
```

To check completed runs from another PowerShell window:

```powershell
$done = (Get-ChildItem runs_full_* -Recurse -Filter signals.csv).Count
$total = 240
"{0}/{1} complete ({2:n1}%)" -f $done, $total, (100*$done/$total)
```

For the default ablation grid, use `$total = 2160`:

```powershell
$done = (Get-ChildItem runs_ablation_* -Recurse -Filter signals.csv).Count
$total = 2160
"{0}/{1} complete ({2:n1}%)" -f $done, $total, (100*$done/$total)
```
