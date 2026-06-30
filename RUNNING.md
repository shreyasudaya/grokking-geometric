# Running Grokking-Geometric

This guide is the recommended path for a fresh Windows clone. It keeps
checkpoints storage-safe, generates per-run plots, and writes aggregate result
tables/figures.

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

## 6. Run The Full Storage-Safe Sweep

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

## 7. Resume An Interrupted Sweep

Use the same run root and pass `-Resume`:

```powershell
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_sweeps.ps1 -RunRoot runs_fresh_YYYYMMDD_HHMMSS -Resume
```

Resume skips any configuration that already has `signals.csv`. If a run folder
exists without `signals.csv`, stale temporary checkpoints are removed and that
configuration restarts cleanly.

## 8. Re-Aggregate Results Manually

```powershell
python summarize_runs.py --runs-dir runs_fresh_YYYYMMDD_HHMMSS --output-dir results\runs_fresh_YYYYMMDD_HHMMSS
```

This writes:

```text
summary.csv
aggregate_summary.png
```

## 9. Storage Policy

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

## 10. What To Commit

Commit source code, configs, and documentation:

```text
README.md
RUNNING.md
requirements.txt
conf/
run.py
run_sweeps.ps1
summarize_runs.py
analysis and plotting scripts
geometry_utils/
```

Do not commit generated artifacts:

```text
data/*.bin
runs/
runs_fresh_*/
outputs/
logs/
results/
plots/*.png
plots/*.csv
*.pt
__pycache__/
```

These are ignored by `.gitignore`.

## 11. CPU Notes

CPU execution is supported because `run.py` falls back to CPU when CUDA is not
available. For CPU, use a smaller run:

```powershell
python run.py device=cpu dataset=modular_addition model.n_layer=1 model.n_embd=64 model.n_head=2 max_iters=2000 eval_interval=400 save_every=400 analysis.hutchinson_samples=1 analysis.power_iters=5 analysis.num_examples=64 analysis.sinkhorn_iters=10
```

Do not run the full 240-run sweep on CPU unless you are prepared for a very
long runtime.
