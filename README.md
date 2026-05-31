# Grokking-Geometric — Reproducibility Notes

Quick instructions to reproduce the grokking baselines used in this repo.

Prerequisites
- Python 3.10+ and the versions in `requirements.txt` (recommended: `torch>=2.1.0`).
- CUDA + appropriate PyTorch build if you want GPU/bfloat16 runs.

Recommended reproducible runs
- Full-batch reproducible run (disable fused AdamW, deterministic ops):

```bash
python train.py --full-batch --disable-fused --deterministic --seed 1337 \
  --max-iters 1000 --eval-iters 10 --eval-interval 100
```

- Quick smoke run (already used for CI/local checks):

```bash
python train.py --disable-fused --max-iters 20 --eval-iters 2 --eval-interval 5
```

What is logged/saved
- `out-grokking/run_metadata.json`: environment and flags used for the run.
- Checkpoints: `out-grokking/ckpt_{iter}.pt` — each checkpoint includes `model`, `optimizer`, `model_args`, `iter_num`, `best_val_loss`, `rng_states`, and `run_metadata`.

Reproducibility tips
- Use `--disable-fused` to avoid fused optimizer numerical differences across PyTorch builds.
- Use `--deterministic` to force deterministic algorithms (may slow training or be unsupported on some builds).
- Match hardware precision to the baseline (float32 / bfloat16) — set `dtype` in `train.py` if necessary.
- To resume exactly, load RNG states from checkpoint and restore them before continuing.

Data
- `data/modular_addition/prepare.py` generates the dataset (p=97). Note that train is intentionally 25% of all equations to produce the small-training regime used in grokking experiments.

Contact
- If you want automated scripts for launching sweeps or to export an environment YAML, tell me and I will add them.
