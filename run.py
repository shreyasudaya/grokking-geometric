#!/usr/bin/env python3
"""
Unified experiment runner: train a transformer, analyze geometry, log to W&B.

Usage:
  python run.py experiment_name=my_exp dataset=modular_addition model.n_layer=2 model.n_embd=256 seed=42
  python run.py --multirun dataset=modular_addition,modular_multiplication model.n_embd=64,128,256,512 seed=0,1,2,3,4
"""
import os
import sys
import re
import math
import json
import random
import time
import platform
from pathlib import Path
from contextlib import nullcontext
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm

from model import GPTConfig, GPT
from hessian import analyze_checkpoint as hessian_analyze
from geometry_utils.ot_solver import layerwise_ot_pipeline
from plot_signals import plot_signals

# ---------------------------------------------------------------------------
#  TRAINING
# ---------------------------------------------------------------------------

def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def _float_tag(value):
    return f"{float(value):g}".replace('-', 'm').replace('.', 'p')


def setup_dataloader(data_dir, block_size, n_output, batch_size, device, device_type,
                     train_fraction=1.0, subset_seed=42):
    eq_length = block_size + 1
    memmaps = {}
    num_eqs = {}
    for split in ['train', 'val']:
        data_path = os.path.join(data_dir, f'{split}.bin')
        memmaps[split] = np.memmap(data_path, dtype=np.uint16, mode='r')
        num_eqs[split] = len(memmaps[split]) // eq_length

    train_fraction = float(train_fraction)
    train_subset = None
    if train_fraction < 1.0:
        if train_fraction <= 0.0:
            raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")
        subset_size = max(1, int(round(num_eqs['train'] * train_fraction)))
        rng = np.random.default_rng(int(subset_seed))
        train_subset = np.sort(
            rng.choice(num_eqs['train'], size=subset_size, replace=False)
        ).astype(np.int64)
        print(f"Using {subset_size:,}/{num_eqs['train']:,} train equations "
              f"({train_fraction:g})")

    def get_batch(split):
        data = memmaps[split]
        if split == 'train' and train_subset is not None:
            choices = torch.randint(0, len(train_subset), (batch_size,)).numpy()
            ix_eq = train_subset[choices]
        else:
            ix_eq = torch.randint(0, num_eqs[split], (batch_size,)).numpy()
        ix = ix_eq * eq_length
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        y[:, :-n_output] = -1
        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
    return get_batch


def train_model(cfg: DictConfig, out_dir: str, device: str, device_type: str, ctx) -> Optional[str]:
    """Train the model. Returns path to final checkpoint directory."""
    os.makedirs(out_dir, exist_ok=True)

    ds_cfg = cfg.dataset
    model_cfg = cfg.model
    data_dir = os.path.join('data', cfg.dataset.dataset_name)

    # Model
    model_args = dict(
        n_layer=model_cfg.n_layer, n_head=model_cfg.n_head, n_embd=model_cfg.n_embd,
        block_size=ds_cfg.block_size, bias=model_cfg.bias,
        vocab_size=ds_cfg.vocab_size, dropout=model_cfg.dropout,
    )
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)

    # Optimizer
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    optimizer, _ = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type,
    )

    # Data
    get_batch = setup_dataloader(
        data_dir, ds_cfg.block_size, ds_cfg.n_output, cfg.batch_size, device, device_type,
        train_fraction=cfg.train_fraction, subset_seed=cfg.seed,
    )

    # Save metadata
    run_metadata = {
        'dataset': cfg.dataset.dataset_name,
        'seed': cfg.seed,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'device': device,
        'platform': platform.platform(),
        'weight_decay': float(cfg.weight_decay),
        'train_fraction': float(cfg.train_fraction),
        'learning_rate': float(cfg.learning_rate),
        'max_iters': int(cfg.max_iters),
    }
    with open(os.path.join(out_dir, 'run_metadata.json'), 'w') as f:
        json.dump(run_metadata, f, indent=2)

    nparams = sum(p.numel() for p in model.parameters())
    cfg_m = model_cfg
    print(f"Model: n_layer={cfg_m.n_layer}, n_embd={cfg_m.n_embd}, n_head={cfg_m.n_head}")
    print(f"  Parameters: {nparams:,}")
    wandb.log({'n_parameters': nparams}, step=0)

    # Training loop
    iter_num = 0
    t0 = time.time()

    @torch.no_grad()
    def estimate_loss(eval_iters=50):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    while iter_num <= cfg.max_iters:
        lr = get_lr(iter_num, cfg.warmup_iters, cfg.max_iters, cfg.learning_rate, cfg.min_lr) if cfg.decay_lr else cfg.learning_rate
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Eval + checkpoint
        if iter_num % cfg.eval_interval == 0:
            losses = estimate_loss()
            msg = f"step {iter_num:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            print(msg)
            wandb.log({
                'train_loss': losses['train'],
                'val_loss': losses['val'],
                'lr': lr,
            }, step=iter_num)
            _save_ckpt(model, optimizer, model_args, cfg, out_dir, iter_num, val_loss=losses['val'])
            _prune_checkpoints(out_dir, cfg)

        elif iter_num % cfg.save_every == 0:
            _save_ckpt(model, optimizer, model_args, cfg, out_dir, iter_num)
            _prune_checkpoints(out_dir, cfg)

        if iter_num >= cfg.max_iters:
            break

        # Forward-backward
        X, Y = get_batch('train')
        with ctx:
            _, loss = model(X, Y)

        scaler.scale(loss).backward()
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if iter_num % 50 == 0:
            t1 = time.time()
            wandb.log({'iter_time_ms': (t1 - t0) * 1000}, step=iter_num)
            t0 = t1

        iter_num += 1

    print(f"Training completed. Checkpoints in {out_dir}")
    return out_dir


def _save_ckpt(model, optimizer, model_args, cfg, out_dir, iter_num, val_loss=None):
    storage = cfg.get('storage', {})
    ckpt = {
        'model': model.state_dict(),
        'model_args': model_args,
        'dataset_name': cfg.dataset.dataset_name,
        'n_output': cfg.dataset.n_output,
        'iter_num': iter_num,
        'val_loss': val_loss,
        'weight_decay': float(cfg.weight_decay),
        'train_fraction': float(cfg.train_fraction),
        'run_metadata': {
            'weight_decay': float(cfg.weight_decay),
            'train_fraction': float(cfg.train_fraction),
            'learning_rate': float(cfg.learning_rate),
            'seed': int(cfg.seed),
        },
    }
    if storage.get('save_optimizer', False):
        ckpt['optimizer'] = optimizer.state_dict()
    if storage.get('save_rng', False):
        ckpt['rng_states'] = {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
    torch.save(ckpt, os.path.join(out_dir, f'ckpt_{iter_num:06d}.pt'))


def _checkpoint_files(checkpoint_dir):
    files = [f for f in os.listdir(checkpoint_dir) if re.match(r'ckpt_\d+\.pt', f)]
    files.sort(key=extract_step)
    return files


def _dir_size_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _prune_checkpoints(checkpoint_dir, cfg):
    storage = cfg.get('storage', {})
    max_checkpoints = int(storage.get('max_checkpoints', 0) or 0)
    max_run_gb = float(storage.get('max_run_gb', 0) or 0)
    keep_last = max(0, int(storage.get('keep_last_checkpoints', 1) or 0))

    ckpt_files = _checkpoint_files(checkpoint_dir)
    protected = set(ckpt_files[-keep_last:]) if keep_last else set()

    def removable():
        return [f for f in _checkpoint_files(checkpoint_dir) if f not in protected]

    if max_checkpoints > 0:
        while len(_checkpoint_files(checkpoint_dir)) > max_checkpoints:
            candidates = removable()
            if not candidates:
                break
            victim = candidates[0]
            os.remove(os.path.join(checkpoint_dir, victim))
            print(f"Pruned checkpoint {victim} (max_checkpoints={max_checkpoints})")

    if max_run_gb > 0:
        max_bytes = max_run_gb * (1024 ** 3)
        while _dir_size_bytes(checkpoint_dir) > max_bytes:
            candidates = removable()
            if not candidates:
                break
            victim = candidates[0]
            os.remove(os.path.join(checkpoint_dir, victim))
            print(f"Pruned checkpoint {victim} (max_run_gb={max_run_gb})")


def _wandb_enabled(cfg):
    return str(cfg.wandb.mode).lower() == 'online'


# ---------------------------------------------------------------------------
#  ANALYSIS
# ---------------------------------------------------------------------------

def extract_step(filename):
    match = re.search(r'ckpt_(\d+)\.pt', filename)
    return int(match.group(1)) if match else -1


def run_analysis(cfg: DictConfig, checkpoint_dir: str, device: str, exp_name: str):
    """Run Hessian + OT analysis over checkpoints. Logs each step to W&B."""
    an = cfg.analysis
    storage = cfg.get('storage', {})
    ckpt_files = _checkpoint_files(checkpoint_dir)
    ckpt_files = ckpt_files[::an.stride]
    keep_last = max(0, int(storage.get('keep_last_checkpoints', 1) or 0))
    protected = set(ckpt_files[-keep_last:]) if keep_last else set()

    df = pd.DataFrame()
    for ckpt_name in tqdm(ckpt_files, desc="Analyzing"):
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        step = extract_step(ckpt_name)

        record = {'step': step}

        try:
            h = hessian_analyze(
                ckpt_path=ckpt_path,
                dataset=cfg.dataset.dataset_name,
                device=device,
                batch_size=cfg.batch_size,
                hutchinson_samples=an.hutchinson_samples,
                power_iters=an.power_iters,
                seed=cfg.seed,
                train_fraction=cfg.train_fraction,
            )
            record['train_loss'] = h['train_loss']
            record['val_loss'] = h['val_loss']
            record['weight_decay'] = float(cfg.weight_decay)
            record['train_fraction'] = float(cfg.train_fraction)
            record['lambda_max'] = h['power_iteration']['lambda_max']
            record['trace'] = h['hutchinson']['trace']
            record['trace_normalized'] = h['hutchinson']['trace_normalized']
            record['frobenius_norm'] = h['hutchinson']['frobenius_norm']
            record['param_l2'] = h.get('param_l2')
            record['param_rms'] = h.get('param_rms')
            record['grad_l2'] = h.get('grad_l2')
            record['grad_rms'] = h.get('grad_rms')
            for prefix in ['train', 'val']:
                stats = h.get(f'{prefix}_stats', {})
                record[f'{prefix}_entropy'] = stats.get('entropy')
                record[f'{prefix}_prob_margin'] = stats.get('prob_margin')
                record[f'{prefix}_logit_margin'] = stats.get('logit_margin')
                record[f'{prefix}_true_logit_margin'] = stats.get('true_logit_margin')
        except Exception as e:
            print(f"  Hessian failed at step {step}: {e}")

        try:
            ot = layerwise_ot_pipeline(
                ckpt_path=ckpt_path,
                dataset=cfg.dataset.dataset_name,
                device=device,
                num_examples=an.num_examples,
                target_dim=an.target_dim,
                epsilon=an.epsilon,
                sinkhorn_iters=an.sinkhorn_iters,
                seed=cfg.seed,
            )
            for i, d in enumerate(ot['distances']):
                record[f'sinkhorn_L{i}_to_L{i+1}'] = d
            record['sinkhorn_mean'] = float(np.mean(ot['distances']))
            for i, v in enumerate(ot.get('baselines', {}).get('linear_cka', [])):
                record[f'cka_L{i}_to_L{i+1}'] = v
            for i, v in enumerate(ot.get('baselines', {}).get('activation_rms', [])):
                record[f'activation_rms_L{i}'] = v
            for i, v in enumerate(ot.get('baselines', {}).get('activation_std', [])):
                record[f'activation_std_L{i}'] = v
            cka_vals = ot.get('baselines', {}).get('linear_cka', [])
            if cka_vals:
                record['cka_mean'] = float(np.mean(cka_vals))
        except Exception as e:
            print(f"  OT failed at step {step}: {e}")

        log_dict = {k: v for k, v in record.items() if k != 'step'}
        wandb.log(log_dict, step=step)

        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        if not storage.get('keep_checkpoints_after_analysis', False) and ckpt_name not in protected:
            try:
                os.remove(ckpt_path)
                print(f"Deleted analyzed checkpoint {ckpt_name}")
            except OSError as e:
                print(f"  Could not delete checkpoint {ckpt_name}: {e}")

    df = df.sort_values('step').reset_index(drop=True)
    csv_path = os.path.join(checkpoint_dir, 'signals.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} records to {csv_path}")

    plot_path = os.path.join(checkpoint_dir, 'signals.png')
    try:
        maxlag = 5 if len(df) < 30 else 10
        plot_signals(csv_path, plot_path, maxlag=maxlag)
    except Exception as e:
        print(f"  Plotting failed: {e}")

    if _wandb_enabled(cfg):
        artifact = wandb.Artifact(f"signals-{exp_name}", type="dataset")
        artifact.add_file(csv_path)
        if os.path.exists(plot_path):
            artifact.add_file(plot_path)
        wandb.log_artifact(artifact)
    _prune_checkpoints(checkpoint_dir, cfg)

    return df


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    exp = (
        f"{cfg.dataset.dataset_name}_L{cfg.model.n_layer}_d{cfg.model.n_embd}"
        f"_wd{_float_tag(cfg.weight_decay)}_tf{_float_tag(cfg.train_fraction)}"
        f"_seed{cfg.seed}"
    )
    out_dir = os.path.abspath(os.path.join(cfg.run_root, exp))
    os.makedirs(out_dir, exist_ok=True)

    device = cfg.device if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)


    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    wandb_tags = list(cfg.wandb.tags) + [cfg.dataset.dataset_name, f"L{cfg.model.n_layer}", f"d{cfg.model.n_embd}"]
    wandb_entity = cfg.wandb.entity if cfg.wandb.entity and cfg.wandb.entity != 'null' else None
    wandb.init(
        project=cfg.wandb.project,
        entity=wandb_entity,
        name=exp,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=wandb_tags,
        notes=cfg.wandb.notes,
        dir=out_dir,
        mode=cfg.wandb.mode,
    )
    wandb.log({'seed': cfg.seed}, step=0)

    if cfg.train:
        print(f"\n{'='*60}\nTRAINING\n{'='*60}")
        train_model(cfg, out_dir, device, device_type, ctx)
    else:
        print(f"Training disabled. Checkpoints: {out_dir}")

    if cfg.analyze:
        print(f"\n{'='*60}\nANALYSIS\n{'='*60}")
        df = run_analysis(cfg, out_dir, device, exp)
        wandb.log({'n_checkpoints': len(df)}, step=0)

    wandb.finish()
    print(f"\nDone. Results in {out_dir}")


if __name__ == '__main__':
    main()
