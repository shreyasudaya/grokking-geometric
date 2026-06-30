#!/usr/bin/env python3
"""
Universal Trajectory Plot: Normalize X-axis by grokking step to show
collapse curves overlap across depths, widths, seeds, and datasets.

Usage:
  python plot_universal.py --runs-dir ./runs --output universal_trajectory.png
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def find_signals_csv(runs_dir):
    """Find all signals.csv under runs_dir."""
    return glob.glob(os.path.join(runs_dir, '**', 'signals.csv'), recursive=True)


def extract_meta(path):
    """Extract experiment metadata from parent directory name."""
    parent = os.path.basename(os.path.dirname(path))
    parts = parent.split('_')
    meta = {'path': path, 'name': parent}
    for p in parts:
        if p.startswith('L') and p[1:].isdigit():
            meta['n_layer'] = int(p[1:])
        elif p.startswith('d') and p[1:].isdigit():
            meta['n_embd'] = int(p[1:])
        elif p.startswith('seed') and p[3:].isdigit():
            meta['seed'] = int(p[3:])
    # Dataset is everything before _L
    meta['dataset'] = '_'.join(parts[:parts.index([x for x in parts if x.startswith('L')][0])]) \
        if any(x.startswith('L') for x in parts) else parent
    return meta


def detect_grokking_step(df, col='val_loss', threshold=0.1):
    """First step where val_loss drops below threshold * max(val_loss)."""
    if col not in df.columns or len(df) < 5:
        return None
    vals = df[col].values
    max_val = np.max(vals)
    if max_val <= 0:
        return None
    below = np.where(vals < threshold * max_val)[0]
    return df['step'].iloc[below[0]] if len(below) else None


def plot_universal(runs_dir, output, metric='sinkhorn_mean', val_loss_thresh=0.1,
                   xmax=3.0, figsize=(14, 6)):
    csv_files = find_signals_csv(runs_dir)
    if not csv_files:
        print(f"No signals.csv found under {runs_dir}")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, col, title, ylabel in [
        (axes[0], 'sinkhorn_mean', 'Universal OT Collapse', 'Sinkhorn Distance'),
        (axes[1], 'trace', 'Universal Hessian Flattening', 'Tr(H)'),
    ]:
        for csv_path in csv_files:
            meta = extract_meta(csv_path)
            df = pd.read_csv(csv_path).sort_values('step')
            if col not in df.columns:
                continue

            gs = detect_grokking_step(df, 'val_loss', val_loss_thresh)
            if gs is None or gs == 0:
                continue

            # Normalize X-axis by grokking step
            tau = df['step'].values / gs
            y = df[col].values

            # Normalize Y-axis to [0, 1]
            y_min, y_max = np.min(y), np.max(y)
            if y_max - y_min > 1e-10:
                y_norm = (y - y_min) / (y_max - y_min)
            else:
                y_norm = np.zeros_like(y)

            # Only plot up to xmax
            mask = tau <= xmax
            if mask.sum() < 3:
                continue

            label = f"{meta['dataset']} L{meta.get('n_layer','?')} d{meta.get('n_embd','?')}"
            ax.plot(tau[mask], y_norm[mask], lw=0.5, alpha=0.6, label=label if label not in [l.get_label() for l in ax.lines] else '')

        ax.axvline(x=1.0, color='gray', ls='--', alpha=0.5, label=f'Grokking (x{val_loss_thresh})')
        ax.set_xlabel(r'Normalized Step $\tau = t / t_{\text{grok}}$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=6, loc='best', ncol=2)
        ax.set_xlim(0, xmax)

        # Normalized y axis for sinkhorn
        if col == 'sinkhorn_mean':
            ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Universal trajectory plot saved to {output}")
    print(f"  Found {len(csv_files)} runs, {len([c for c in csv_files if detect_grokking_step(pd.read_csv(c).sort_values('step'), 'val_loss', val_loss_thresh) is not None])} with detectable grokking.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Universal trajectory plot')
    parser.add_argument('--runs-dir', type=str, default='./runs', help='directory with experiment subdirs')
    parser.add_argument('--output', type=str, default='plots/universal_trajectory.png')
    parser.add_argument('--metric', type=str, default='sinkhorn_mean')
    parser.add_argument('--val-loss-thresh', type=float, default=0.1)
    parser.add_argument('--xmax', type=float, default=3.0)
    args = parser.parse_args()
    plot_universal(args.runs_dir, args.output, args.metric, args.val_loss_thresh, args.xmax)
