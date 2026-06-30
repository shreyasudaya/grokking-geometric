#!/usr/bin/env python3
"""Aggregate finished grokking-geometry runs.

Finds every runs/*/signals.csv, computes run-level metrics, writes a summary
CSV, and creates compact publication-oriented overview plots.
"""
import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_run_name(name):
    match = re.match(r'(.+)_L(\d+)_d(\d+)(?:_wd([^_]+))?(?:_tf([^_]+))?_seed(\d+)$', name)
    if not match:
        return {
            'dataset': name,
            'n_layer': np.nan,
            'n_embd': np.nan,
            'weight_decay': np.nan,
            'train_fraction': np.nan,
            'seed': np.nan,
        }
    def parse_tag(value):
        if value is None:
            return np.nan
        return float(value.replace('m', '-').replace('p', '.'))
    return {
        'dataset': match.group(1),
        'n_layer': int(match.group(2)),
        'n_embd': int(match.group(3)),
        'weight_decay': parse_tag(match.group(4)),
        'train_fraction': parse_tag(match.group(5)),
        'seed': int(match.group(6)),
    }


def first_below_step(df, col, frac=0.1):
    if col not in df or df[col].dropna().empty:
        return np.nan
    vals = df[col].astype(float).to_numpy()
    max_val = np.nanmax(vals)
    if not np.isfinite(max_val) or max_val <= 0:
        return np.nan
    hits = np.where(vals < frac * max_val)[0]
    return float(df['step'].iloc[hits[0]]) if len(hits) else np.nan


def collapse_stats(df, cols):
    cols = [c for c in cols if c in df]
    if not cols:
        return np.nan, np.nan, np.nan, np.nan
    series = df[cols].astype(float).mean(axis=1)
    if series.dropna().empty:
        return np.nan, np.nan, np.nan, np.nan
    peak_idx = int(series.idxmax())
    peak = float(series.loc[peak_idx])
    final = float(series.iloc[-1])
    collapse = (peak - final) / peak if peak > 0 else np.nan
    peak_step = float(df.loc[peak_idx, 'step'])
    return peak, final, collapse, peak_step


def summarize_run(csv_path):
    run_dir = Path(csv_path).parent
    df = pd.read_csv(csv_path).sort_values('step').reset_index(drop=True)
    meta = parse_run_name(run_dir.name)

    sink_cols = [c for c in df.columns if c.startswith('sinkhorn_L')]
    cka_cols = [c for c in df.columns if c.startswith('cka_L')]
    svcca_cols = [c for c in df.columns if c.startswith('svcca_L')]
    act_cols = [c for c in df.columns if c.startswith('activation_rms_L')]

    sink_peak, sink_final, sink_collapse, sink_peak_step = collapse_stats(df, sink_cols)
    cka_peak, cka_final, cka_collapse, cka_peak_step = collapse_stats(df, cka_cols)
    svcca_peak, svcca_final, svcca_collapse, svcca_peak_step = collapse_stats(df, svcca_cols)
    act_peak, act_final, act_collapse, act_peak_step = collapse_stats(df, act_cols)

    out = {
        'run': run_dir.name,
        'path': str(run_dir),
        **meta,
        'n_points': len(df),
        'step_final': float(df['step'].iloc[-1]) if len(df) else np.nan,
        'train_loss_final': float(df['train_loss'].iloc[-1]) if 'train_loss' in df else np.nan,
        'val_loss_final': float(df['val_loss'].iloc[-1]) if 'val_loss' in df else np.nan,
        'grokking_step_10pct': first_below_step(df, 'val_loss', frac=0.1),
        'sinkhorn_peak': sink_peak,
        'sinkhorn_final': sink_final,
        'sinkhorn_collapse_frac': sink_collapse,
        'sinkhorn_peak_step': sink_peak_step,
        'cka_peak': cka_peak,
        'cka_final': cka_final,
        'cka_change_frac': cka_collapse,
        'cka_peak_step': cka_peak_step,
        'svcca_peak': svcca_peak,
        'svcca_final': svcca_final,
        'svcca_change_frac': svcca_collapse,
        'svcca_peak_step': svcca_peak_step,
        'activation_rms_peak': act_peak,
        'activation_rms_final': act_final,
        'activation_rms_change_frac': act_collapse,
        'activation_rms_peak_step': act_peak_step,
    }

    metric_cols = [
        'trace', 'trace_normalized', 'lambda_max', 'param_l2', 'param_rms',
        'grad_l2', 'grad_rms',
        'train_entropy', 'val_entropy',
        'train_prob_margin', 'val_prob_margin',
        'train_logit_margin', 'val_logit_margin',
        'train_true_logit_margin', 'val_true_logit_margin',
        'svcca_mean',
        'intervention_clean_loss', 'intervention_clean_accuracy',
    ]
    metric_cols.extend(c for c in df.columns if c.startswith('intervene_'))

    for col in metric_cols:
        if col in df:
            vals = df[col].astype(float)
            out[f'{col}_peak'] = float(vals.max())
            out[f'{col}_final'] = float(vals.iloc[-1])
    return out


def write_plots(summary, output_dir):
    output_dir = Path(output_dir)
    if summary.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for dataset, group in summary.groupby('dataset'):
        ax.scatter(
            group['grokking_step_10pct'],
            group['sinkhorn_collapse_frac'] * 100,
            label=dataset,
            alpha=0.8,
        )
    ax.set_xlabel('Grokking step (val loss < 10% max)')
    ax.set_ylabel('Sinkhorn collapse (%)')
    ax.set_title('OT Collapse vs Grokking Time')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    pivot = summary.pivot_table(
        index='n_layer',
        columns='n_embd',
        values='sinkhorn_collapse_frac',
        aggfunc='mean',
    )
    image = ax.imshow(pivot.to_numpy() * 100, aspect='auto', origin='lower')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(i)) for i in pivot.index])
    ax.set_xlabel('Embedding dim')
    ax.set_ylabel('Layers')
    ax.set_title('Mean Sinkhorn Collapse (%)')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_dir / 'aggregate_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Aggregate runs/*/signals.csv')
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for csv_path in sorted(runs_dir.glob('*/signals.csv')):
        try:
            rows.append(summarize_run(csv_path))
        except Exception as exc:
            print(f'Skipping {csv_path}: {exc}')

    summary = pd.DataFrame(rows)
    summary_path = output_dir / 'summary.csv'
    summary.to_csv(summary_path, index=False)
    write_plots(summary, output_dir)
    print(f'Wrote {len(summary)} runs to {summary_path}')
    if not summary.empty:
        print(summary[['run', 'val_loss_final', 'grokking_step_10pct',
                       'sinkhorn_collapse_frac']].to_string(index=False))


if __name__ == '__main__':
    main()
