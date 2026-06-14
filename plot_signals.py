import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ruptures as rpt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def detect_cpts(series, model='l1', pen=10):
    """Detect changepoints using PELT.
    
    model='l1' (cost=absolute deviation) is preferred over 'rbf' for finding
    the *start* of a regime change (inflection point). RBF kernel detects the
    *end* of a transition (variance collapse), which lags the actual inflection.
    """
    algo = rpt.Pelt(model=model, min_size=5, jump=1).fit(series.values.reshape(-1, 1))
    cpts = algo.predict(pen=pen)
    return [c for c in cpts if c < len(series)]


def make_stationary(series):
    """Ensure stationarity via first-differencing if ADF test fails."""
    cleaned = series.dropna()
    if len(cleaned) < 6:
        return cleaned, False
    adf = adfuller(cleaned, autolag='AIC')
    p = adf[1]
    if p < 0.05:
        return cleaned, False
    diffed = cleaned.diff().dropna()
    return diffed, True


def granger_test(df, cause_col, effect_col, maxlag=10):
    """Granger causality test on stationary (differenced) data."""
    data = df[[cause_col, effect_col]].dropna()
    if len(data) < maxlag + 5:
        return None, None, None, None

    # If either series is non-stationary, difference both
    _, cause_diffed = make_stationary(data[cause_col])
    _, effect_diffed = make_stationary(data[effect_col])

    if cause_diffed or effect_diffed:
        d = data.diff().dropna()
        stationary_data = d.dropna()
    else:
        stationary_data = data

    if len(stationary_data) < maxlag + 5:
        return None, None, None, None

    try:
        res = grangercausalitytests(stationary_data, maxlag=maxlag, verbose=False)
    except Exception:
        return None, None, None, None

    best_lag, best_p = 1, 1.0
    for lag, result in res.items():
        p = result[0]['params_ftest'][1]
        if p < best_p:
            best_p = p
            best_lag = lag
    return best_lag, best_p, cause_diffed, effect_diffed


def plot_signals(csv_path, output_path, pen=10, maxlag=10):
    df = pd.read_csv(csv_path)
    df = df.sort_values('step').reset_index(drop=True)

    sinkhorn_cols = [c for c in df.columns if c.startswith('sinkhorn_')]
    has_sinkhorn = len(sinkhorn_cols) > 0

    # PELT on trace (RBF, low pen — only picks up major regime shifts)
    trace_valid = df['trace'].dropna()
    trace_cpts = detect_cpts(trace_valid, model='rbf', pen=max(1.0, pen * 0.1)) if len(trace_valid) > 10 else []

    # PELT on sinkhorn (L1 cost — detects inflection/start of collapse, not end)
    sinkhorn_cpts = []
    if has_sinkhorn:
        if len(sinkhorn_cols) > 1:
            df['sinkhorn_mean'] = df[sinkhorn_cols].mean(axis=1)
            sinkhorn_main = 'sinkhorn_mean'
        else:
            sinkhorn_main = sinkhorn_cols[0]
        sinkhorn_valid = df[sinkhorn_main].dropna()
        if len(sinkhorn_valid) > 10:
            sinkhorn_cpts = detect_cpts(sinkhorn_valid, model='l1', pen=pen)
    else:
        sinkhorn_main = None

    # Granger causality: Sinkhorn -> trace (on stationary data)
    if sinkhorn_main is not None:
        lag, pval, cause_diffed, effect_diffed = granger_test(df, sinkhorn_main, 'trace', maxlag=maxlag)
        if lag is not None:
            stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
            note = ''
            if cause_diffed or effect_diffed:
                note = ' (differenced)'
            print(f"Granger: {sinkhorn_main} -> trace   lag={lag}, p={pval:.6f} {stars}{note}")

    # Build plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    steps = df['step'].values

    # A: Loss
    ax = axes[0]
    if 'train_loss' in df.columns:
        ax.plot(steps, df['train_loss'].values, label='Train Loss', alpha=0.7)
    if 'val_loss' in df.columns:
        ax.plot(steps, df['val_loss'].values, label='Val Loss', alpha=0.7)
    ax.set_ylabel('Loss')
    ax.legend(fontsize=9)
    ax.set_title('A: Loss Curves')
    ax.grid(True, alpha=0.3)

    # B: Hessian
    ax = axes[1]
    ax_twin = ax.twinx()
    if 'lambda_max' in df.columns:
        ax.plot(steps, df['lambda_max'].values, label=r'$\lambda_{\max}$', color='tab:blue', alpha=0.8)
    if 'trace' in df.columns:
        ax_twin.plot(steps, df['trace'].values, label=r'Tr($H$)', color='tab:orange', alpha=0.8)
    ax.set_ylabel(r'$\lambda_{\max}$', color='tab:blue')
    ax_twin.set_ylabel(r'Tr($H$)', color='tab:orange')
    ax_twin.tick_params(axis='y', labelcolor='tab:orange')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    for c in trace_cpts:
        if 0 < c - 1 < len(steps):
            step_val = steps[c - 1]
            ax.axvline(x=step_val, color='orange', ls='--', alpha=0.6, lw=1.2,
                       label=f'Trace CP (step {step_val})' if c == trace_cpts[0] else '')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
    ax.set_title('B: Hessian Curvature')
    ax.grid(True, alpha=0.3)

    # C: Sinkhorn
    ax = axes[2]
    if has_sinkhorn:
        for col in sinkhorn_cols:
            label = col.replace('sinkhorn_', '')
            ax.plot(steps, df[col].values, label=label, alpha=0.7, lw=1)
        for c in sinkhorn_cpts:
            if 0 < c - 1 < len(steps):
                step_val = steps[c - 1]
                ax.axvline(x=step_val, color='green', ls='--', alpha=0.6, lw=1.2,
                           label=f'Sinkhorn CP (step {step_val})' if c == sinkhorn_cpts[0] else '')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Sinkhorn Distance')
    ax.legend(fontsize=9)
    ax.set_title('C: Layerwise Optimal Transport Distances')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

    # Print PELT results (c-1 converts ruptures' 1-indexed to 0-indexed step value)
    if trace_cpts:
        print(f"PELT change points on trace: {[steps[c - 1] for c in trace_cpts if 0 < c - 1 < len(steps)]}")
    if sinkhorn_cpts:
        print(f"PELT change points on sinkhorn: {[steps[c - 1] for c in sinkhorn_cpts if 0 < c - 1 < len(steps)]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot geometric signals with PELT + Granger')
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output', type=str, default='geometric_signals.png')
    parser.add_argument('--pen', type=float, default=10, help='PELT penalty')
    parser.add_argument('--maxlag', type=int, default=10, help='Granger max lag')
    args = parser.parse_args()

    plot_signals(args.csv, args.output, pen=args.pen, maxlag=args.maxlag)
