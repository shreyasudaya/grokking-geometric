import pandas as pd, numpy as np, os
from scipy.stats import pearsonr

def load_signals(run_dir):
    csv = os.path.join(run_dir, 'signals.csv')
    if not os.path.exists(csv): return None
    df = pd.read_csv(csv).sort_values('step')
    if len(df) < 3: return None
    return df

results = []
runs_dir = 'runs'
for d in sorted(os.listdir(runs_dir)):
    if 'modular_addition_' not in d or d.endswith('_seed42'):
        continue
    run = os.path.join(runs_dir, d)
    df = load_signals(run)
    if df is None: continue

    s = df['sinkhorn_mean'].values
    t = df['trace_normalized'].values if 'trace_normalized' in df.columns else df['trace'].values

    s_norm = (s - s.min()) / (s.max() - s.min() + 1e-10)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-10)

    sinkhorn_collapse_step = None
    trace_collapse_step = None

    for i in range(1, len(s_norm)):
        if s_norm[i] < 0.3 and s_norm[i-1] >= 0.3:
            sinkhorn_collapse_step = int(df['step'].iloc[i])
            break
    for i in range(1, len(t_norm)):
        if t_norm[i] < 0.3 and t_norm[i-1] >= 0.3:
            trace_collapse_step = int(df['step'].iloc[i])
            break

    val_final = df['val_loss'].iloc[-1]
    grokked = val_final < 0.5
    corr = np.corrcoef(s, t)[0, 1]

    best_lag = 0
    best_corr = -1
    max_lag = min(5, len(s) // 2)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            c = pearsonr(s[:lag], t[-lag:])[0] if len(s[:lag]) > 2 else 0
        elif lag > 0:
            c = pearsonr(s[lag:], t[:-lag])[0] if len(s[lag:]) > 2 else 0
        else:
            c = pearsonr(s, t)[0] if len(s) > 2 else 0
        if abs(c) > abs(best_corr):
            best_corr = c
            best_lag = lag

    results.append({
        'run': d, 'grokked': grokked, 'n_points': len(df),
        'sinkhorn_collapse': sinkhorn_collapse_step,
        'trace_collapse': trace_collapse_step,
        'raw_corr': corr, 'best_lag': best_lag, 'best_corr': best_corr,
        'sinkhorn_min': float(s.min()), 'sinkhorn_max': float(s.max()),
        'trace_max': float(t.max()),
    })

r = pd.DataFrame(results)
print("=" * 80)
print("  OT-Hessian temporal relationship across architectures")
print("=" * 80)
for _, row in r.iterrows():
    sc = row['sinkhorn_collapse']; tc = row['trace_collapse']
    if sc and tc:
        diff = sc - tc
        lag_str = f"Sinkhorn leads by {diff} steps" if diff > 0 else (f"Trace leads by {-diff} steps" if diff < 0 else "Simultaneous")
    elif sc: lag_str = "Only sinkhorn collapse"
    elif tc: lag_str = "Only trace collapse"
    else: lag_str = "No detectable collapse"
    print(f"  {row['run']:35s} | grokked={str(row['grokked']):>5} | corr={row['raw_corr']:+.3f} | lead={row['best_lag']:+d} | {lag_str}")

g = r[r['grokked'] == True]; ng = r[r['grokked'] == False]
print(f"\nResults: {len(g)} grokked, {len(ng)} did not")
print(f"  Sinkhorn collapsed in all grokked runs: {g['sinkhorn_collapse'].notna().all()}")
print(f"  Trace collapsed in all grokked runs:   {g['trace_collapse'].notna().all()}")
print(f"  Sinkhorn leads trace (lag>0) in:       {(r['best_lag'] > 0).sum()}/{len(r)} runs")
print(f"  Trace leads sinkhorn (lag<0) in:       {(r['best_lag'] < 0).sum()}/{len(r)} runs")
print(f"  Simultaneous (lag=0) in:               {(r['best_lag'] == 0).sum()}/{len(r)} runs")

print("\n--- Deep dive: modular_addition_L4_d256_seed0 ---")
df = load_signals('runs/modular_addition_L4_d256_seed0')
print(df[['step','val_loss','sinkhorn_mean','trace','trace_normalized','lambda_max']].to_string())

print("\n--- Deep dive: modular_addition_L6_d256_seed0 ---")
df = load_signals('runs/modular_addition_L6_d256_seed0')
print(df[['step','val_loss','sinkhorn_mean','trace','trace_normalized','lambda_max']].to_string())

print("\n--- Deep dive: modular_addition_L1_d64_seed0 ---")
df = load_signals('runs/modular_addition_L1_d64_seed0')
print(df[['step','val_loss','sinkhorn_mean','trace','trace_normalized','lambda_max']].to_string())
