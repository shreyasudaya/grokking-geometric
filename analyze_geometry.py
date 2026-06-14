import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from hessian import analyze_checkpoint as hessian_analyze
from geometry_utils.ot_solver import layerwise_ot_pipeline


def extract_step(filename):
    match = re.search(r'ckpt_(\d+)\.pt', filename)
    return int(match.group(1)) if match else -1


def detect_stride(checkpoint_dir, max_samples=200):
    files = [f for f in os.listdir(checkpoint_dir) if re.match(r'ckpt_\d+\.pt', f)]
    n = len(files)
    if n <= max_samples:
        return 1
    return max(1, n // max_samples)


def analyze_geometry(checkpoint_dir, dataset=None, stride=1, device='cuda',
                     num_examples=512, hutchinson_samples=5, power_iters=20,
                     target_dim=None, epsilon=0.05, sinkhorn_iters=30,
                     seed=42):
    ckpt_files = [f for f in os.listdir(checkpoint_dir) if re.match(r'ckpt_\d+\.pt', f)]
    ckpt_files.sort(key=extract_step)
    ckpt_files = ckpt_files[::stride]

    n_layer = None
    records = []

    for ckpt_name in tqdm(ckpt_files, desc="Analyzing checkpoints"):
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        step = extract_step(ckpt_name)

        h_results = None
        try:
            h_results = hessian_analyze(
                ckpt_path=ckpt_path,
                dataset=dataset,
                device=device,
                batch_size=512,
                hutchinson_samples=hutchinson_samples,
                power_iters=power_iters,
                seed=seed,
            )
        except Exception as e:
            h_results = None

        ot_results = None
        try:
            ot_results = layerwise_ot_pipeline(
                ckpt_path=ckpt_path,
                dataset=dataset,
                device=device,
                num_examples=num_examples,
                target_dim=target_dim,
                epsilon=epsilon,
                sinkhorn_iters=sinkhorn_iters,
                seed=seed,
            )
        except Exception as e:
            ot_results = None

        record = {'step': step}

        if h_results is not None:
            record['train_loss'] = h_results.get('train_loss')
            record['val_loss'] = h_results.get('val_loss')
            record['lambda_max'] = h_results['power_iteration']['lambda_max']
            record['trace'] = h_results['hutchinson']['trace']
            record['trace_normalized'] = h_results['hutchinson']['trace_normalized']
            record['frobenius_norm'] = h_results['hutchinson']['frobenius_norm']
            if n_layer is None and 'n_layer' in h_results:
                n_layer = h_results.get('n_layer')

        if ot_results is not None:
            for i, d in enumerate(ot_results['distances']):
                record[f'sinkhorn_L{i}_to_L{i+1}'] = d

        records.append(record)

    df = pd.DataFrame(records).sort_values('step').reset_index(drop=True)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Master geometry analysis harness')
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--output', type=str, default='geometric_signals.csv')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-examples', type=int, default=256)
    parser.add_argument('--hutchinson-samples', type=int, default=5)
    parser.add_argument('--power-iters', type=int, default=20)
    parser.add_argument('--target-dim', type=int, default=None)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--sinkhorn-iters', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.stride is None:
        args.stride = detect_stride(args.checkpoint_dir)
        print(f"Auto-stride: {args.stride}")

    df = analyze_geometry(
        checkpoint_dir=args.checkpoint_dir,
        dataset=args.dataset,
        stride=args.stride,
        device=args.device,
        num_examples=args.num_examples,
        hutchinson_samples=args.hutchinson_samples,
        power_iters=args.power_iters,
        target_dim=args.target_dim,
        epsilon=args.epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
        seed=args.seed,
    )

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} records to {args.output}")
    print(f"Columns: {', '.join(df.columns)}")
