import argparse
import json
import os

import numpy as np
import torch

from model import GPTConfig, GPT
from geometry_utils.ot_solver import DATASET_CONFIGS, get_fixed_val_batch


def _prepare_model(ckpt, device, block_size):
    model = GPT(GPTConfig(**ckpt['model_args']))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    for block in model.transformer.h:
        attn = block.attn
        attn.flash = False
        if not hasattr(attn, 'bias'):
            attn.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size, device=device))
                .view(1, 1, block_size, block_size),
            )
    return model


def _loss_and_accuracy(model, X, Y):
    with torch.no_grad():
        logits, loss = model(X, Y)
    valid = Y != -1
    if valid.sum().item() == 0:
        accuracy = np.nan
    else:
        pred = logits.argmax(dim=-1)
        accuracy = (pred[valid] == Y[valid]).float().mean().item()
    return float(loss.item()), float(accuracy)


def _layer_modules(model):
    layers = [('embed', model.transformer.drop)]
    layers.extend((f'layer_{i}', block) for i, block in enumerate(model.transformer.h))
    return layers


def _apply_intervention(out, mode, noise_scale):
    if mode == 'zero':
        return torch.zeros_like(out)
    if mode == 'mean_ablate':
        return out.mean(dim=0, keepdim=True).expand_as(out)
    if mode == 'shuffle':
        perm = torch.randperm(out.shape[0], device=out.device)
        return out.index_select(0, perm)
    if mode == 'noise':
        scale = out.detach().float().std(unbiased=False).clamp_min(1e-8)
        return out + torch.randn_like(out) * scale.to(out.dtype) * float(noise_scale)
    raise ValueError(f"Unknown intervention mode: {mode}")


def evaluate_interventions(model, X, Y, modes=None, noise_scale=0.5, seed=42):
    modes = list(modes or ['mean_ablate', 'shuffle', 'noise'])
    clean_loss, clean_accuracy = _loss_and_accuracy(model, X, Y)
    rows = []

    for layer_idx, (layer_name, module) in enumerate(_layer_modules(model)):
        for mode_idx, mode in enumerate(modes):
            torch.manual_seed(int(seed) + 1009 * layer_idx + 37 * mode_idx)

            def hook(_module, _inp, out, intervention_mode=mode):
                return _apply_intervention(out, intervention_mode, noise_scale)

            handle = module.register_forward_hook(hook)
            try:
                loss, accuracy = _loss_and_accuracy(model, X, Y)
            finally:
                handle.remove()

            rows.append({
                'layer': layer_name,
                'mode': mode,
                'loss': loss,
                'accuracy': accuracy,
                'loss_delta': loss - clean_loss,
                'accuracy_delta': accuracy - clean_accuracy,
            })

    return {
        'clean_loss': clean_loss,
        'clean_accuracy': clean_accuracy,
        'rows': rows,
    }


def intervention_pipeline(ckpt_path, dataset=None, device='cuda', num_examples=128,
                          modes=None, noise_scale=0.5, seed=42):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    dataset_name = ckpt.get('dataset_name', dataset or 'modular_addition')
    n_output = ckpt.get('n_output', DATASET_CONFIGS[dataset_name]['n_output'])
    dc = DATASET_CONFIGS[dataset_name]
    model = _prepare_model(ckpt, device, dc['block_size'])
    X, Y = get_fixed_val_batch(dataset_name, dc['block_size'], n_output,
                               num_examples, device)

    results = evaluate_interventions(
        model,
        X,
        Y,
        modes=modes,
        noise_scale=noise_scale,
        seed=seed,
    )
    results.update({
        'checkpoint': ckpt_path,
        'dataset': dataset_name,
        'iter_num': ckpt.get('iter_num', 0),
        'num_examples': int(X.shape[0]),
    })
    return results


def flatten_intervention_results(results):
    flat = {
        'intervention_clean_loss': results.get('clean_loss'),
        'intervention_clean_accuracy': results.get('clean_accuracy'),
    }
    by_mode = {}
    for row in results.get('rows', []):
        layer = 'E' if row['layer'] == 'embed' else row['layer'].replace('layer_', 'L')
        mode = row['mode']
        prefix = f"intervene_{mode}_{layer}"
        flat[f'{prefix}_loss_delta'] = row['loss_delta']
        flat[f'{prefix}_accuracy_delta'] = row['accuracy_delta']
        by_mode.setdefault(mode, {'loss_delta': [], 'accuracy_delta': []})
        by_mode[mode]['loss_delta'].append(row['loss_delta'])
        by_mode[mode]['accuracy_delta'].append(row['accuracy_delta'])

    for mode, vals in by_mode.items():
        flat[f'intervene_{mode}_loss_delta_mean'] = float(np.nanmean(vals['loss_delta']))
        flat[f'intervene_{mode}_accuracy_delta_mean'] = float(np.nanmean(vals['accuracy_delta']))
    return flat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Activation intervention metrics')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset', type=str, default=None, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-examples', type=int, default=128)
    parser.add_argument('--modes', type=str, default='mean_ablate,shuffle,noise')
    parser.add_argument('--noise-scale', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--json', type=str, default=None)
    args = parser.parse_args()

    results = intervention_pipeline(
        ckpt_path=args.ckpt,
        dataset=args.dataset,
        device=args.device,
        num_examples=args.num_examples,
        modes=[m.strip() for m in args.modes.split(',') if m.strip()],
        noise_scale=args.noise_scale,
        seed=args.seed,
    )

    print(f"Checkpoint step={results['iter_num']} ({results['dataset']})")
    print(f"Clean: loss={results['clean_loss']:.6f}, accuracy={results['clean_accuracy']:.4f}")
    for row in results['rows']:
        print(
            f"{row['mode']:>11s} {row['layer']:>8s}: "
            f"loss_delta={row['loss_delta']:+.6f}, "
            f"accuracy_delta={row['accuracy_delta']:+.4f}"
        )

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.json}")
