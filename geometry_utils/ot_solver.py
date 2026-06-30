import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from contextlib import nullcontext
from model import GPTConfig, GPT

DATASET_CONFIGS = {
    'modular_addition':       {'vocab_size': 99, 'block_size': 4,  'n_output': 1},
    'modular_subtraction':    {'vocab_size': 99, 'block_size': 4,  'n_output': 1},
    'modular_multiplication': {'vocab_size': 99, 'block_size': 4,  'n_output': 1},
    'symmetric_group':        {'vocab_size': 7,  'block_size': 16, 'n_output': 5},
    'permutation_composition': {'vocab_size': 8,  'block_size': 19, 'n_output': 6},
}


def get_fixed_val_batch(dataset_name, block_size, n_output, num_examples, device):
    data_dir = os.path.join('data', dataset_name)
    data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    eq_length = block_size + 1
    num_eq = len(data) // eq_length
    n = min(num_examples, num_eq)
    ix = torch.arange(n, device='cpu') * eq_length
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    y[:, :-n_output] = -1
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def extract_hidden_states(model, X, Y, batch_size=None):
    """
    Extract hidden state activations at every layer.

    Hooks are placed on the embedding dropout (pre-block) and on each Block.
    Returns a list of tensors [embed_out, layer_0_out, ..., layer_{L-1}_out].
    Each tensor shape: (N, T, d_model).

    model is temporarily set to eval mode; original mode restored on exit.
    """
    was_training = model.training
    model.eval()

    n_layer = model.config.n_layer
    device = next(model.parameters()).device
    N = X.shape[0]

    activations = {}
    handles = []

    def make_embed_hook():
        def hook(module, inp, out):
            activations['embed'] = out.detach()
        return hook

    def make_block_hook(idx):
        def hook(module, inp, out):
            activations[f'layer_{idx}'] = out.detach()
        return hook

    handles.append(model.transformer.drop.register_forward_hook(make_embed_hook()))
    for i, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(make_block_hook(i)))

    with torch.no_grad():
        if batch_size is not None:
            n_batches = (N + batch_size - 1) // batch_size
            all_activations = {k: [] for k in ['embed'] + [f'layer_{i}' for i in range(n_layer)]}
            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, N)
                Xb = X[start:end]
                Yb = Y[start:end]
                model(Xb, Yb)
                for k in all_activations:
                    all_activations[k].append(activations[k])
            for k in all_activations:
                activations[k] = torch.cat(all_activations[k], dim=0)
        else:
            model(X, Y)

    for h in handles:
        h.remove()
    if was_training:
        model.train()

    keys = sorted(activations.keys(), key=lambda k: (0, k) if k == 'embed' else (1, k))
    return [activations[k] for k in keys]


class RandomProjection:
    """Johnson-Lindenstrauss random projection with Rademacher entries.

    Projects from d_in to d_out using P(±1/√d_out) = 1/2 each.
    """

    def __init__(self, d_in, d_out, seed=42):
        self.d_in = d_in
        self.d_out = d_out
        rng = torch.Generator().manual_seed(seed)
        self.proj = torch.randint(0, 2, (d_out, d_in), dtype=torch.float32, generator=rng)
        self.proj.mul_(2).add_(-1).div_(np.sqrt(d_out))

    def to(self, device):
        self.proj = self.proj.to(device)
        return self

    def __call__(self, x):
        N, T, d_in = x.shape
        return (x.reshape(-1, d_in) @ self.proj.T).reshape(N * T, self.d_out)


def batch_sinkhorn(C_batch, epsilon, num_iters=50, tol=1e-6, return_plan=False):
    """Batched log-domain stabilised Sinkhorn-Knopp.

    Args:
        C_batch: (B, N, N) cost matrices (squared Euclidean distances).
        epsilon: entropic regularisation strength.
        num_iters: maximum Sinkhorn iterations.
        tol: convergence threshold on row-scaling change.
        return_plan: if True also return (B, N, N) transport plans.

    Returns:
        distances: (B,) Sinkhorn distances sum(P * C).
        plans (optional): (B, N, N) transport plans.
    """
    B, N = C_batch.shape[0], C_batch.shape[-1]
    device = C_batch.device
    dtype = C_batch.dtype

    M = -C_batch / epsilon                         # (B, N, N) log-kernel
    log_a = torch.full((1, N), -np.log(N), device=device, dtype=dtype)   # uniform marginal
    log_b = torch.full((1, N), -np.log(N), device=device, dtype=dtype)

    u = torch.zeros(B, N, device=device, dtype=dtype)
    v = torch.zeros(B, N, device=device, dtype=dtype)

    for _ in range(num_iters):
        u_prev = u
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1)
        if torch.norm(u - u_prev) < tol:
            break

    # log-P = u_i + M_ij + v_j  ->  (B, N, N)
    log_P = u.unsqueeze(2) + M + v.unsqueeze(1)
    distances = torch.sum(torch.exp(log_P) * C_batch, dim=(1, 2))

    if return_plan:
        return distances, torch.exp(log_P)
    return distances


def layerwise_sinkhorn_distances(states, epsilon=0.01, num_iters=50, project=None):
    """Sinkhorn distances between every consecutive pair of layers.

    Args:
        states: list of (N, T, d) tensors per layer (embed + each Block).
        epsilon: entropic regularisation.
        num_iters: Sinkhorn iterations.
        project: RandomProjection instance or None.

    Returns:
        distances: list of float, length len(states) - 1.
        cost_matrices: list of (P, N) cost matrices (one per pair) for inspection.
    """
    if project is not None:
        pts = [project(h) for h in states]
    else:
        pts = [h.reshape(-1, h.shape[-1]) for h in states]

    N = pts[0].shape[0]
    L = len(pts)

    cost_list = []
    for l in range(L - 1):
        a = pts[l]      # (N, d)
        b = pts[l + 1]  # (N, d)
        C = torch.cdist(a, b, p=2).pow(2)   # (N, N)
        cost_list.append(C.unsqueeze(0))

    C_batch = torch.cat(cost_list, dim=0)   # (L-1, N, N)
    distances = batch_sinkhorn(C_batch, epsilon, num_iters=num_iters)

    return distances.tolist()


def layerwise_baseline_geometry(states, project=None, svcca_variance=0.99,
                                svcca_max_components=32):
    """Cheap representation baselines for comparison with OT.

    Returns activation scale per layer and linear CKA between consecutive
    layers. CKA is computed in feature space, so it avoids building an NxN
    Gram matrix.
    """
    if project is not None:
        pts = [project(h).float() for h in states]
    else:
        pts = [h.reshape(-1, h.shape[-1]).float() for h in states]

    activation_rms = [float(torch.sqrt((x.pow(2).mean())).item()) for x in pts]
    activation_std = [float(x.std(unbiased=False).item()) for x in pts]

    cka = []
    svcca_mean = []
    svcca_top = []
    for a, b in zip(pts[:-1], pts[1:]):
        a = a - a.mean(dim=0, keepdim=True)
        b = b - b.mean(dim=0, keepdim=True)
        cross = torch.linalg.matrix_norm(a.T @ b, ord='fro').pow(2)
        aa = torch.linalg.matrix_norm(a.T @ a, ord='fro')
        bb = torch.linalg.matrix_norm(b.T @ b, ord='fro')
        denom = aa * bb
        cka.append(float((cross / denom).item()) if denom.item() > 0 else 0.0)
        mean_corr, top_corr = svcca_similarity(
            a, b, variance=svcca_variance, max_components=svcca_max_components,
        )
        svcca_mean.append(mean_corr)
        svcca_top.append(top_corr)

    return {
        'activation_rms': activation_rms,
        'activation_std': activation_std,
        'linear_cka': cka,
        'svcca_mean': svcca_mean,
        'svcca_top': svcca_top,
    }


def _pca_scores(x, variance=0.99, max_components=32):
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[0] < 2 or x.shape[1] < 1:
        return x[:, :1]

    u, s, _ = torch.linalg.svd(x, full_matrices=False)
    energy = s.pow(2)
    total = energy.sum()
    if total.item() <= 0:
        return x[:, :1]

    keep = int(torch.searchsorted(torch.cumsum(energy, dim=0) / total,
                                  torch.tensor(float(variance), device=x.device)).item() + 1)
    keep = max(1, min(keep, int(max_components), s.numel(), x.shape[0] - 1))
    return u[:, :keep] * s[:keep]


def _inv_sqrt_psd(mat, eps=1e-5):
    vals, vecs = torch.linalg.eigh(mat)
    vals = vals.clamp_min(eps)
    return (vecs * torch.rsqrt(vals).unsqueeze(0)) @ vecs.T


def svcca_similarity(a, b, variance=0.99, max_components=32, eps=1e-5):
    """SVCCA similarity between two sample-by-feature activation matrices."""
    x = _pca_scores(a, variance=variance, max_components=max_components)
    y = _pca_scores(b, variance=variance, max_components=max_components)
    n = min(x.shape[0], y.shape[0])
    if n < 2:
        return 0.0, 0.0

    x = x[:n] - x[:n].mean(dim=0, keepdim=True)
    y = y[:n] - y[:n].mean(dim=0, keepdim=True)
    denom = max(1, n - 1)
    cxx = (x.T @ x) / denom
    cyy = (y.T @ y) / denom
    cxy = (x.T @ y) / denom
    cca = _inv_sqrt_psd(cxx, eps=eps) @ cxy @ _inv_sqrt_psd(cyy, eps=eps)
    corr = torch.linalg.svdvals(cca).clamp(0.0, 1.0)
    if corr.numel() == 0:
        return 0.0, 0.0
    return float(corr.mean().item()), float(corr[0].item())


def layerwise_ot_pipeline(ckpt_path, dataset=None, device='cuda', num_examples=512,
                          target_dim=None, epsilon=0.01, sinkhorn_iters=50, seed=42,
                          svcca_variance=0.99, svcca_max_components=32):
    """Load a checkpoint, extract hidden states, and compute layerwise Sinkhorn distances.

    Returns a dict with full results.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = ckpt['model_args']
    dataset_name = ckpt.get('dataset_name', dataset or 'modular_addition')
    n_output = ckpt.get('n_output', DATASET_CONFIGS[dataset_name]['n_output'])
    iter_num = ckpt.get('iter_num', 0)

    dc = DATASET_CONFIGS[dataset_name]
    block_size = dc['block_size']

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)
    model.load_state_dict(ckpt['model'])
    for block in model.transformer.h:
        attn = block.attn
        attn.flash = False
        if not hasattr(attn, 'bias'):
            attn.register_buffer("bias", torch.tril(
                torch.ones(block_size, block_size, device=device)
            ).view(1, 1, block_size, block_size))

    X, Y = get_fixed_val_batch(dataset_name, block_size, n_output, num_examples, device)
    actual_n = X.shape[0]

    states = extract_hidden_states(model, X, Y, batch_size=256)
    # states: list of (actual_n, block_size, d_model)

    n_embd = model_args['n_embd']
    if target_dim is not None:
        proj = RandomProjection(n_embd, target_dim, seed=seed).to(device)
    else:
        proj = None

    distances = layerwise_sinkhorn_distances(states, epsilon=epsilon,
                                              num_iters=sinkhorn_iters, project=proj)
    baselines = layerwise_baseline_geometry(
        states,
        project=proj,
        svcca_variance=svcca_variance,
        svcca_max_components=svcca_max_components,
    )

    results = {
        'checkpoint': ckpt_path,
        'iter_num': iter_num,
        'dataset': dataset_name,
        'num_examples': actual_n,
        'n_layer': model_args['n_layer'],
        'n_embd': n_embd,
        'target_dim': target_dim or n_embd,
        'epsilon': epsilon,
        'sinkhorn_iters': sinkhorn_iters,
        'distances': distances,
        'baselines': baselines,
    }
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Layerwise Sinkhorn distances')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset', type=str, default=None, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-examples', type=int, default=512)
    parser.add_argument('--target-dim', type=int, default=None, help='JL projection target dim')
    parser.add_argument('--epsilon', type=float, default=0.01, help='entropic regularisation')
    parser.add_argument('--sinkhorn-iters', type=int, default=50)
    parser.add_argument('--svcca-variance', type=float, default=0.99)
    parser.add_argument('--svcca-max-components', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--json', type=str, default=None, help='save results to JSON')
    args = parser.parse_args()

    results = layerwise_ot_pipeline(
        ckpt_path=args.ckpt,
        dataset=args.dataset,
        device=args.device,
        num_examples=args.num_examples,
        target_dim=args.target_dim,
        epsilon=args.epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
        svcca_variance=args.svcca_variance,
        svcca_max_components=args.svcca_max_components,
        seed=args.seed,
    )

    print(f"\nCheckpoint step={results['iter_num']}  ({results['dataset']})")
    print(f"Examples: {results['num_examples']},  d={results['target_dim']},  "
          f"eps={results['epsilon']}")
    L = len(results['distances'])
    print(f"Layerwise Sinkhorn distances (embed -> layer_0 -> ... -> layer_{L-1}):")
    for i, d in enumerate(results['distances']):
        print(f"  layer_{i} -> layer_{i+1}: {d:.6f}")

    if args.json:
        def convert(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, torch.Tensor):
                return o.item()
            if isinstance(o, (np.floating,)):
                return float(o)
            return o
        with open(args.json, 'w') as f:
            json.dump(results, f, default=convert, indent=2)
        print(f"Saved to {args.json}")
