import os
import json
import argparse
import numpy as np
import torch
from contextlib import nullcontext
from model import GPTConfig, GPT

DATASET_CONFIGS = {
    'modular_addition':       {'vocab_size': 99, 'block_size': 4,  'n_output': 1},
    'modular_subtraction':    {'vocab_size': 99, 'block_size': 4,  'n_output': 1},
    'modular_multiplication': {'vocab_size': 99, 'block_size': 4,  'n_output': 1},
    'symmetric_group':        {'vocab_size': 7,  'block_size': 16, 'n_output': 5},
    'permutation_composition': {'vocab_size': 8,  'block_size': 19, 'n_output': 6},
}


def compute_hvp(model, loss_fn, z_list):
    params = [p for p in model.parameters() if p.requires_grad]
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True)

    z_flat = torch.cat([zi.contiguous().view(-1) for zi in z_list])
    grad_flat = torch.cat([g.contiguous().view(-1) for g in grads])
    dot = (grad_flat * z_flat).sum()

    hvp = torch.autograd.grad(dot, params, retain_graph=False)
    hvp_flat = torch.cat([h.contiguous().view(-1) for h in hvp])
    return hvp_flat, dot.item()


def hutchinson_trace(model, loss_fn, num_samples=10, seed=42):
    params = [p for p in model.parameters() if p.requires_grad]
    trace_sum = 0.0
    frob_sq_sum = 0.0

    for i in range(num_samples):
        torch.manual_seed(seed + i)

        z = [torch.randint(0, 2, p.shape, device=p.device, dtype=p.dtype).mul_(2).add_(-1) for p in params]
        hvp_flat, _ = compute_hvp(model, loss_fn, z)

        z_flat = torch.cat([zi.contiguous().view(-1) for zi in z])
        trace_sum += (z_flat * hvp_flat).sum().item()
        frob_sq_sum += (hvp_flat ** 2).sum().item()

    trace_est = trace_sum / num_samples
    frob_norm = np.sqrt(frob_sq_sum / num_samples)

    return {
        'trace': trace_est,
        'trace_normalized': trace_est / frob_norm if frob_norm > 0 else 0.0,
        'frobenius_norm': frob_norm,
    }


def power_iteration(model, loss_fn, num_iters=50, seed=42):
    params = [p for p in model.parameters() if p.requires_grad]
    device = params[0].device
    nparams = sum(p.numel() for p in params)

    torch.manual_seed(seed)
    v_flat = torch.randn(nparams, device=device)
    v_flat.div_(torch.norm(v_flat))

    history = []

    def split(vf, param_list):
        out, idx = [], 0
        for p in param_list:
            out.append(vf[idx:idx + p.numel()].view(p.shape))
            idx += p.numel()
        return out

    for _ in range(num_iters):
        v = split(v_flat, params)
        hvp_flat, _ = compute_hvp(model, loss_fn, v)

        eig = (v_flat * hvp_flat).sum().item()
        history.append(eig)

        v_flat = hvp_flat / torch.norm(hvp_flat)

    return {
        'lambda_max': history[-1],
        'eigenvalue_history': history,
    }


def evaluate_loss(model, X, Y, ctx):
    model.eval()
    with torch.no_grad(), ctx:
        _, loss = model(X, Y)
    model.train()
    return loss.item()


def gradient_norm(model, loss_fn):
    params = [p for p in model.parameters() if p.requires_grad]
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False, allow_unused=True)
    sq_sum = 0.0
    n_elems = 0
    for grad in grads:
        if grad is None:
            continue
        g = grad.detach().float()
        sq_sum += g.pow(2).sum().item()
        n_elems += g.numel()
    grad_l2 = float(np.sqrt(sq_sum))
    grad_rms = float(np.sqrt(sq_sum / n_elems)) if n_elems else 0.0
    return grad_l2, grad_rms


def classification_stats(model, X, Y, ctx):
    model.eval()
    with torch.no_grad(), ctx:
        logits, _ = model(X, Y)
    valid = Y != -1
    if valid.sum().item() == 0:
        model.train()
        return {
            'entropy': np.nan,
            'prob_margin': np.nan,
            'logit_margin': np.nan,
            'true_logit_margin': np.nan,
        }

    logits = logits[valid].float()
    targets = Y[valid]
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean().item()

    top2_probs = torch.topk(probs, k=2, dim=-1).values
    top2_logits = torch.topk(logits, k=2, dim=-1).values
    prob_margin = (top2_probs[:, 0] - top2_probs[:, 1]).mean().item()
    logit_margin = (top2_logits[:, 0] - top2_logits[:, 1]).mean().item()

    true_logits = logits.gather(1, targets.view(-1, 1)).squeeze(1)
    other_logits = logits.clone()
    other_logits.scatter_(1, targets.view(-1, 1), -torch.inf)
    true_logit_margin = (true_logits - other_logits.max(dim=-1).values).mean().item()

    model.train()
    return {
        'entropy': float(entropy),
        'prob_margin': float(prob_margin),
        'logit_margin': float(logit_margin),
        'true_logit_margin': float(true_logit_margin),
    }


def analyze_checkpoint(ckpt_path, dataset=None, device='cuda', batch_size=512,
                       hutchinson_samples=10, power_iters=50, seed=42,
                       train_fraction=1.0):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = ckpt['model_args']
    dataset_name = ckpt.get('dataset_name', dataset or 'modular_addition')
    n_output = ckpt.get('n_output', DATASET_CONFIGS[dataset_name]['n_output'])
    iter_num = ckpt.get('iter_num', 0)
    ckpt_val_loss = ckpt.get('val_loss', None)
    train_fraction = float(ckpt.get('train_fraction', train_fraction))
    dc = DATASET_CONFIGS[dataset_name]
    block_size = dc['block_size']

    print(f"Checkpoint step={iter_num}, dataset={dataset_name}, val_loss={ckpt_val_loss}")

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.train()
    with torch.no_grad():
        param_sq = sum(p.detach().float().pow(2).sum().item() for p in model.parameters())
        n_params = sum(p.numel() for p in model.parameters())
        param_l2 = float(np.sqrt(param_sq))
        param_rms = float(np.sqrt(param_sq / n_params)) if n_params else 0.0
    for block in model.transformer.h:
        attn = block.attn
        attn.flash = False
        if not hasattr(attn, 'bias'):
            attn.register_buffer("bias", torch.tril(torch.ones(block_size, block_size, device=device))
                                 .view(1, 1, block_size, block_size))

    data_dir = os.path.join('data', dataset_name)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    eq_length = block_size + 1
    train_num_eq = len(train_data) // eq_length
    train_subset = None
    if train_fraction < 1.0:
        if train_fraction <= 0.0:
            raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")
        subset_size = max(1, int(round(train_num_eq * train_fraction)))
        rng = np.random.default_rng(int(seed))
        train_subset = np.sort(
            rng.choice(train_num_eq, size=subset_size, replace=False)
        ).astype(np.int64)

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        num_eq = len(data) // eq_length
        if split == 'train' and train_subset is not None:
            choices = torch.randint(0, len(train_subset), (batch_size,)).numpy()
            ix_eq = train_subset[choices]
        else:
            ix_eq = torch.randint(0, num_eq, (batch_size,)).numpy()
        ix = ix_eq * eq_length
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        y[:, :-n_output] = -1
        if device.startswith('cuda'):
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    device_type = 'cuda' if device.startswith('cuda') else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

    X, Y = get_batch('train')
    train_loss = evaluate_loss(model, X, Y, ctx)
    Xv, Yv = get_batch('val')
    val_loss = evaluate_loss(model, Xv, Yv, ctx)
    print(f"  train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    train_stats = classification_stats(model, X, Y, ctx)
    val_stats = classification_stats(model, Xv, Yv, ctx)

    def loss_fn():
        with ctx:
            _, loss = model(X, Y)
        return loss

    grad_l2, grad_rms = gradient_norm(model, loss_fn)

    print("Hutchinson trace estimate...")
    hutch = hutchinson_trace(model, loss_fn, num_samples=hutchinson_samples, seed=seed)

    print("Power iteration...")
    power = power_iteration(model, loss_fn, num_iters=power_iters, seed=seed + 1000)

    results = {
        'checkpoint': ckpt_path,
        'iter_num': iter_num,
        'dataset': dataset_name,
        'ckpt_val_loss': ckpt_val_loss,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'param_l2': param_l2,
        'param_rms': param_rms,
        'grad_l2': grad_l2,
        'grad_rms': grad_rms,
        'train_stats': train_stats,
        'val_stats': val_stats,
        'hutchinson': hutch,
        'power_iteration': power,
    }
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hessian curvature estimators')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset', type=str, default=None, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hutchinson-samples', type=int, default=10)
    parser.add_argument('--power-iters', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-fraction', type=float, default=1.0)
    parser.add_argument('--json', type=str, default=None, help='save results to JSON file')
    args = parser.parse_args()

    results = analyze_checkpoint(
        ckpt_path=args.ckpt,
        dataset=args.dataset,
        device=args.device,
        batch_size=args.batch_size,
        hutchinson_samples=args.hutchinson_samples,
        power_iters=args.power_iters,
        seed=args.seed,
        train_fraction=args.train_fraction,
    )

    print("\n=== Results ===")
    print(f"Iter: {results['iter_num']}, val_loss: {results['val_loss']:.6f}")
    print(f"Tr(H) = {results['hutchinson']['trace']:.6f}")
    print(f"Tr(H)/||H||_F = {results['hutchinson']['trace_normalized']:.6f}")
    print(f"||H||_F = {results['hutchinson']['frobenius_norm']:.6f}")
    print(f"lambda_max = {results['power_iteration']['lambda_max']:.6f}")

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
