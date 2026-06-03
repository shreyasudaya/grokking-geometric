import os
import argparse
import torch
import numpy as np
from model import GPTConfig, GPT

parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', type=str, default='out-grokking', help='checkpoint directory')
parser.add_argument('--max-step', type=int, default=4000, help='max checkpoint step to analyze')
parser.add_argument('--batch-size', type=int, default=256, help='batch size for Hessian computation')
parser.add_argument('--power-iters', type=int, default=15, help='power iteration count')
parser.add_argument('--hutchinson-samples', type=int, default=5, help='Hutchinson trace samples')
args = parser.parse_args()

out_dir = args.out_dir
batch_size = args.batch_size
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def get_data_config(ckpt_path):
    try:
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception:
        ck = torch.load(ckpt_path, map_location='cpu')
    block_size = ck['model_args']['block_size']
    n_output = ck.get('n_output', 1)
    dataset_name = ck.get('dataset_name', 'modular_addition')
    data_dir = os.path.join('data', dataset_name)
    return data_dir, block_size, n_output, ck

def get_eval_batch(data_dir, block_size, n_output):
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    eq_length = block_size + 1
    num_equations = len(train_data) // eq_length
    ix_eq = torch.randint(0, num_equations, (batch_size,))
    ix = ix_eq * eq_length

    x = torch.stack([torch.from_numpy((train_data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((train_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    y[:, :-n_output] = -1
    return x.to(device), y.to(device)

def get_params(model):
    return [p for p in model.parameters() if p.requires_grad]

def compute_hvp(loss, params, v):
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grad_flat = torch.cat([g.view(-1) for g in grads])
    grad_v = torch.sum(grad_flat * v)
    hvp = torch.autograd.grad(grad_v, params, retain_graph=True)
    return torch.cat([h.view(-1) for h in hvp])

def power_iteration(loss, params, num_iters=20):
    num_params = sum(p.numel() for p in params)
    v = torch.randn(num_params, device=device)
    v = v / torch.norm(v)

    for _ in range(num_iters):
        Hv = compute_hvp(loss, params, v)
        v = Hv / torch.norm(Hv)

    lambda_max = torch.sum(v * compute_hvp(loss, params, v)).item()
    return lambda_max

def hutchinson_trace(loss, params, num_samples=10):
    num_params = sum(p.numel() for p in params)
    trace_est = 0.0

    for _ in range(num_samples):
        v = torch.randint(0, 2, (num_params,), device=device).float() * 2 - 1
        Hv = compute_hvp(loss, params, v)
        trace_est += torch.sum(v * Hv).item()

    return trace_est / num_samples

print("Scanning checkpoints...")

checkpoints = [
    f for f in os.listdir(out_dir)
    if f.startswith('ckpt_')
    and f.endswith('.pt')
    and int(f.split('_')[1].split('.')[0]) <= args.max_step
]

checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

if len(checkpoints) == 0:
    print(f"No checkpoints found in {out_dir}")
    exit()

_, block_size, n_output, _ = get_data_config(os.path.join(out_dir, checkpoints[0]))
data_dir, _, _, _ = get_data_config(os.path.join(out_dir, checkpoints[0]))
print(f"Dataset config: data_dir={data_dir}, block_size={block_size}, n_output={n_output}")

X, Y = get_eval_batch(data_dir, block_size, n_output)

results = []

for ckpt_name in checkpoints:
    step = int(ckpt_name.split('_')[1].split('.')[0])
    ckpt_path = os.path.join(out_dir, ckpt_name)

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception:
        checkpoint = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**checkpoint['model_args']))
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    params = get_params(model)

    logits, loss = model(X, Y)

    print(f"Analyzing Step {step}...")
    lambda_max = power_iteration(loss, params, num_iters=args.power_iters)
    trace = hutchinson_trace(loss, params, num_samples=args.hutchinson_samples)

    print(f"--> Step {step} | Loss: {loss.item():.4f} | Lambda Max: {lambda_max:.2f} | Trace: {trace:.2f}")

    results.append((step, loss.item(), lambda_max, trace))

np.save(os.path.join(out_dir, 'hessian_metrics.npy'), np.array(results))
print("Analysis complete. Saved to hessian_metrics.npy")
