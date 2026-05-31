import os
import torch
import numpy as np
from model import GPTConfig, GPT

# --- Configuration ---
out_dir = 'out-grokking'
dataset = 'modular_addition'
batch_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
# --- Load Data for Evaluation ---
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

def get_eval_batch():
    # Grab a fixed batch of training data to compute the Hessian over
    block_size = 4
    eq_length = block_size + 1
    num_equations = len(train_data) // eq_length
    ix_eq = torch.randint(0, num_equations, (batch_size,))
    ix = ix_eq * eq_length 
    
    x = torch.stack([torch.from_numpy((train_data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((train_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    y[:, :-1] = -1 # Mask intermediate tokens just like in training
    return x.to(device), y.to(device)

# --- Hessian Utility Functions ---
def get_params(model):
    return [p for p in model.parameters() if p.requires_grad]

def compute_hvp(loss, params, v):
    """Computes the Hessian-Vector Product (H*v) using double backprop."""
    # First derivative (Gradient)
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    
    # Flatten grads and multiply by vector v
    grad_flat = torch.cat([g.view(-1) for g in grads])
    grad_v = torch.sum(grad_flat * v)
    
    # Second derivative (Hessian-vector product)
    hvp = torch.autograd.grad(grad_v, params, retain_graph=True)
    return torch.cat([h.view(-1) for h in hvp])

def power_iteration(loss, params, num_iters=20):
    """Finds the top eigenvalue (lambda_max) of the Hessian."""
    num_params = sum(p.numel() for p in params)
    v = torch.randn(num_params, device=device)
    v = v / torch.norm(v)
    
    for _ in range(num_iters):
        Hv = compute_hvp(loss, params, v)
        v = Hv / torch.norm(Hv)
        
    lambda_max = torch.sum(v * compute_hvp(loss, params, v)).item()
    return lambda_max

def hutchinson_trace(loss, params, num_samples=10):
    """Estimates the Hessian Trace."""
    num_params = sum(p.numel() for p in params)
    trace_est = 0.0
    
    for _ in range(num_samples):
        # Rademacher distribution {-1, 1}
        v = torch.randint(0, 2, (num_params,), device=device).float() * 2 - 1
        Hv = compute_hvp(loss, params, v)
        trace_est += torch.sum(v * Hv).item()
        
    return trace_est / num_samples

print("Scanning checkpoints...")

checkpoints = [
    f for f in os.listdir(out_dir)
    if f.startswith('ckpt_')
    and f.endswith('.pt')
    and int(f.split('_')[1].split('.')[0]) <= 4000
]

# Sort chronologically by checkpoint step
checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

# Prepare a batch of data to use for all Hessian calculations
X, Y = get_eval_batch()

results = []

for ckpt_name in checkpoints:
    step = int(ckpt_name.split('_')[1].split('.')[0])
    ckpt_path = os.path.join(out_dir, ckpt_name)
    
    # Load model
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except Exception:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**checkpoint['model_args']))
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    params = get_params(model)
    
    # Forward pass
    logits, loss = model(X, Y)
    
    # Compute Geometric metrics
    print(f"Analyzing Step {step}...")
    lambda_max = power_iteration(loss, params, num_iters=15)
    trace = hutchinson_trace(loss, params, num_samples=5) # Keep samples low for speed
    
    print(f"--> Step {step} | Loss: {loss.item():.4f} | Lambda Max: {lambda_max:.2f} | Trace: {trace:.2f}")
    
    results.append((step, loss.item(), lambda_max, trace))

# Save results for plotting
np.save(os.path.join(out_dir, 'hessian_metrics.npy'), np.array(results))
print("Analysis complete. Saved to hessian_metrics.npy")