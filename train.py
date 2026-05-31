import os
import time
import math
import json
import random
import platform
import sys
import argparse
import torch
import numpy as np
from contextlib import nullcontext
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# GROKKING HYPERPARAMETERS
# -----------------------------------------------------------------------------
# I/O
out_dir = 'out-grokking'
eval_interval = 100       # Frequent evaluation to catch the phase transition
eval_iters = 50           # Number of batches to estimate validation loss
always_save_checkpoint = True

# Data configuration
dataset = 'modular_addition'
batch_size = 512          # Grokking requires massive batches (often full-batch)
block_size = 4            # Length of input sequence (e.g., [a, +, b, =])
vocab_size = 99           # Assuming p=97 (0-96), plus '+' (97), plus '=' (98)

# Model configuration (Tiny Transformer)
n_layer = 2
n_head = 4
n_embd = 128
dropout = 0.0             # Grokking is cleanest with zero dropout
bias = False              # Do we use bias inside LayerNorm and Linear layers?

# AdamW Optimizer (Heavily tuned for Grokking)
learning_rate = 1e-3      # Fast learning rate
max_iters = 20000        # Grokking requires escaping a long plateau
weight_decay = 0.5      # CRITICAL: 1.0+ forces the transition to the "rich" regime
beta1 = 0.9
beta2 = 0.98              # Standard beta2 for grokking literature
grad_clip = 1.0 

# Learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = max_iters
min_lr = 1e-4

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# default dtype choice; user can force different behavior via flags later
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = False     # Set to False if PyTorch 2.0+ torch.compile gives you errors

# Command-line args for reproducibility and modes
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--deterministic', action='store_true', help='enable deterministic PyTorch ops')
parser.add_argument('--disable-fused', action='store_true', help='disable fused AdamW for reproducibility')
parser.add_argument('--full-batch', action='store_true', help='use full-batch training (batch_size = entire train set)')
parser.add_argument('--max-iters', type=int, default=None, help='override max_iters for quick tests')
parser.add_argument('--eval-iters', type=int, default=None, help='override eval_iters for quick tests')
parser.add_argument('--eval-interval', type=int, default=None, help='override eval_interval for quick tests')
args = parser.parse_args()
# -----------------------------------------------------------------------------

# Setup system context
os.makedirs(out_dir, exist_ok=True)
# Seed everything for reproducibility
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
if args.deterministic:
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        print('Warning: unable to enable fully deterministic algorithms on this build')
    torch.backends.cudnn.benchmark = False

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Apply CLI overrides for quick tests
if args.max_iters is not None:
    max_iters = int(args.max_iters)
    print(f"Overriding max_iters -> {max_iters}")
if args.eval_iters is not None:
    eval_iters = int(args.eval_iters)
    print(f"Overriding eval_iters -> {eval_iters}")
if args.eval_interval is not None:
    eval_interval = int(args.eval_interval)
    print(f"Overriding eval_interval -> {eval_interval}")

# -----------------------------------------------------------------------------
# DATA LOADER (Strict Equation Alignment)
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)

# If requested, set batch_size to full training set size (full-batch)
if args.full_batch:
    train_mem = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    eq_length_tmp = block_size + 1
    num_equations_tmp = len(train_mem) // eq_length_tmp
    batch_size = int(num_equations_tmp)
    print(f"Using full-batch training: setting batch_size = {batch_size}")

def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # Force exact equation boundaries
    eq_length = block_size + 1 
    num_equations = len(data) // eq_length
    ix_eq = torch.randint(0, num_equations, (batch_size,))
    ix = ix_eq * eq_length 
    
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    

    y[:, :-1] = -1
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# -----------------------------------------------------------------------------
# MODEL INITIALIZATION
# -----------------------------------------------------------------------------
print("Initializing a new model from scratch...")
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)

gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# allow forcing fused AdamW off for reproducibility
if args.disable_fused:
    optimizer, fused_used = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, use_fused_override=False)
else:
    # preserve default detection when not explicitly disabled
    optimizer, fused_used = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, use_fused_override=None)

# write run metadata for reproducibility
run_metadata = {
    'seed': args.seed,
    'deterministic': bool(args.deterministic),
    'disable_fused': bool(args.disable_fused),
    'full_batch': bool(args.full_batch),
    'torch_version': torch.__version__,
    'cuda_version': torch.version.cuda,
    'device': device,
    'dtype': dtype,
    'fused_adamw': bool(fused_used),
    'platform': platform.platform(),
    'python_version': sys.version.replace('\n', ' '),
}
with open(os.path.join(out_dir, 'run_metadata.json'), 'w') as f:
    json.dump(run_metadata, f, indent=2)

if compile_model:
    print("Compiling the model... (takes a ~minute)")
    model = torch.compile(model) 

# -----------------------------------------------------------------------------
# TRAINING HELPERS
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (learning_rate - min_lr)

# -----------------------------------------------------------------------------
# MAIN TRAINING LOOP
# -----------------------------------------------------------------------------
print("Starting training loop...")
X, Y = get_batch('train')
t0 = time.time()
iter_num = 0
best_val_loss = 1e9

while True:
    # Set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate and Save Checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'rng_states': {
                        'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        'numpy': np.random.get_state(),
                        'python': random.getstate(),
                    },
                    'run_metadata': run_metadata,
            }
            # Append iter_num so we have a HISTORY for Optimal Transport / Hessian tracking
            ckpt_filename = f'ckpt_{iter_num}.pt'
            print(f"saving checkpoint to {out_dir}/{ckpt_filename}")
            torch.save(checkpoint, os.path.join(out_dir, ckpt_filename))

    # Termination condition
    if iter_num > max_iters:
        break

    # Forward, Backward, Update
    with ctx:
        logits, loss = model(X, Y)
    
    # Async prefetch next batch
    X, Y = get_batch('train')
    
    scaler.scale(loss).backward()
    
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % 10 == 0:
        lossf = loss.item()
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        
    iter_num += 1

print("Training completed.")