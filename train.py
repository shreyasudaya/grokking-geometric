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
# DATASET CONFIGURATIONS
# -----------------------------------------------------------------------------
DATASET_CONFIGS = {
    'modular_addition': {
        'vocab_size': 99,
        'block_size': 4,
        'n_output': 1,
    },
    'modular_subtraction': {
        'vocab_size': 99,
        'block_size': 4,
        'n_output': 1,
    },
    'modular_multiplication': {
        'vocab_size': 99,
        'block_size': 4,
        'n_output': 1,
    },
    'symmetric_group': {
        'vocab_size': 7,
        'block_size': 16,
        'n_output': 5,
    },
    'permutation_composition': {
        'vocab_size': 8,
        'block_size': 19,
        'n_output': 6,
    },
}

# -----------------------------------------------------------------------------
# GROKKING HYPERPARAMETERS
# -----------------------------------------------------------------------------
out_dir = 'out-grokking'
eval_interval = 100
eval_iters = 50
always_save_checkpoint = True

# Data configuration (defaults, can be overridden by --dataset)
dataset = 'modular_addition'
batch_size = 512
block_size = 4
vocab_size = 99
n_output = 1

# Model configuration
n_layer = 2
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

# AdamW Optimizer
learning_rate = 1e-3
max_iters = 50000
weight_decay = 1.0
beta1 = 0.9
beta2 = 0.98
grad_clip = 1.0

# Learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = max_iters
min_lr = 1e-4

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = False

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='modular_addition', choices=list(DATASET_CONFIGS.keys()), help='dataset / operation to train on')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--deterministic', action='store_true', help='enable deterministic PyTorch ops')
parser.add_argument('--disable-fused', action='store_true', help='disable fused AdamW for reproducibility')
parser.add_argument('--full-batch', action='store_true', help='use full-batch training')
parser.add_argument('--max-iters', type=int, default=None, help='override max_iters')
parser.add_argument('--eval-iters', type=int, default=None, help='override eval_iters')
parser.add_argument('--eval-interval', type=int, default=None, help='override eval_interval')
parser.add_argument('--save-every', type=int, default=None, help='save checkpoint every N steps')
parser.add_argument('--n-layer', type=int, default=None, help='override n_layer')
parser.add_argument('--n-head', type=int, default=None, help='override n_head')
parser.add_argument('--n-embd', type=int, default=None, help='override n_embd')
parser.add_argument('--out-dir', type=str, default=None, help='override output directory')
parser.add_argument('--weight-decay', type=float, default=None, help='override weight_decay')
parser.add_argument('--learning-rate', type=float, default=None, help='override learning_rate')
parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint in out_dir')
args = parser.parse_args()

# Apply dataset config
dc = DATASET_CONFIGS[args.dataset]
dataset = args.dataset
vocab_size = dc['vocab_size']
block_size = dc['block_size']
n_output = dc['n_output']

# Apply model arch overrides
if args.n_layer is not None:
    n_layer = int(args.n_layer)
if args.n_head is not None:
    n_head = int(args.n_head)
if args.n_embd is not None:
    n_embd = int(args.n_embd)

if args.out_dir is not None:
    out_dir = args.out_dir

# Recompute lr_decay_iters in case max_iters changed
lr_decay_iters = max_iters

# Setup output directory
os.makedirs(out_dir, exist_ok=True)

# Seed everything
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

# Apply CLI overrides
if args.max_iters is not None:
    max_iters = int(args.max_iters)
    lr_decay_iters = max_iters
if args.eval_iters is not None:
    eval_iters = int(args.eval_iters)
if args.eval_interval is not None:
    eval_interval = int(args.eval_interval)
if args.save_every is not None:
    save_every = int(args.save_every)
else:
    save_every = eval_interval
if args.weight_decay is not None:
    weight_decay = float(args.weight_decay)
if args.learning_rate is not None:
    learning_rate = float(args.learning_rate)

# -----------------------------------------------------------------------------
# DATA LOADER
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)

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

    eq_length = block_size + 1
    num_equations = len(data) // eq_length
    ix_eq = torch.randint(0, num_equations, (batch_size,))
    ix = ix_eq * eq_length

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    # Mask all positions except the last n_output tokens
    y[:, :-n_output] = -1

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# MODEL INITIALIZATION
# -----------------------------------------------------------------------------
print(f"Initializing model for dataset '{dataset}'...")
print(f"  vocab_size={vocab_size}, block_size={block_size}, n_output={n_output}")
print(f"  n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)

gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
if args.disable_fused:
    optimizer, fused_used = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, use_fused_override=False)
else:
    optimizer, fused_used = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, use_fused_override=None)

run_metadata = {
    'dataset': dataset,
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
# RESUME
# -----------------------------------------------------------------------------
if args.resume:
    ckpt_files = [f for f in os.listdir(out_dir) if f.startswith('ckpt_') and f.endswith('.pt')]
    if not ckpt_files:
        print("No checkpoints found, starting from scratch.")
    else:
        latest_ckpt = max(ckpt_files, key=lambda f: int(f.replace('ckpt_', '').replace('.pt', '')))
        ckpt_path = os.path.join(out_dir, latest_ckpt)
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        try:
            torch.set_rng_state(ckpt['rng_states']['torch'].cpu())
            if torch.cuda.is_available() and ckpt['rng_states']['cuda'] is not None:
                torch.cuda.set_rng_state_all(ckpt['rng_states']['cuda'])
            np.random.set_state(ckpt['rng_states']['numpy'])
            random.setstate(ckpt['rng_states']['python'])
        except Exception as e:
            print(f"Warning: could not restore RNG states ({e}), using fresh seeds")
        iter_num = ckpt['iter_num'] + 1
        print(f"Resumed at iteration {iter_num}")

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
print(f"Starting training loop for {max_iters} iterations...")
X, Y = get_batch('train')
t0 = time.time()
iter_num = 0
best_val_loss = 1e9

def save_checkpoint(iter_num, val_loss=None):
    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'dataset_name': dataset,
        'n_output': n_output,
        'iter_num': iter_num,
        'val_loss': val_loss,
        'run_metadata': run_metadata,
        'rng_states': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        },
    }
    ckpt_filename = f'ckpt_{iter_num}.pt'
    torch.save(ckpt, os.path.join(out_dir, ckpt_filename))

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        save_checkpoint(iter_num, val_loss=losses['val'])
    elif iter_num % save_every == 0:
        save_checkpoint(iter_num)

    if iter_num > max_iters:
        break

    with ctx:
        logits, loss = model(X, Y)

    X, Y = get_batch('train')

    scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % 10 == 0:
        lossf = loss.item()
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    iter_num += 1

print("Training completed.")

