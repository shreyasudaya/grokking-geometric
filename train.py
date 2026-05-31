import os
import time
import math
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
weight_decay = 1.0        # CRITICAL: 1.0+ forces the transition to the "rich" regime
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
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = False     # Set to False if PyTorch 2.0+ torch.compile gives you errors
# -----------------------------------------------------------------------------

# Setup system context
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# DATA LOADER (Strict Equation Alignment)
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)

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
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

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