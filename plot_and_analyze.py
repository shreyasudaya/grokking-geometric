import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import GPTConfig, GPT

out_dir = 'out-grokking'
data_dir = os.path.join('data', 'modular_addition')
block_size = 4
vocab_size = 99

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ptdtype = torch.float32

def load_checkpoints(out_dir):
    cks = [f for f in os.listdir(out_dir) if f.startswith('ckpt_') and f.endswith('.pt')]
    def step_of(fn):
        return int(fn.split('_')[1].split('.')[0])
    cks.sort(key=step_of)
    return [os.path.join(out_dir, f) for f in cks]

def sample_batch_from_memmap(split, batch_size):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    eq_length = block_size + 1
    num_equations = len(data) // eq_length
    ix_eq = np.random.randint(0, num_equations, (batch_size,))
    ix = ix_eq * eq_length
    x = np.stack([data[i:i+block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i+1:i+1+block_size].astype(np.int64) for i in ix])
    y[:, :-1] = -1
    return torch.from_numpy(x), torch.from_numpy(y)

def compute_losses_for_checkpoint(ckpt_path, eval_iters=10, batch_size=256):
    try:
        ck = torch.load(ckpt_path, map_location='cpu')
    except Exception:
        # Retry allowing pickled objects when PyTorch enforces weights_only by default
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = GPT(GPTConfig(**ck['model_args']))
    model.load_state_dict(ck['model'])
    model.to(device)
    model.eval()

    train_losses = []
    val_losses = []
    for _ in range(eval_iters):
        xb, yb = sample_batch_from_memmap('train', batch_size)
        xb = xb.to(device)
        yb = yb.to(device)
        with torch.no_grad():
            _, loss = model(xb, yb)
        train_losses.append(loss.item())

        xb, yb = sample_batch_from_memmap('val', batch_size)
        xb = xb.to(device)
        yb = yb.to(device)
        with torch.no_grad():
            _, loss = model(xb, yb)
        val_losses.append(loss.item())

    return np.mean(train_losses), np.mean(val_losses)

def detect_phase_transition(steps, val_losses):
    # find largest negative drop in validation loss between successive checkpoints
    deltas = np.diff(val_losses)
    idx = np.argmin(deltas)
    return steps[idx+1], deltas[idx]

def main():
    ckpts = load_checkpoints(out_dir)
    steps = []
    train_ls = []
    val_ls = []
    for ck in ckpts:
        step = int(os.path.basename(ck).split('_')[1].split('.')[0])
        print(f"Evaluating checkpoint {ck} ...")
        t_loss, v_loss = compute_losses_for_checkpoint(ck, eval_iters=5, batch_size=256)
        steps.append(step)
        train_ls.append(t_loss)
        val_ls.append(v_loss)

    steps = np.array(steps)
    train_ls = np.array(train_ls)
    val_ls = np.array(val_ls)

    # save history
    np.save(os.path.join(out_dir, 'loss_history.npy'), np.vstack([steps, train_ls, val_ls]))

    # detect phase transition
    pt_step, pt_delta = detect_phase_transition(steps, val_ls)
    print(f"Detected phase transition at step {pt_step} with val loss delta {pt_delta:.4f}")

    # plot
    plt.figure(figsize=(8,5))
    plt.plot(steps, train_ls, label='train loss')
    plt.plot(steps, val_ls, label='val loss')
    plt.axvline(pt_step, color='k', linestyle='--', label=f'phase transition @ {pt_step}')
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('loss (log)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_plot.png'))

    # separate phase-transition plot (zoom)
    plt.figure(figsize=(8,4))
    plt.plot(steps, val_ls, label='val loss')
    plt.axvline(pt_step, color='k', linestyle='--', label=f'phase transition @ {pt_step}')
    plt.xlabel('step')
    plt.ylabel('val loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'phase_transition.png'))

    print('Plots saved to out-grokking/')

    # run geometric analysis script (analyze.py)
    print('Running geometric analysis via analyze.py ...')
    os.system(f'{sys.executable} analyze.py')

if __name__ == '__main__':
    import sys
    main()
