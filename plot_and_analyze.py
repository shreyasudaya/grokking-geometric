import os
import math
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import GPTConfig, GPT

parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', type=str, default='out-grokking', help='checkpoint directory')
parser.add_argument('--eval-iters', type=int, default=5, help='evaluation batches')
parser.add_argument('--batch-size', type=int, default=256, help='evaluation batch size')
args = parser.parse_args()

out_dir = args.out_dir
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_checkpoints(out_dir):
    cks = [f for f in os.listdir(out_dir) if f.startswith('ckpt_') and f.endswith('.pt')]
    def step_of(fn):
        return int(fn.split('_')[1].split('.')[0])
    cks.sort(key=step_of)
    return [os.path.join(out_dir, f) for f in cks]

def get_data_config(ckpt_path):
    try:
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception:
        ck = torch.load(ckpt_path, map_location='cpu')
    block_size = ck['model_args']['block_size']
    n_output = ck.get('n_output', 1)
    dataset_name = ck.get('dataset_name', 'modular_addition')
    data_dir = os.path.join('data', dataset_name)
    return data_dir, block_size, n_output

def sample_batch_from_memmap(split, batch_size, data_dir, block_size, n_output):
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
    y[:, :-n_output] = -1
    return torch.from_numpy(x), torch.from_numpy(y)

def compute_losses_for_checkpoint(ckpt_path, eval_iters, batch_size, data_dir, block_size, n_output):
    try:
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception:
        ck = torch.load(ckpt_path, map_location='cpu')
    model = GPT(GPTConfig(**ck['model_args']))
    model.load_state_dict(ck['model'])
    model.to(device)
    model.eval()

    train_losses = []
    val_losses = []
    for _ in range(eval_iters):
        xb, yb = sample_batch_from_memmap('train', batch_size, data_dir, block_size, n_output)
        xb = xb.to(device)
        yb = yb.to(device)
        with torch.no_grad():
            _, loss = model(xb, yb)
        train_losses.append(loss.item())

        xb, yb = sample_batch_from_memmap('val', batch_size, data_dir, block_size, n_output)
        xb = xb.to(device)
        yb = yb.to(device)
        with torch.no_grad():
            _, loss = model(xb, yb)
        val_losses.append(loss.item())

    return np.mean(train_losses), np.mean(val_losses)

def detect_phase_transition(steps, val_losses):
    deltas = np.diff(val_losses)
    idx = np.argmin(deltas)
    return steps[idx+1], deltas[idx]

def main():
    ckpts = load_checkpoints(out_dir)
    if len(ckpts) == 0:
        print(f"No checkpoints found in {out_dir}")
        return

    data_dir, block_size, n_output = get_data_config(ckpts[0])
    print(f"Dataset config: data_dir={data_dir}, block_size={block_size}, n_output={n_output}")

    steps = []
    train_ls = []
    val_ls = []
    for ck in ckpts:
        step = int(os.path.basename(ck).split('_')[1].split('.')[0])
        print(f"Evaluating checkpoint {ck} ...")
        t_loss, v_loss = compute_losses_for_checkpoint(ck, args.eval_iters, args.batch_size, data_dir, block_size, n_output)
        steps.append(step)
        train_ls.append(t_loss)
        val_ls.append(v_loss)

    steps = np.array(steps)
    train_ls = np.array(train_ls)
    val_ls = np.array(val_ls)

    np.save(os.path.join(out_dir, 'loss_history.npy'), np.vstack([steps, train_ls, val_ls]))

    pt_step, pt_delta = detect_phase_transition(steps, val_ls)
    print(f"Detected phase transition at step {pt_step} with val loss delta {pt_delta:.4f}")

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

    plt.figure(figsize=(8,4))
    plt.plot(steps, val_ls, label='val loss')
    plt.axvline(pt_step, color='k', linestyle='--', label=f'phase transition @ {pt_step}')
    plt.xlabel('step')
    plt.ylabel('val loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'phase_transition.png'))

    print('Plots saved to', out_dir)

    print('Running geometric analysis via analyze.py ...')
    os.system(f'{sys.executable} analyze.py --out-dir {out_dir}')

if __name__ == '__main__':
    main()
