import torch, os, numpy as np
from model import GPTConfig, GPT

def verify_run(out_dir, device):
    ckpts = [f for f in os.listdir(out_dir) if f.startswith('ckpt_')]
    sorted_ckpts = sorted(ckpts, key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    for label, ckpt_name in [('first', sorted_ckpts[0]), ('last', sorted_ckpts[-1])]:
        step = int(ckpt_name.split('_')[1].split('.')[0])
        ck = torch.load(os.path.join(out_dir, ckpt_name), map_location='cpu', weights_only=False)
        model = GPT(GPTConfig(**ck['model_args']))
        model.load_state_dict(ck['model'])
        model.to(device)
        model.eval()
        data_dir = os.path.join('data', ck.get('dataset_name'))
        bs = ck['model_args']['block_size']
        nout = ck.get('n_output', 1)
        for split in ['train', 'val']:
            data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
            eq_len = bs + 1
            n_eq = len(data) // eq_len
            losses = []
            for _ in range(10):
                ix_eq = np.random.randint(0, n_eq, (512,))
                ix = ix_eq * eq_len
                x = np.stack([data[i:i+bs].astype(np.int64) for i in ix])
                y = np.stack([data[i+1:i+1+bs].astype(np.int64) for i in ix])
                y[:, :-nout] = -1
                xb = torch.from_numpy(x).to(device)
                yb = torch.from_numpy(y).to(device)
                with torch.no_grad():
                    _, loss = model(xb, yb)
                losses.append(loss.item())
            print(f'  {label}({step}) {split}: {np.mean(losses):.6f}')
    
    print('  Val loss:')
    for ckpt_name in sorted_ckpts[::5]:
        step = int(ckpt_name.split('_')[1].split('.')[0])
        ck = torch.load(os.path.join(out_dir, ckpt_name), map_location='cpu', weights_only=False)
        model = GPT(GPTConfig(**ck['model_args']))
        model.load_state_dict(ck['model'])
        model.to(device)
        model.eval()
        data_dir = os.path.join('data', ck.get('dataset_name'))
        bs = ck['model_args']['block_size']
        nout = ck.get('n_output', 1)
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        eq_len = bs + 1
        n_eq = len(data) // eq_len
        losses = []
        for _ in range(5):
            ix_eq = np.random.randint(0, n_eq, (512,))
            ix = ix_eq * eq_len
            x = np.stack([data[i:i+bs].astype(np.int64) for i in ix])
            y = np.stack([data[i+1:i+1+bs].astype(np.int64) for i in ix])
            y[:, :-nout] = -1
            xb = torch.from_numpy(x).to(device)
            yb = torch.from_numpy(y).to(device)
            with torch.no_grad():
                _, loss = model(xb, yb)
            losses.append(loss.item())
        print(f'    step {step}: val={np.mean(losses):.4f}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

for out_dir, label in [('out-check-add', 'addition'), ('out-check-sub', 'subtraction')]:
    print(f'\n=== modular_{label} ===')
    if os.path.exists(out_dir):
        verify_run(out_dir, device)
    else:
        print(f'  {out_dir} not found')
