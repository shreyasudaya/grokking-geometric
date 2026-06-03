import os
import numpy as np
from itertools import permutations

os.makedirs('data/symmetric_group', exist_ok=True)

n = 5
elements = list(permutations(range(n)))  # 120 elements of S_5
op_token = n         # 5
eq_token = n + 1     # 6

data = []
for p in elements:
    for q in elements:
        r = tuple(p[q[i]] for i in range(n))
        seq = list(p) + [op_token] + list(q) + [eq_token] + list(r)
        data.append(seq)

data = np.array(data, dtype=np.uint16)
np.random.shuffle(data)

split_idx = int(len(data) * 0.25)
train_data = data[:split_idx]
val_data = data[split_idx:]

train_data.tofile('data/symmetric_group/train.bin')
val_data.tofile('data/symmetric_group/val.bin')

print(f"Data generation complete!")
print(f"Train size: {len(train_data)} equations")
print(f"Val size: {len(val_data)} equations")
print(f"Vocab size needed: {n + 2}")
