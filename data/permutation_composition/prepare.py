import os
import numpy as np
from itertools import permutations

os.makedirs('data/permutation_composition', exist_ok=True)

n = 6
elements = list(permutations(range(n)))  # 720 elements of S_6
op_token = n         # 6
eq_token = n + 1     # 7

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

train_data.tofile('data/permutation_composition/train.bin')
val_data.tofile('data/permutation_composition/val.bin')

print(f"Data generation complete!")
print(f"Train size: {len(train_data)} equations")
print(f"Val size: {len(val_data)} equations")
print(f"Vocab size needed: {n + 2}")
