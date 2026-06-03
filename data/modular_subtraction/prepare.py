import os
import numpy as np

os.makedirs('data/modular_subtraction', exist_ok=True)

p = 97

data = []
for a in range(p):
    for b in range(p):
        c = (a - b) % p
        data.append([a, p, b, p+1, c])

data = np.array(data)
np.random.shuffle(data)

split_idx = int(len(data) * 0.25)
train_data = data[:split_idx]
val_data = data[split_idx:]

train_data.astype(np.uint16).tofile('data/modular_subtraction/train.bin')
val_data.astype(np.uint16).tofile('data/modular_subtraction/val.bin')

print(f"Data generation complete!")
print(f"Train size: {len(train_data)} equations")
print(f"Val size: {len(val_data)} equations")
print(f"Vocab size needed: {p + 2}")
