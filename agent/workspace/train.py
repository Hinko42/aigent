import sys, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "/bench")
from score import load_train, evaluate, VOCAB_SIZE

import os; torch.manual_seed(int(os.environ.get("SEED", "0")))
torch.set_num_threads(3)
BLOCK, D, HEADS, LAYERS, BATCH, MAX_STEPS = 128, 128, 4, 3, 32, 1000

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, D)
        self.pos = nn.Embedding(BLOCK, D)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB_SIZE)
    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1)))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

data = load_train()
model = Model()
opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
params = sum(p.numel() for p in model.parameters())
start, step = time.time(), 0
model.train()
while step < MAX_STEPS:
    ix = torch.randint(0, len(data) - BLOCK - 1, (BATCH,))
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK + 1] for i in ix])
    loss = F.cross_entropy(model(x).view(-1, VOCAB_SIZE), y.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    step += 1
    if step % 100 == 0:
        print(step, round(loss.item(), 3), flush=True)

bpc = evaluate(model, BLOCK)
result = {"val_bits_per_char": round(bpc, 4), "steps": step, "params": params}
print(result)
json.dump(result, open("/work/result.json", "w"))
