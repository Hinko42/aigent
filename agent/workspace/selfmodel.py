import sys, os, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "/bench")
from score import load_train, evaluate, VOCAB_SIZE, _data, _n

SEED = int(os.environ.get("SEED", "0"))
COND = os.environ.get("COND", "none")
LAMBDA = float(os.environ.get("LAMBDA", "0.1"))
torch.manual_seed(SEED)
torch.set_num_threads(3)
BLOCK, D, HEADS, LAYERS, BATCH, MAX_STEPS = 128, 128, 4, 3, 32, 1000
BASE_LR, WARMUP = 8e-3, 100

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
    def hidden(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1)))
        for b in self.blocks:
            x = b(x)
        return self.ln(x)
    def forward(self, idx):
        return self.head(self.hidden(idx))

data = load_train()
model = Model()
state = torch.get_rng_state()
self_head = nn.Sequential(nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 1))
torch.set_rng_state(state)
opt = torch.optim.AdamW(list(model.parameters()) + list(self_head.parameters()), lr=5e-3)

model.train()
for step in range(MAX_STEPS):
    ix = torch.randint(0, len(data) - BLOCK - 1, (BATCH,))
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK + 1] for i in ix])
    h = model.hidden(x)
    logits = model.head(h)
    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
    total = loss
    sloss = torch.tensor(0.0)
    if COND != "none":
        with torch.no_grad():
            tgt = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), reduction="none")
        hin = h.detach() if COND == "detached" else h
        pred = self_head(hin).view(-1)
        sloss = F.mse_loss(pred, tgt)
        total = loss + LAMBDA * sloss
    lr_now = BASE_LR * min(1.0, (step + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / MAX_STEPS)))
    for g in opt.param_groups:
        g["lr"] = lr_now
    opt.zero_grad()
    total.backward()
    opt.step()
    if (step + 1) % 100 == 0:
        print(step + 1, round(loss.item(), 3), round(sloss.item(), 3), flush=True)

bpc = evaluate(model, BLOCK)

def corr(a, b):
    a = a.double() - a.double().mean()
    b = b.double() - b.double().mean()
    return ((a * b).sum() / (a.norm() * b.norm())).item()

def auroc(score, wrong):
    n = len(score)
    pos = int(wrong.sum().item())
    neg = n - pos
    if pos == 0 or neg == 0:
        return None
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[torch.argsort(score)] = torch.arange(1, n + 1, dtype=torch.float64)
    return (ranks[wrong].sum().item() - pos * (pos + 1) / 2) / (pos * neg)

def r(v):
    return None if v is None else round(v, 4)

model.eval()
val = _data[_n:]
L, E, M, W, P = [], [], [], [], []
starts = list(range(0, len(val) - BLOCK - 1, BLOCK))
with torch.no_grad():
    for k in range(0, len(starts), 16):
        idxs = starts[k:k + 16]
        x = torch.stack([val[i:i + BLOCK] for i in idxs])
        y = torch.stack([val[i + 1:i + BLOCK + 1] for i in idxs])
        h = model.hidden(x)
        logits = model.head(h)
        lp = F.log_softmax(logits, dim=-1)
        per = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ent = -(lp.exp() * lp).sum(-1)
        mp = lp.exp().max(-1).values
        wrong = logits.argmax(-1) != y
        L.append(per.reshape(-1))
        E.append(ent.reshape(-1))
        M.append(mp.reshape(-1))
        W.append(wrong.reshape(-1))
        if COND != "none":
            P.append(self_head(h).reshape(-1))
L, E, M, W = torch.cat(L), torch.cat(E), torch.cat(M), torch.cat(W)
res = {"cond": COND, "seed": SEED, "bpc": round(bpc, 4),
       "corr_entropy": r(corr(E, L)), "auroc_entropy": r(auroc(E, W)),
       "auroc_maxprob": r(auroc(-M, W))}
if COND != "none":
    P = torch.cat(P)
    res["corr_head"] = r(corr(P, L))
    res["auroc_head"] = r(auroc(P, W))
print(res)
json.dump(res, open("/work/sm_%s_%d.json" % (COND, SEED), "w"))
