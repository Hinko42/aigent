import sys, os, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, "/bench")
from score import load_train, VOCAB_SIZE, _data, _n

SEED = int(os.environ.get("SEED", "0"))
COND = os.environ.get("COND", "none")
M = int(os.environ.get("M", "256"))
DECAY = float(os.environ.get("DECAY", "0.97"))
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
        self.xattn = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.xgate = nn.Parameter(torch.zeros(1))
    def features(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1)))
        for b in self.blocks:
            x = b(x)
        return self.ln(x)
    def read(self, hn, mem):
        if mem.size(1) == 0:
            return hn
        a, _ = self.xattn(hn, mem, mem, need_weights=False)
        return hn + self.xgate * a

def update(mh, ms, last, hn, per, step):
    if COND == "none":
        return mh, ms, last
    B, T = per.shape
    if COND == "surprise":
        sc = torch.cat([last.unsqueeze(1), per[:, :-1]], dim=1)
        ms = ms * DECAY
    elif COND == "recent":
        sc = (step * T + torch.arange(T, dtype=torch.float32)).unsqueeze(0).expand(B, T)
    else:
        sc = torch.rand(B, T)
    h_all = torch.cat([mh, hn.detach()], dim=1)
    s_all = torch.cat([ms, sc], dim=1)
    k = min(M, h_all.size(1))
    top = s_all.topk(k, dim=1).indices
    mh = torch.gather(h_all, 1, top.unsqueeze(-1).expand(-1, -1, D))
    ms = torch.gather(s_all, 1, top)
    return mh, ms, per[:, -1]

data = load_train()
model = Model()
opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
S = len(data) // BATCH
streams = torch.stack([data[i * S:(i + 1) * S] for i in range(BATCH)])
n_seg = (S - 1) // BLOCK
mh, ms, last = torch.zeros(BATCH, 0, D), torch.zeros(BATCH, 0), torch.zeros(BATCH)

model.train()
for step in range(MAX_STEPS):
    seg = step % n_seg
    if seg == 0:
        mh, ms, last = torch.zeros(BATCH, 0, D), torch.zeros(BATCH, 0), torch.zeros(BATCH)
    s = seg * BLOCK
    x = streams[:, s:s + BLOCK]
    y = streams[:, s + 1:s + BLOCK + 1]
    hn = model.features(x)
    logits = model.head(model.read(hn, mh))
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
    lr_now = BASE_LR * min(1.0, (step + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / MAX_STEPS)))
    for g in opt.param_groups:
        g["lr"] = lr_now
    opt.zero_grad()
    loss.backward()
    opt.step()
    if COND != "none":
        with torch.no_grad():
            per = F.cross_entropy(logits.detach().reshape(-1, VOCAB_SIZE), y.reshape(-1), reduction="none").view(BATCH, BLOCK)
            mh, ms, last = update(mh, ms, last, hn, per, step)
    if (step + 1) % 100 == 0:
        print(step + 1, round(loss.item(), 3), round(model.xgate.item(), 3), flush=True)

def run_eval(use_mem):
    val = _data[_n:]
    NS = 8
    SV = len(val) // NS
    vs = torch.stack([val[i * SV:(i + 1) * SV] for i in range(NS)])
    nseg = (SV - 1) // BLOCK
    emh, ems, elast = torch.zeros(NS, 0, D), torch.zeros(NS, 0), torch.zeros(NS)
    total, count = 0.0, 0
    with torch.no_grad():
        for sg in range(nseg):
            x = vs[:, sg * BLOCK:(sg + 1) * BLOCK]
            y = vs[:, sg * BLOCK + 1:(sg + 1) * BLOCK + 1]
            hn = model.features(x)
            out = model.read(hn, emh) if use_mem else hn
            logits = model.head(out)
            per = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), reduction="none").view(NS, BLOCK)
            total += per.sum().item()
            count += per.numel()
            if use_mem:
                emh, ems, elast = update(emh, ems, elast, hn, per, sg)
    return total / count / math.log(2)

model.eval()
res = {"cond": COND, "seed": SEED, "M": M,
       "bpc": round(run_eval(True), 4), "bpc_nomem": round(run_eval(False), 4),
       "gate": round(model.xgate.item(), 4)}
print(res)
json.dump(res, open("/work/mem_%s_%d.json" % (COND, SEED), "w"))
