import json, os, subprocess, sys

WORK = r"C:\agent\workspace"
BENCH = r"C:\agent\benchmark"
BASE = {0: 2.4200, 1: 2.4004, 2: 2.3823}

LR_LINE = "opt = torch.optim.AdamW(model.parameters(), lr=5e-3)"
WARMUP_COSINE = [
    (LR_LINE, LR_LINE + "\nimport math\nBASE_LR = 5e-3\nWARMUP = 100"),
    ("    opt.zero_grad()",
     "    lr_now = BASE_LR * min(1.0, (step + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / MAX_STEPS)))\n"
     "    for g in opt.param_groups:\n        g[\"lr\"] = lr_now\n    opt.zero_grad()"),
]
CLIP = [
    ("    opt.step()", "    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n    opt.step()"),
]
WARMUP_8E3 = [(o, n.replace("BASE_LR = 5e-3", "BASE_LR = 8e-3")) for o, n in WARMUP_COSINE]
IDEAS = {"peak8": WARMUP_8E3, "warmup_cosine": WARMUP_COSINE, "clip": CLIP, "both": WARMUP_COSINE + CLIP}

def run(seed):
    res = os.path.join(WORK, "result.json")
    if os.path.exists(res):
        os.remove(res)
    cmd = ["docker", "run", "--rm", "--network", "none", "--memory", "3g", "--cpus", "3",
           "-e", "SEED=%d" % seed, "-v", WORK + ":/work", "-v", BENCH + ":/bench:ro",
           "agent-sandbox", "python", "train_candidate.py"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if not os.path.exists(res):
        print("run failed:", (p.stderr or p.stdout)[-400:])
        return None
    return json.load(open(res))["val_bits_per_char"]

name = sys.argv[1]
text = open(os.path.join(WORK, "train.py"), encoding="utf-8").read()
for old, new in IDEAS[name]:
    assert text.count(old) == 1, "pattern not found exactly once: " + old
    text = text.replace(old, new)
open(os.path.join(WORK, "train_candidate.py"), "w", encoding="utf-8", newline="\n").write(text)

scores = {}
for seed in (0, 1, 2):
    scores[seed] = run(seed)
    print("seed", seed, scores[seed], "(baseline %.4f)" % BASE[seed], flush=True)
if all(v is not None for v in scores.values()):
    mean = sum(scores.values()) / 3
    base = sum(BASE.values()) / 3
    print("%s: mean %.4f vs baseline %.4f -> gain %.4f" % (name, mean, base, base - mean))
