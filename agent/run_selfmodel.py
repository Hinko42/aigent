import json, os, subprocess

WORK = r"C:\agent\workspace"
BENCH = r"C:\agent\benchmark"
CONDS = ["none", "detached", "attached"]
KEYS = ["bpc", "corr_entropy", "corr_head", "auroc_entropy", "auroc_maxprob", "auroc_head"]

def run(cond, seed):
    out = os.path.join(WORK, "sm_%s_%d.json" % (cond, seed))
    if not os.path.exists(out):
        cmd = ["docker", "run", "--rm", "--name", "agent-sm", "--network", "none",
               "--memory", "3g", "--cpus", "3", "-e", "SEED=%d" % seed, "-e", "COND=" + cond,
               "-v", WORK + ":/work", "-v", BENCH + ":/bench:ro",
               "agent-sandbox", "python", "selfmodel.py"]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", "agent-sm"], capture_output=True)
    if os.path.exists(out):
        return json.load(open(out))
    print("failed:", cond, seed)
    return None

results = {c: [] for c in CONDS}
for seed in (0, 1, 2):
    for cond in CONDS:
        res = run(cond, seed)
        if res:
            results[cond].append(res)
            print(cond, seed, {k: res.get(k) for k in KEYS}, flush=True)

print("\nMEAN OVER SEEDS")
print("%-9s" % "cond" + "".join("%15s" % k for k in KEYS))
for cond in CONDS:
    rows = results[cond]
    cells = []
    for k in KEYS:
        vals = [r[k] for r in rows if r.get(k) is not None]
        cells.append("%15s" % ("-" if len(vals) < len(rows) or not vals else "%.4f" % (sum(vals) / len(vals))))
    print("%-9s" % cond + "".join(cells))
