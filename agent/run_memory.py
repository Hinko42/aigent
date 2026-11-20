import json, os, subprocess

WORK = r"C:\agent\workspace"
BENCH = r"C:\agent\benchmark"
CONDS = ["none", "recent", "random", "surprise"]

def run(cond, seed):
    out = os.path.join(WORK, "mem_%s_%d.json" % (cond, seed))
    if not os.path.exists(out):
        cmd = ["docker", "run", "--rm", "--name", "agent-mem", "--network", "none",
               "--memory", "3g", "--cpus", "3", "-e", "SEED=%d" % seed, "-e", "COND=" + cond,
               "-v", WORK + ":/work", "-v", BENCH + ":/bench:ro",
               "agent-sandbox", "python", "memory_exp.py"]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", "agent-mem"], capture_output=True)
    if os.path.exists(out):
        return json.load(open(out))
    print("failed:", cond, seed)
    return None

results = {c: {} for c in CONDS}
for seed in (0, 1, 2):
    for cond in CONDS:
        r = run(cond, seed)
        if r:
            results[cond][seed] = r
            print(cond, seed, "bpc", r["bpc"], "nomem", r["bpc_nomem"], "gate", r["gate"], flush=True)

def diff(a, b):
    seeds = [s for s in (0, 1, 2) if s in results[a] and s in results[b]]
    if not seeds:
        return None
    d = [results[a][s]["bpc"] - results[b][s]["bpc"] for s in seeds]
    return sum(d) / len(d), sum(1 for v in d if v < 0), len(d)

print("\nMEANS")
for c in CONDS:
    rows = list(results[c].values())
    if rows:
        n = len(rows)
        print("%-9s bpc %.4f  nomem %.4f  gate %.3f" % (c, sum(r["bpc"] for r in rows) / n,
              sum(r["bpc_nomem"] for r in rows) / n, sum(r["gate"] for r in rows) / n))

print("\nDIFFERENCES (negative = first is better)")
for a, b in [("recent", "none"), ("random", "none"), ("surprise", "none"),
             ("surprise", "recent"), ("surprise", "random")]:
    d = diff(a, b)
    if d:
        print("%-9s vs %-9s mean %+.4f   first better on %d of %d seeds" % (a, b, d[0], d[1], d[2]))
