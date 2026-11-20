import json, os, subprocess, sys, time

WORK = r"C:\agent\workspace"
BENCH = r"C:\agent\benchmark"
LOGS = r"C:\agent\logs"
LOG = os.path.join(LOGS, "queue_log.jsonl")
STOP = os.path.join(LOGS, "STOP")

LR_LINE = "opt = torch.optim.AdamW(model.parameters(), lr=5e-3)"
BASE_EDITS = [
    (LR_LINE, LR_LINE + "\nimport math\nBASE_LR = 8e-3\nWARMUP = 100"),
    ("    opt.zero_grad()",
     "    lr_now = BASE_LR * min(1.0, (step + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / MAX_STEPS)))\n"
     "    for g in opt.param_groups:\n        g[\"lr\"] = lr_now\n    opt.zero_grad()"),
]
CANDS = {
    "peak12": [("BASE_LR = 8e-3", "BASE_LR = 1.2e-2")],
    "peak16": [("BASE_LR = 8e-3", "BASE_LR = 1.6e-2")],
    "warm50": [("WARMUP = 100", "WARMUP = 50")],
    "warm200": [("WARMUP = 100", "WARMUP = 200")],
    "final01": [("(0.1 + 0.9 * 0.5 *", "(0.01 + 0.99 * 0.5 *")],
    "final30": [("(0.1 + 0.9 * 0.5 *", "(0.3 + 0.7 * 0.5 *")],
    "wd01": [(LR_LINE, "opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.1)")],
    "dropout": [("nn.MultiheadAttention(D, HEADS, batch_first=True)",
                 "nn.MultiheadAttention(D, HEADS, dropout=0.1, batch_first=True)"),
                ("nn.Linear(4 * D, D))", "nn.Linear(4 * D, D), nn.Dropout(0.1))")],
    "deep": [("128, 128, 4, 3, 32, 1000", "128, 128, 4, 4, 32, 1000")],
    "wide": [("128, 128, 4, 3, 32, 1000", "128, 192, 4, 3, 32, 1000")],
}

def apply(text, edits):
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text

def run(seed):
    res = os.path.join(WORK, "result.json")
    if os.path.exists(res):
        os.remove(res)
    cmd = ["docker", "run", "--rm", "--name", "agent-queue", "--network", "none",
           "--memory", "3g", "--cpus", "3", "-e", "SEED=%d" % seed,
           "-v", WORK + ":/work", "-v", BENCH + ":/bench:ro",
           "agent-sandbox", "python", "train_candidate.py"]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", "agent-queue"], capture_output=True)
        return None
    if not os.path.exists(res):
        return None
    return json.load(open(res))["val_bits_per_char"]

def load():
    out = {}
    if os.path.exists(LOG):
        for line in open(LOG, encoding="utf-8"):
            if line.strip():
                e = json.loads(line)
                out[e["name"]] = e
    return out

def log(e):
    e["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(e) + "\n")

def summary():
    done = load()
    print("\nname        status        mean     gain vs base")
    for e in sorted(done.values(), key=lambda e: e.get("mean") or 99):
        m, g = e.get("mean"), e.get("gain")
        print("%-11s %-13s %-8s %s" % (e["name"], e["status"],
              "-" if m is None else "%.4f" % m, "-" if g is None else "%+.4f" % g))

def main():
    if "report" in sys.argv:
        summary()
        return
    os.makedirs(LOGS, exist_ok=True)
    base_text = apply(open(os.path.join(WORK, "train.py"), encoding="utf-8").read(), BASE_EDITS)
    done = load()
    for name, edits in [("base", [])] + list(CANDS.items()):
        if os.path.exists(STOP):
            print("STOP file found, finishing.")
            break
        if name in done:
            continue
        t0 = time.time()
        try:
            text = apply(base_text, edits)
        except AssertionError:
            log({"name": name, "status": "bad_patch"})
            continue
        with open(os.path.join(WORK, "train_candidate.py"), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        base = done.get("base")
        scores = {"0": run(0)}
        s0 = scores["0"]
        if name != "base" and base and (s0 is None or not (s0 <= base["scores"]["0"] + 0.05)):
            status = "early_abort"
        else:
            for seed in (1, 2):
                scores[str(seed)] = run(seed)
            status = "ok" if all(v is not None for v in scores.values()) else "failed"
        mean = gain = None
        if status == "ok":
            mean = sum(scores.values()) / 3
            if base:
                gain = base["mean"] - mean
        entry = {"name": name, "status": status, "scores": scores, "mean": mean,
                 "gain": gain, "seconds": int(time.time() - t0)}
        log(entry)
        done[name] = entry
        print(name, status, scores, "mean", mean, "gain", gain, flush=True)
        if name == "base" and status != "ok":
            print("Base run failed, stopping.")
            return
    summary()

main()
