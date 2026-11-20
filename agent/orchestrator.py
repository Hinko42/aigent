import json, os, re, shutil, subprocess, sys, time, urllib.request

ROOT = r"C:\agent"
WORK = os.path.join(ROOT, "workspace")
BENCH = os.path.join(ROOT, "benchmark")
LOGS = os.path.join(ROOT, "logs")
TRAIN = os.path.join(WORK, "train.py")
RESULT = os.path.join(WORK, "result.json")
HISTORY = os.path.join(LOGS, "history.jsonl")
BEST_COPY = os.path.join(LOGS, "best_train.py")
LLM = "qwen2.5-coder:7b"
MIN_GAIN = 0.03      # an edit must beat the best score by this much (bits/char)
RUN_TIMEOUT = 900    # seconds per training run

def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()

def write(p, t):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(t)

def log(entry):
    entry["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def load_history():
    if not os.path.exists(HISTORY):
        return []
    return [json.loads(l) for l in read(HISTORY).splitlines() if l.strip()]

def run_training():
    if os.path.exists(RESULT):
        os.remove(RESULT)
    cmd = ["docker", "run", "--rm", "--name", "agent-run", "--network", "none",
           "--memory", "3g", "--cpus", "3",
           "-v", WORK + ":/work", "-v", BENCH + ":/bench:ro",
           "agent-sandbox", "python", "train.py"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", "agent-run"], capture_output=True)
        return None, "timeout"
    if p.returncode != 0 or not os.path.exists(RESULT):
        return None, (p.stderr or p.stdout)[-500:]
    return json.loads(read(RESULT))["val_bits_per_char"], ""

def guard(text):
    # the agent may not touch block size, step budget, scoring or the score import
    ok = re.search(r"^BLOCK,.*MAX_STEPS\s*=\s*128,.*,\s*1000\s*$", text, re.M)
    return bool(ok) and "from score import" in text and "evaluate(model, BLOCK)" in text

def ask_llm(script, history):
    lines = ["- %s: %s -> %s %s" % (h["status"], h["hypothesis"], h.get("bpc"), (h.get("error") or "")[-200:].replace("\n", " ")) for h in history[-8:]]
    prompt = (
        "You are improving a tiny character-level transformer. Goal: lower the validation "
        "bits-per-character (val_bits_per_char) after exactly 1000 training steps on CPU. "
        "Here is the current train.py:\n\n" + script +
        "\n\nRecent experiments (status: idea -> score):\n" + ("\n".join(lines) or "none yet") +
        "\n\nPropose ONE small change you have not tried yet (examples: learning rate, "
        "lr schedule or warmup, weight decay, initialization, layer norm placement, "
        "activation, width/depth, dropout, residual scaling). Do not change BLOCK, "
        "MAX_STEPS, or the evaluate call. Reply with JSON only, with keys: "
        "hypothesis (one sentence), old (exact text copied from train.py that "
        "appears exactly once), new (the replacement text). The result must be valid Python: every name must be defined before it is used, and old lines you do not re-add in new are deleted. If the idea needs changes in several places, make old one larger block that covers all of them. Do not repeat an idea that already crashed unless you fix the reported error."
    )
    body = json.dumps({"model": LLM, "stream": False, "format": "json",
                       "messages": [{"role": "user", "content": prompt}],
                       "options": {"num_ctx": 16384, "temperature": 0.8}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    return json.loads(out["message"]["content"])

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    os.makedirs(LOGS, exist_ok=True)
    if os.path.exists(BEST_COPY):
        write(TRAIN, read(BEST_COPY))   # always start from the best known version
    history = load_history()
    scores = [h["bpc"] for h in history if h["status"] in ("baseline", "accepted")]
    best = min(scores) if scores else None
    if best is None:
        print("Running baseline...")
        bpc, err = run_training()
        if bpc is None:
            print("Baseline failed:", err)
            return
        best = bpc
        shutil.copy(TRAIN, BEST_COPY)
        log({"hypothesis": "baseline", "status": "baseline", "bpc": bpc})
        history = load_history()
        print("Baseline:", bpc)
    for i in range(n):
        print("--- experiment %d/%d (best %.4f) ---" % (i + 1, n, best))
        script = read(TRAIN)
        try:
            prop = ask_llm(script, history)
        except Exception as e:
            print("LLM error:", e)
            continue
        hyp = str(prop.get("hypothesis", ""))[:300]
        old, new = prop.get("old"), prop.get("new")
        entry = {"hypothesis": hyp, "old": old, "new": new, "bpc": None}
        print("Idea:", hyp)
        if not isinstance(old, str) or not isinstance(new, str) or script.count(old) != 1:
            entry["status"] = "invalid_edit"
        else:
            patched = script.replace(old, new)
            if not guard(patched):
                entry["status"] = "rejected_guard"
            else:
                write(TRAIN, patched)
                bpc, err = run_training()
                entry["bpc"] = bpc
                if bpc is None:
                    entry["status"] = "crashed"
                    entry["error"] = err
                    write(TRAIN, script)
                elif bpc < best - MIN_GAIN:
                    entry["status"] = "accepted"
                    best = bpc
                    shutil.copy(TRAIN, BEST_COPY)
                else:
                    entry["status"] = "no_gain"
                    write(TRAIN, script)
        print("Result:", entry["status"], entry["bpc"])
        log(entry)
        history = load_history()

main()
