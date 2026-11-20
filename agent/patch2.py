p = r"C:\agent\try_ideas.py"
s = open(p, encoding="utf-8").read()
a = 'IDEAS = {"warmup_cosine"'
assert a in s
s = s.replace(a, 'WARMUP_8E3 = [(o, n.replace("BASE_LR = 5e-3", "BASE_LR = 8e-3")) for o, n in WARMUP_COSINE]\nIDEAS = {"peak8": WARMUP_8E3, "warmup_cosine"')
open(p, "w", encoding="utf-8").write(s)
print("patched")
