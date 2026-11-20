p = r"C:\agent\orchestrator.py"
s = open(p, encoding="utf-8").read()
a = """    lines = ["- %s: %s -> %s" % (h["status"], h["hypothesis"], h.get("bpc")) for h in history[-8:]]"""
b = """    lines = ["- %s: %s -> %s %s" % (h["status"], h["hypothesis"], h.get("bpc"), (h.get("error") or "")[-200:].replace("\\n", " ")) for h in history[-8:]]"""
assert a in s
s = s.replace(a, b)
c = '"appears exactly once), new (the replacement text)."'
d = '"appears exactly once), new (the replacement text). The result must be valid Python: every name must be defined before it is used, and old lines you do not re-add in new are deleted. If the idea needs changes in several places, make old one larger block that covers all of them. Do not repeat an idea that already crashed unless you fix the reported error."'
assert c in s
s = s.replace(c, d)
open(p, "w", encoding="utf-8").write(s)
print("patched")
