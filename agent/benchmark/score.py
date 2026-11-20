import math
import torch
import torch.nn.functional as F

_text = open("/bench/data.txt", encoding="utf-8").read()
_chars = sorted(set(_text))
VOCAB_SIZE = len(_chars)
_stoi = {c: i for i, c in enumerate(_chars)}
_data = torch.tensor([_stoi[c] for c in _text])
_n = int(len(_data) * 0.9)

def load_train():
    return _data[:_n]

def evaluate(model, block):
    model.eval()
    val = _data[_n:]
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val) - block - 1, block):
            x = val[i:i + block].unsqueeze(0)
            y = val[i + 1:i + block + 1].unsqueeze(0)
            logits = model(x)
            total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum").item()
            count += y.numel()
    return total / count / math.log(2)
