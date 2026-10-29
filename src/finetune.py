"""Fine-tune a pre-trained LM on SST-2, AG News and PubMed-RCT (2 fine-tuning seeds each) and save item predictions.

usage: python src/finetune.py TOKENIZER PRETRAIN_SEED
Classifier = mean-pooled final hidden states (right padding; causal attention so pads never affect real tokens)
-> linear head; full fine-tuning with identical hyper-parameters for every tokenizer.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from model import GPT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "processed"
TASKS = {"sst2": 2, "agnews": 4, "pubmed": 5}
HP = dict(max_len=256, bs=32, epochs=2, lr=2e-4, wd=0.01, warmup_frac=0.06)
FT_SEEDS = [0, 1]


def load(task, split, tok):
    rows = [json.loads(l) for l in open(DATA / f"cls_{task}_{split}.jsonl")]
    enc = tok.encode_batch([r["text"] for r in rows])
    ids = [e.ids for e in enc]
    trunc = float(np.mean([len(x) > HP["max_len"] for x in ids]))
    ids = [x[: HP["max_len"]] or [0] for x in ids]
    return ids, np.array([r["label"] for r in rows]), trunc, float(np.mean([len(x) for x in ids]))


def batches(ids, y, bs, pad, shuffle_rng=None):
    order = np.arange(len(ids))
    if shuffle_rng is not None:
        shuffle_rng.shuffle(order)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        L = max(len(ids[k]) for k in idx)
        x = torch.full((len(idx), L), pad, dtype=torch.long)
        m = torch.zeros((len(idx), L))
        for r, k in enumerate(idx):
            x[r, :len(ids[k])] = torch.tensor(ids[k])
            m[r, :len(ids[k])] = 1
        yield x, m, torch.from_numpy(y[idx]), idx


class Clf(nn.Module):
    def __init__(self, lm, n_cls):
        super().__init__()
        self.lm = lm
        self.head = nn.Linear(lm.wte.weight.shape[1], n_cls)

    def forward(self, x, m):
        h = self.lm.hidden(x)
        h = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
        return self.head(h)


def macro_f1(y, p, k):
    f = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def run(name, pseed, task, fseed, tok, ckpt, dev):
    torch.manual_seed(fseed)
    lm = GPT(ckpt["vocab"], ckpt["cfg"]["d"], ckpt["cfg"]["n_layer"], ckpt["cfg"]["n_head"], ckpt["cfg"]["ctx"])
    lm.load_state_dict(ckpt["state"])
    model = Clf(lm, TASKS[task]).to(dev)
    pad = ckpt["vocab"] - 1
    tr_ids, tr_y, tr_trunc, tr_len = load(task, "train", tok)
    te_ids, te_y, te_trunc, te_len = load(task, "test", tok)
    steps = HP["epochs"] * math.ceil(len(tr_ids) / HP["bs"])
    warm = int(HP["warmup_frac"] * steps)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / (steps - warm)))
    rng = np.random.RandomState(fseed)
    t0 = time.time()
    model.train()
    for ep in range(HP["epochs"]):
        for x, m, y, _ in batches(tr_ids, tr_y, HP["bs"], pad, rng):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x.to(dev), m.to(dev))
            loss = F.cross_entropy(logits.float(), y.to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
    model.eval()
    preds = np.zeros(len(te_ids), dtype=int)
    with torch.no_grad():
        for x, m, y, idx in batches(te_ids, te_y, 128, pad):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds[idx] = model(x.to(dev), m.to(dev)).argmax(-1).cpu().numpy()
    return {"tokenizer": name, "pretrain_seed": pseed, "task": task, "ft_seed": fseed,
            "acc": float(np.mean(preds == te_y)), "macro_f1": macro_f1(te_y, preds, TASKS[task]),
            "train_trunc_frac": tr_trunc, "test_trunc_frac": te_trunc, "train_mean_tokens": tr_len,
            "test_mean_tokens": te_len, "sec": time.time() - t0, "preds": preds.tolist(), "labels": te_y.tolist()}


def main(name, pseed):
    out = ROOT / "results" / "ft" / f"{name}_s{pseed}"
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    tok = Tokenizer.from_file(str(ROOT / "results" / "tokenizers" / f"{name}.json"))
    ckpt = torch.load(ROOT / "results" / "lm" / f"{name}_s{pseed}" / "model.pt", map_location="cpu")
    for task in TASKS:
        for fs in FT_SEEDS:
            f = out / f"{task}_ft{fs}.json"
            if f.exists():
                continue
            r = run(name, pseed, task, fs, tok, ckpt, dev)
            json.dump(r, open(f, "w"))
            print(name, pseed, task, fs, f"acc={r['acc']:.4f} f1={r['macro_f1']:.4f} trunc={r['test_trunc_frac']:.3f} "
                  f"{r['sec']:.0f}s", flush=True)
    json.dump({"done": True, "hp": HP}, open(out / "done.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
