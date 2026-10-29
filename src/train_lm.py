"""Pre-train one LM for one tokenizer and seed (1 epoch over the shared 200 MB corpus), then report BPB.

usage: python src/train_lm.py TOKENIZER SEED
Writes results/lm/{TOK}_s{SEED}/{model.pt, metrics.json, train_log.jsonl}.
"""
import fcntl
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from model import GPT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ENC = ROOT / "results" / "encoded"
CFG = dict(d=512, n_layer=6, n_head=8, ctx=512, tokens_per_step=32768, micro_bs=int(os.environ.get("MICRO_BS", 32)),
           lr=1e-3, min_lr_frac=0.1, warmup=200, wd=0.1, clip=1.0)


def token_byte_lengths(tok):
    """UTF-8 byte length of every token id (separator = 1 byte)."""
    from bpe import base_alphabet
    inv = {c: i for i, c in enumerate(base_alphabet())}
    lens = np.zeros(tok.get_vocab_size() + 1, dtype=np.int64)
    for s, i in tok.get_vocab().items():
        lens[i] = sum(1 for ch in s if ch in inv)
    lens[-1] = 1
    return lens


@torch.no_grad()
def bpb(model, ids, blen, ctx, dev, bs=16):
    """Bits per byte on non-overlapping windows of ctx+1 tokens (predicting the last ctx tokens of each)."""
    model.eval()
    n = (len(ids) - 1) // ctx
    x = torch.from_numpy(ids[: n * ctx + 1].astype(np.int64))
    nll, nbytes = 0.0, 0
    for i in range(0, n, bs):
        j = min(n, i + bs)
        inp = torch.stack([x[k * ctx:(k + 1) * ctx] for k in range(i, j)]).to(dev)
        tgt = torch.stack([x[k * ctx + 1:(k + 1) * ctx + 1] for k in range(i, j)]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inp)
        nll += F.cross_entropy(logits.float().view(-1, logits.shape[-1]), tgt.view(-1), reduction="sum").item()
        nbytes += int(blen[tgt.cpu().numpy()].sum())
    model.train()
    return nll / math.log(2) / nbytes, nll / (n * ctx)


def main(name, seed):
    out = ROOT / "results" / "lm" / f"{name}_s{seed}"
    if (out / "metrics.json").exists():
        print("already done", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    lockf = open(out / ".lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("LOCKED by another process", out)
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda"
    tok = Tokenizer.from_file(str(ROOT / "results" / "tokenizers" / f"{name}.json"))
    V = tok.get_vocab_size() + 1
    blen = token_byte_lengths(tok)
    data = np.load(ENC / f"{name}.lm_train.npy")
    ctx = CFG["ctx"]
    n_seq = (len(data) - 1) // ctx
    order = np.random.RandomState(seed).permutation(n_seq)
    seq_per_step = CFG["tokens_per_step"] // ctx
    accum = seq_per_step // CFG["micro_bs"]
    steps = n_seq // seq_per_step
    model = GPT(V, CFG["d"], CFG["n_layer"], CFG["n_head"], ctx).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": CFG["wd"]}, {"params": nodecay, "weight_decay": 0}],
                            lr=CFG["lr"], betas=(0.9, 0.95), fused=True)
    print(f"{name} s{seed}: {len(data)/1e6:.1f}M tokens, {n_seq} seqs, {steps} steps, {n_params/1e6:.1f}M params",
          flush=True)
    log = open(out / "train_log.jsonl", "w")
    t0 = time.time()
    x_all = torch.from_numpy(data.astype(np.int64))
    for step in range(steps):
        lr_t = CFG["lr"] * (step + 1) / CFG["warmup"] if step < CFG["warmup"] else \
            CFG["lr"] * (CFG["min_lr_frac"] + (1 - CFG["min_lr_frac"]) * 0.5 *
                         (1 + math.cos(math.pi * (step - CFG["warmup"]) / max(1, steps - CFG["warmup"]))))
        for g in opt.param_groups:
            g["lr"] = lr_t
        tot = 0.0
        for a in range(accum):
            idx = order[step * seq_per_step + a * CFG["micro_bs"]: step * seq_per_step + (a + 1) * CFG["micro_bs"]]
            b = torch.stack([x_all[k * ctx:k * ctx + ctx + 1] for k in idx]).to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(b[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, V), b[:, 1:].reshape(-1)) / accum
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == steps - 1:
            rec = {"step": step, "loss": tot, "lr": lr_t, "elapsed": time.time() - t0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % 200 == 0:
                print(rec, flush=True)
    train_sec = time.time() - t0
    torch.save({"cfg": CFG, "vocab": V, "state": model.state_dict()}, out / "model.pt")
    m = {"tokenizer": name, "seed": seed, "n_params": n_params, "steps": steps, "train_tokens": steps * CFG["tokens_per_step"],
         "train_sec": train_sec, "final_train_loss": tot}
    for split in ["eval_general", "eval_domain"]:
        b_, nll_tok = bpb(model, np.load(ENC / f"{name}.{split}.npy"), blen, ctx, dev)
        m[f"{split}_bpb"] = b_
        m[f"{split}_nll_per_token"] = nll_tok
    json.dump(m, open(out / "metrics.json", "w"), indent=1)
    print(m, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
