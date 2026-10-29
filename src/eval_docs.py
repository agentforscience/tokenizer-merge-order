"""Per-document bits-per-byte for a pre-trained LM (enables paired, document-level bootstrap across tokenizers).

usage: python src/eval_docs.py TOKENIZER SEED
The token stream and windows are identical to train_lm.bpb(); each predicted token's NLL and byte length are
attributed to the document containing it (documents are delimited by the separator id).
Writes results/lm/{TOK}_s{SEED}/doc_bpb.json: {split: {"nll_bits": [...], "bytes": [...]}}.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from model import GPT  # noqa: E402
from train_lm import token_byte_lengths  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main(name, seed, bs=16):
    d = ROOT / "results" / "lm" / f"{name}_s{seed}"
    ck = torch.load(d / "model.pt", map_location="cpu")
    cfg, V = ck["cfg"], ck["vocab"]
    model = GPT(V, cfg["d"], cfg["n_layer"], cfg["n_head"], cfg["ctx"]).cuda()
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(str(ROOT / "results" / "tokenizers" / f"{name}.json"))
    blen = token_byte_lengths(tok)
    ctx = cfg["ctx"]
    out = {}
    for split in ["eval_general", "eval_domain"]:
        ids = np.load(ROOT / "results" / "encoded" / f"{name}.{split}.npy").astype(np.int64)
        doc_of = np.concatenate([[0], np.cumsum(ids == V - 1)[:-1]])  # separator belongs to the doc it ends
        n_docs = int((ids == V - 1).sum())
        n = (len(ids) - 1) // ctx
        x = torch.from_numpy(ids[: n * ctx + 1])
        tok_nll = np.zeros(n * ctx)
        for i in range(0, n, bs):
            j = min(n, i + bs)
            inp = torch.stack([x[k * ctx:(k + 1) * ctx] for k in range(i, j)]).cuda()
            tgt = torch.stack([x[k * ctx + 1:(k + 1) * ctx + 1] for k in range(i, j)]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(inp)
            tok_nll[i * ctx:j * ctx] = F.cross_entropy(lg.float().view(-1, V), tgt.view(-1),
                                                       reduction="none").cpu().numpy()
        tgt_pos = np.arange(1, n * ctx + 1)
        docs = doc_of[tgt_pos]
        bits = np.bincount(docs, weights=tok_nll / math.log(2), minlength=n_docs)
        nbytes = np.bincount(docs, weights=blen[ids[tgt_pos]], minlength=n_docs)
        out[split] = {"nll_bits": bits.tolist(), "bytes": nbytes.tolist()}
        print(name, seed, split, "bpb", bits.sum() / nbytes.sum(), flush=True)
    json.dump(out, open(d / "doc_bpb.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
