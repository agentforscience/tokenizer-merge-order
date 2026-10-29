"""Encode LM/eval corpora with every tokenizer (uint16 .npy) and compute intrinsic tokenizer metrics.

Documents are joined with a single EOS-like separator id (= vocab_size, outside the BPE vocabulary).
Intrinsic metrics (on held-out eval text):
  bytes_per_token, tokens_per_word, renyi_eff_2.5 (Zouhar et al. 2023), vocab_used_frac (on 20 MB of LM train),
  top1k_single_frac (share of the 1000 most frequent eval words encoded as ONE token),
  seg_agree_freq (share of eval word tokens segmented identically to FREQ).
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "processed"
TOK = ROOT / "results" / "tokenizers"
ENC = ROOT / "results" / "encoded"
NAMES = ["FREQ", "DOMAIN", "RAND_LEARN", "REV_LEARN", "RAND_ORDER", "REV_ORDER", "RAND_LEARN_t1", "RAND_ORDER_t1"]


def encode_file(tok, path):
    docs = open(path, encoding="utf-8").read().split("\n\n")
    sep = tok.get_vocab_size()
    out = []
    for i in range(0, len(docs), 2000):
        for e in tok.encode_batch(docs[i:i + 2000]):
            out.extend(e.ids)
            out.append(sep)
    return np.array(out, dtype=np.uint16), sum(len(d.encode()) + 1 for d in docs)


def renyi_eff(counts, alpha=2.5):
    p = np.array(list(counts.values()), dtype=float)
    p /= p.sum()
    h_a = np.log(np.sum(p ** alpha)) / (1 - alpha)
    return float(h_a / np.log(len(p)))


def main():
    ENC.mkdir(parents=True, exist_ok=True)
    toks = {n: Tokenizer.from_file(str(TOK / f"{n}.json")) for n in NAMES}
    report = {}
    pre = toks["FREQ"].pre_tokenizer
    words = {}
    for split in ["general", "domain"]:
        text = open(DATA / f"eval_{split}.txt", encoding="utf-8").read()
        ws = [w for w, _ in pre.pre_tokenize_str(text)]
        words[split] = ws
    freq_seg = {}
    for n, tok in toks.items():
        r = {"vocab_size": tok.get_vocab_size()}
        for split in ["lm_train", "eval_general", "eval_domain"]:
            path = DATA / f"{split}.txt"
            f = ENC / f"{n}.{split}.npy"
            if not f.exists():
                ids, nbytes = encode_file(tok, path)
                np.save(f, ids)
                json.dump({"n_bytes": nbytes, "n_tokens": int(len(ids))}, open(ENC / f"{n}.{split}.json", "w"))
            meta = json.load(open(ENC / f"{n}.{split}.json"))
            r[f"{split}_tokens"] = meta["n_tokens"]
            r[f"{split}_bytes_per_token"] = meta["n_bytes"] / meta["n_tokens"]
        ids = np.load(ENC / f"{n}.lm_train.npy")[:5_000_000]
        r["vocab_used_frac_5Mtok"] = float(len(np.unique(ids)) - 1) / tok.get_vocab_size()
        for split in ["general", "domain"]:
            ws = words[split]
            uniq = list(Counter(ws))
            seg = {w: tuple(x.value for x in tok.model.tokenize(w)) for w in uniq}
            if n == "FREQ":
                freq_seg[split] = seg
            cnt = Counter(ws)
            ntok = sum(len(seg[w]) * c for w, c in cnt.items())
            r[f"{split}_tokens_per_word"] = ntok / len(ws)
            alpha = [w for w, _ in cnt.most_common() if re.fullmatch(r"Ġ?[A-Za-z]+", w)][:1000]
            r[f"{split}_top1k_single_frac"] = float(np.mean([len(seg[w]) == 1 for w in alpha]))
            r[f"{split}_top1k_tokens_per_word"] = float(np.mean([len(seg[w]) for w in alpha]))
            r[f"{split}_seg_agree_freq"] = sum(c for w, c in cnt.items() if seg[w] == freq_seg[split][w]) / len(ws)
            uc = Counter()
            for w, c in cnt.items():
                for t in seg[w]:
                    uc[t] += c
            r[f"{split}_renyi_eff_2.5"] = renyi_eff(uc)
            if True:
                ex = ["Ġthe", "Ġbecause", "Ġfootball", "Ġrandomized", "Ġpatients", "Ġhypertension",
                      "Ġinternational", "Ġplacebo"]
                r["examples"] = {w: [t for t in (x.value for x in tok.model.tokenize(w))] for w in ex}
        report[n] = r
        print(n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "examples"},
              flush=True)
        print("   ", r["examples"])
    json.dump(report, open(ROOT / "results" / "intrinsics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
