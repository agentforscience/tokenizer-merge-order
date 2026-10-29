"""Build the text corpora used by every experiment.

Outputs (datasets/processed/):
  lm_train.txt        - LM pre-training text: WikiText-103 (general) + PubMed-200k-RCT (domain), 3:1 by bytes,
                        documents separated by blank lines, shuffled at document level (fixed seed).
  tok_general.txt     - 25% subsample of the general part (tokenizer-training corpus)
  tok_domain.txt      - 25% subsample of the domain part (tokenizer-training corpus)
  eval_general.txt    - WikiText-103 test split (held-out, general)
  eval_domain.txt     - PubMed-20k-RCT test abstracts (held-out, domain)
  cls_{sst2,agnews,pubmed}_{train,test}.jsonl - downstream classification sets
"""
import glob
import json
import random
import re
from pathlib import Path

import pandas as pd

SEED = 0
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "processed"
HUB = "/data/transformers/hub"
GENERAL_MB, DOMAIN_MB = 150, 50
TOK_FRACTION = 0.25
N_CLS_TRAIN, N_CLS_TEST = 20000, 3000


def wikitext_articles(path_glob):
    """Split WikiText lines into articles at top-level ' = Title = ' headings."""
    lines = []
    for f in sorted(glob.glob(path_glob)):
        lines += pd.read_parquet(f).text.tolist()
    arts, cur = [], []
    for ln in lines:
        if re.match(r"^ = [^=].* = \n$", ln) and cur:
            arts.append("".join(cur).strip())
            cur = []
        if ln.strip():
            cur.append(ln.strip(" ") if ln.startswith(" ") else ln)
    if cur:
        arts.append("".join(cur).strip())
    return [a for a in arts if len(a) > 200]


def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    wt = glob.glob(f"{HUB}/datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1")[0]

    # ---------------- general ----------------
    arts = wikitext_articles(f"{wt}/train-*.parquet")
    rng.shuffle(arts)
    general, size = [], 0
    for a in arts:
        if size >= GENERAL_MB * 1e6:
            break
        general.append(a)
        size += len(a.encode())
    print(f"general: {len(general)} articles, {size/1e6:.1f} MB")

    # ---------------- domain (dedup vs PubMed-20k val/test) ----------------
    held = pd.concat([pd.read_parquet(ROOT / f"datasets/raw/pubmed20k/{s}.parquet") for s in ["validation", "test"]])
    held_set = set(held.text.str.strip())
    p200 = pd.read_parquet(ROOT / "datasets/raw/pubmed200k/train.parquet")
    p200["t"] = p200.text.str.strip()
    n0 = len(p200)
    p200 = p200[~p200.t.isin(held_set)]
    print(f"pubmed200k: removed {n0-len(p200)} sentences overlapping PubMed-20k val/test")
    # pseudo-abstracts: a new abstract starts when label returns to background (0) after a later section
    docs, cur, prev = [], [], 0
    for t, lab in zip(p200.t.tolist(), p200.labels.tolist()):
        if cur and lab < prev and lab == 0:
            docs.append(" ".join(cur))
            cur = []
        cur.append(t)
        prev = lab
    docs.append(" ".join(cur))
    rng.shuffle(docs)
    domain, size = [], 0
    for d in docs:
        if size >= DOMAIN_MB * 1e6:
            break
        domain.append(d)
        size += len(d.encode())
    print(f"domain: {len(domain)} abstracts, {size/1e6:.1f} MB")

    lm = general + domain
    rng.shuffle(lm)
    (OUT / "lm_train.txt").write_text("\n\n".join(lm))
    (OUT / "tok_general.txt").write_text("\n\n".join(rng.sample(general, int(len(general) * TOK_FRACTION))))
    (OUT / "tok_domain.txt").write_text("\n\n".join(rng.sample(domain, int(len(domain) * TOK_FRACTION))))

    # ---------------- held-out evaluation text ----------------
    test_arts = wikitext_articles(f"{wt}/test-*.parquet")
    (OUT / "eval_general.txt").write_text("\n\n".join(test_arts))
    t20 = pd.read_parquet(ROOT / "datasets/raw/pubmed20k/test.parquet").sort_values(["abstract_id", "sentence_id"])
    abs_ = t20.groupby("abstract_id").text.apply(lambda s: " ".join(x.strip() for x in s)).tolist()
    rng.shuffle(abs_)
    (OUT / "eval_domain.txt").write_text("\n\n".join(abs_[:1000]))
    print("eval general MB", sum(len(a.encode()) for a in test_arts) / 1e6,
          "eval domain MB", sum(len(a.encode()) for a in abs_[:1000]) / 1e6)

    # ---------------- classification ----------------
    def dump(name, split, df):
        with open(OUT / f"cls_{name}_{split}.jsonl", "w") as f:
            for t, y in zip(df.text, df.label):
                f.write(json.dumps({"text": t, "label": int(y)}) + "\n")
        print(name, split, len(df), df.label.value_counts().to_dict())

    sst = glob.glob(f"{HUB}/datasets--stanfordnlp--sst2/snapshots/*/data")[0]
    tr = pd.read_parquet(f"{sst}/train-00000-of-00001.parquet").rename(columns={"sentence": "text"})
    va = pd.read_parquet(f"{sst}/validation-00000-of-00001.parquet").rename(columns={"sentence": "text"})
    dump("sst2", "train", tr.sample(N_CLS_TRAIN, random_state=SEED))
    dump("sst2", "test", va)  # SST-2 test labels are hidden -> use the 872-sentence validation split
    ag = glob.glob(f"{HUB}/datasets--fancyzhx--ag_news/snapshots/*/data")[0]
    tr = pd.read_parquet(f"{ag}/train-00000-of-00001.parquet")
    te = pd.read_parquet(f"{ag}/test-00000-of-00001.parquet")
    dump("agnews", "train", tr.sample(N_CLS_TRAIN, random_state=SEED))
    dump("agnews", "test", te.sample(N_CLS_TEST, random_state=SEED))
    labs = ["background", "objective", "methods", "results", "conclusions"]
    tr = pd.read_parquet(ROOT / "datasets/raw/pubmed20k/train.parquet")
    te = pd.read_parquet(ROOT / "datasets/raw/pubmed20k/test.parquet")
    for d in (tr, te):
        d["label"] = d.label.map(labs.index)
    dump("pubmed", "train", tr.sample(N_CLS_TRAIN, random_state=SEED))
    dump("pubmed", "test", te.sample(N_CLS_TEST, random_state=SEED))


if __name__ == "__main__":
    main()
