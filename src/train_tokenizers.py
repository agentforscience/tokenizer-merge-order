"""Train the six tokenizers of the study (vocab 8192, byte-level BPE).

FREQ        standard most-frequent-first on the 3:1 general:domain tokenizer corpus
DOMAIN      most-frequent-first, domain pre-token counts up-weighted x DOMAIN_WEIGHT
RAND_LEARN  uniformly random pair among pairs with count >= MIN_COUNT
REV_LEARN   least frequent pair among pairs with count >= MIN_COUNT
RAND_ORDER  FREQ merges, random dependency-respecting priority order
REV_ORDER   FREQ merges, dependency-respecting reversed priority order
"""
import json
import pickle
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bpe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "processed"
OUT = ROOT / "results" / "tokenizers"
VOCAB = 8192
MIN_COUNT = 50
DOMAIN_WEIGHT = 8
MIN_WORD_COUNT = 2  # pre-tokens seen once are dropped from the training dictionary (speed)


def job(args):
    name, wc, rule = args
    t = time.time()
    vocab, merges, st = bpe.train_bpe(wc, VOCAB, rule=rule, min_count=MIN_COUNT, seed=0)
    bpe.save(str(OUT / name), vocab, merges, {"rule": rule, "train_sec": time.time() - t})
    with open(OUT / f"{name}.raw.pkl", "wb") as f:
        pickle.dump((vocab, merges, st), f)
    print(name, "done", time.time() - t, flush=True)
    return name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "wordcounts.pkl"
    if cache.exists():
        g, d = pickle.load(open(cache, "rb"))
    else:
        g = bpe.count_words(DATA / "tok_general.txt")
        d = bpe.count_words(DATA / "tok_domain.txt")
        pickle.dump((g, d), open(cache, "wb"))
    total = g + d
    keep = {w for w, c in total.items() if c >= MIN_WORD_COUNT}
    print(f"pre-tokens: {len(total)} unique, {len(keep)} with count>={MIN_WORD_COUNT}")
    std = {w: total[w] for w in keep}
    dom = {w: g.get(w, 0) + DOMAIN_WEIGHT * d.get(w, 0) for w in keep}
    jobs = [("FREQ", std, "freq"), ("DOMAIN", dom, "freq"), ("RAND_LEARN", std, "randpair"),
            ("REV_LEARN", std, "revfreq")]
    with Pool(4) as p:
        p.map(job, jobs)

    vocab, merges, _ = pickle.load(open(OUT / "FREQ.raw.pkl", "rb"))
    for name, mode in [("RAND_ORDER", "random"), ("REV_ORDER", "reverse")]:
        m2 = bpe.reorder(vocab, merges, mode, seed=0)
        bpe.save(str(OUT / name), vocab, m2, {"rule": f"reorder_{mode}_of_FREQ"})

    # sanity check: compare our FREQ merges with HuggingFace's BpeTrainer on the same word counts
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    hf = Tokenizer(models.BPE())
    hf.pre_tokenizer = None
    tr = trainers.BpeTrainer(vocab_size=VOCAB, initial_alphabet=bpe.base_alphabet(), show_progress=False,
                             min_frequency=0)
    # byte-level pre-tokens contain no ASCII whitespace, so repeat each word `count` times, space-separated
    hf_counts = Counter(std)
    hf.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    def it2():
        buf = []
        for w, c in hf_counts.items():
            buf.append((w + " ") * int(c))
            if len(buf) >= 1000:
                yield "".join(buf)
                buf = []
        yield "".join(buf)
    hf.train_from_iterator(it2(), tr)
    hf_merges = [tuple(m) if not isinstance(m, str) else tuple(m.split(" "))
                 for m in json.loads(hf.to_str())["model"]["merges"]]
    ours = [tuple(m) for m in merges]
    res = {}
    for k in (500, 1000, 4000, len(ours)):
        res[f"overlap_first_{k}"] = len(set(ours[:k]) & set(hf_merges[:k])) / k
    print("FREQ vs HF BpeTrainer merge-set overlap:", res)
    json.dump(res, open(OUT / "hf_sanity_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
