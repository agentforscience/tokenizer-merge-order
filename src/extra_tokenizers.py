"""Robustness: second random draws of the two random tokenizers (tokenizer seed 1)."""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bpe  # noqa: E402
from train_tokenizers import MIN_COUNT, MIN_WORD_COUNT, OUT, VOCAB  # noqa: E402

g, d = pickle.load(open(OUT / "wordcounts.pkl", "rb"))
total = g + d
std = {w: c for w, c in total.items() if c >= MIN_WORD_COUNT}
vocab, merges, _ = bpe.train_bpe(std, VOCAB, rule="randpair", min_count=MIN_COUNT, seed=1, log_every=0)
bpe.save(str(OUT / "RAND_LEARN_t1"), vocab, merges, {"rule": "randpair", "seed": 1})
vocab, merges, _ = pickle.load(open(OUT / "FREQ.raw.pkl", "rb"))
bpe.save(str(OUT / "RAND_ORDER_t1"), vocab, bpe.reorder(vocab, merges, "random", seed=1), {"rule": "reorder_random", "seed": 1})
print("ok")
