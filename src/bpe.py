"""Byte-level BPE trainer with pluggable merge-selection rules, plus merge-priority re-orderings.

Learning-order rules (change WHICH merges are learned):
  freq      - standard BPE: merge the most frequent adjacent pair
  revfreq   - merge the LEAST frequent pair among pairs with count >= min_count
  randpair  - merge a uniformly random pair among pairs with count >= min_count
  (domain weighting is done by the caller through word-count weights, with rule=freq)

Application-order transforms (keep the vocabulary and merge set, change merge PRIORITY):
  reorder_random   - random topological order (each merge still comes after the merges creating its two inputs)
  reorder_reverse  - topological order that greedily prefers the LATEST-learned merge (dependency-respecting reversal)

Tokenizers are exported to HuggingFace `tokenizers` ByteLevel BPE, whose BPE model applies merges by rank = position
in the merge list, i.e. exactly the merge priority studied here.
"""
import heapq
import json
import random
from collections import Counter, defaultdict

from tokenizers import Tokenizer, decoders, models, pre_tokenizers

PRE = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)


def base_alphabet():
    """The 256 byte-level unicode symbols, ordered by byte value (GPT-2 bytes_to_unicode)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    m = dict(zip(bs, cs))
    return [chr(m[b]) for b in range(256)]


def count_words(path, chunk_chars=1_000_000):
    """Count byte-level pre-tokens of a text file (documents separated by blank lines)."""
    c = Counter()
    text = open(path, encoding="utf-8").read()
    for doc in text.split("\n\n"):
        c.update(w for w, _ in PRE.pre_tokenize_str(doc))
    return c


def train_bpe(word_counts, vocab_size, rule="freq", min_count=50, seed=0, log_every=1000):
    """Train BPE on a {pretoken_string: weight} dict. Returns (vocab dict str->id, merges list[(a,b)], stats)."""
    rng = random.Random(seed)
    alphabet = base_alphabet()
    id2tok = list(alphabet)
    tok2id = {t: i for i, t in enumerate(id2tok)}
    words, counts = [], []
    for w, c in word_counts.items():
        words.append([tok2id[ch] for ch in w])
        counts.append(c)

    pair_cnt = defaultdict(float)
    where = defaultdict(set)
    for wi, sym in enumerate(words):
        c = counts[wi]
        for a, b in zip(sym, sym[1:]):
            pair_cnt[(a, b)] += c
            where[(a, b)].add(wi)

    # selection structures
    heap = []  # lazy heap for freq / revfreq
    elig, elig_pos = [], {}  # indexable set for randpair

    def touch(p):
        c = pair_cnt.get(p, 0)
        if rule == "freq":
            if c > 0:
                heapq.heappush(heap, (-c, p))
        elif rule == "revfreq":
            if c >= min_count:
                heapq.heappush(heap, (c, p))
        elif rule == "randpair":
            if c >= min_count and p not in elig_pos:
                elig_pos[p] = len(elig)
                elig.append(p)
            elif c < min_count and p in elig_pos:
                i = elig_pos.pop(p)
                last = elig.pop()
                if i < len(elig):
                    elig[i] = last
                    elig_pos[last] = i

    for p in list(pair_cnt):
        touch(p)

    def select():
        if rule in ("freq", "revfreq"):
            while heap:
                k, p = heapq.heappop(heap)
                c = -k if rule == "freq" else k
                if pair_cnt.get(p, 0) == c and (rule == "freq" or c >= min_count):
                    return p, c
            return None, 0
        if not elig:
            return None, 0
        p = elig[rng.randrange(len(elig))]
        return p, pair_cnt[p]

    merges, merge_counts = [], []
    while len(id2tok) < vocab_size:
        p, c = select()
        if p is None:
            print("no eligible pairs left; stopping at", len(id2tok))
            break
        a, b = p
        s = id2tok[a] + id2tok[b]
        if s in tok2id:
            new = tok2id[s]
        else:
            new = len(id2tok)
            id2tok.append(s)
            tok2id[s] = new
        merges.append((id2tok[a], id2tok[b]))
        merge_counts.append(c)
        changed = set()
        for wi in list(where[p]):
            sym = words[wi]
            cw = counts[wi]
            # remove old pairs of this word
            for x, y in zip(sym, sym[1:]):
                pair_cnt[(x, y)] -= cw
                where[(x, y)].discard(wi)
                changed.add((x, y))
            out, i = [], 0
            while i < len(sym):
                if i + 1 < len(sym) and sym[i] == a and sym[i + 1] == b:
                    out.append(new)
                    i += 2
                else:
                    out.append(sym[i])
                    i += 1
            words[wi] = out
            for x, y in zip(out, out[1:]):
                pair_cnt[(x, y)] += cw
                where[(x, y)].add(wi)
                changed.add((x, y))
        for q in changed:
            if pair_cnt.get(q, 0) <= 1e-9:
                pair_cnt.pop(q, None)
                where.pop(q, None)
            touch(q)
        if log_every and len(merges) % log_every == 0:
            print(f"  [{rule}] {len(merges)} merges, vocab {len(id2tok)}, last={s!r} count={c:.0f}", flush=True)
    vocab = {t: i for i, t in enumerate(id2tok)}
    return vocab, merges, {"merge_counts": merge_counts}


def reorder(vocab, merges, mode, seed=0):
    """Dependency-respecting re-ordering of merge priorities (same merge set, same vocabulary)."""
    alphabet = set(base_alphabet())
    producer = {}  # token string -> index of the first merge producing it
    for i, (a, b) in enumerate(merges):
        producer.setdefault(a + b, i)
    deps = []
    for a, b in merges:
        deps.append({producer[t] for t in (a, b) if t not in alphabet})
    children = defaultdict(list)
    indeg = [len(d) for d in deps]
    for i, d in enumerate(deps):
        for j in d:
            children[j].append(i)
    rng = random.Random(seed)
    if mode == "random":
        ready = [i for i in range(len(merges)) if indeg[i] == 0]
    else:  # reverse: max-heap on original index
        ready = [-i for i in range(len(merges)) if indeg[i] == 0]
        heapq.heapify(ready)
    order = []
    while ready:
        if mode == "random":
            k = rng.randrange(len(ready))
            ready[k], ready[-1] = ready[-1], ready[k]
            i = ready.pop()
        else:
            i = -heapq.heappop(ready)
        order.append(i)
        for ch in children[i]:
            indeg[ch] -= 1
            if indeg[ch] == 0:
                if mode == "random":
                    ready.append(ch)
                else:
                    heapq.heappush(ready, -ch)
    assert len(order) == len(merges)
    return [merges[i] for i in order]


def build_hf(vocab, merges):
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, ignore_merges=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    return tok


def save(path_prefix, vocab, merges, extra=None):
    tok = build_hf(vocab, merges)
    tok.save(f"{path_prefix}.json")
    with open(f"{path_prefix}.meta.json", "w") as f:
        json.dump({"n_merges": len(merges), "vocab_size": len(vocab), **(extra or {})}, f)
    return tok
