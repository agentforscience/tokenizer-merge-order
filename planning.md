# Planning: Tokenizer Merge-Order Sensitivity

## Motivation & Novelty Assessment

### Why This Research Matters
BPE tokenizers are trained once and then frozen into every model built on them, yet the *order* in which merges are
learned (and, at inference time, the priority order in which they are applied) is almost never varied. If downstream
quality is sensitive to merge order, tokenizer design choices (frequency vs. domain weighting vs. alternatives) deserve
more care; if models are robust, practitioners can freely re-weight or re-order merges (e.g. for domain adaptation)
without hurting general performance.

### Gap in Existing Work
(The resource-finder stage was skipped for this workspace: no `papers/`, `datasets/`, `code/` or literature review
were provided, so this is based on prior knowledge.) Prior work compares tokenizer *algorithms* (BPE vs. Unigram vs.
WordPiece; Bostrom & Durrett 2020), vocabulary size, compression (Gallé 2019; Goldman et al. 2024), Rényi efficiency
(Zouhar et al. 2023), BPE-dropout (Provilkov et al. 2020) and domain-adapted vocabularies (e.g. SciBERT, PubMedBERT,
Gururangan et al. 2020). We are not aware of a controlled study that varies **only the merge selection order / priority**
of BPE while holding data, vocabulary size, model and training budget fixed, and that separates
(a) *which merges are learned* from (b) *in what priority they are applied*.

### Our Novel Contribution
A controlled from-scratch study: six byte-level BPE tokenizers (8,192 vocab) that differ only in merge order, each used
to pre-train identical small GPT-style LMs on the same text (general Wikipedia + biomedical abstracts), evaluated with
tokenizer-agnostic bits-per-byte and fine-tuned on general (SST-2, AG News) and domain (PubMed-RCT) classification.
Two families of manipulation:
* **Learning order** (changes the vocabulary): standard most-frequent-first (FREQ), domain-weighted frequency (DOMAIN),
  uniformly random pair among frequent pairs (RAND-LEARN), least-frequent-first (REV-LEARN).
* **Application order** (same vocabulary and merges as FREQ, only merge priority changed): dependency-respecting random
  shuffle (RAND-ORDER) and dependency-respecting reversal (REV-ORDER). These isolate "merge sequence" in its pure form.

### Experiment Justification
- Exp 1 (tokenizer intrinsics: compression, vocab utilisation, Rényi efficiency, common-word fragmentation): explains *why*
  any downstream differences arise; needed to interpret the "suboptimal subword boundaries" mechanism in the hypothesis.
- Exp 2 (LM pre-training, BPB on held-out general and domain text, 2 seeds): the most sensitive, tokenizer-agnostic
  measure of model quality; tests general vs. domain trade-offs.
- Exp 3 (fine-tuning on SST-2, AG News, PubMed-RCT, 2 pre-train seeds × 2 fine-tune seeds): tests the hypothesis'
  "% performance on tasks" claims directly.
- Exp 4 (application-order controls): disentangles vocabulary content from merge priority.

## Research Question
Does the order of BPE merges (learning order and application priority) affect downstream LM quality, and specifically:
(H1) FREQ beats alternatives by 3–8% on general tasks; (H2) DOMAIN beats FREQ by 5–12% on the domain;
(H3) random order is only 2–5% worse than FREQ; (H4) reverse-frequency degrades most (10–20%).

## Hypothesis Decomposition / Metrics
"Performance" is operationalised as (i) relative change in held-out bits-per-byte (lower is better; reported as % change
vs FREQ), (ii) relative change in fine-tuned accuracy / macro-F1 vs FREQ. Each sub-hypothesis is supported if the
observed relative difference (with 95% CI) falls in / overlaps the stated range and is in the stated direction.

## Proposed Methodology
1. Data: WikiText-103 (general, ~150 MB train subset; test split held-out), PubMed-200k-RCT sentences (domain, ~50 MB,
   exact-deduplicated against PubMed-20k-RCT val/test); 3:1 general:domain by bytes. Tokenizer corpus = 25% subsample.
2. Own BPE trainer (word-count dictionary with incremental pair counts) supporting selection rules; validated against
   HuggingFace `tokenizers` BpeTrainer for FREQ. Export to HF ByteLevel-BPE for fast encoding (rank = merge priority).
3. LM: GPT-style decoder (8 layers, d=512, 8 heads, ctx 256 tokens, ~30M params), 1 epoch over the same text for every
   tokenizer, AdamW, cosine LR, bf16; 2 seeds.
4. Eval: BPB = total NLL (nats) / ln2 / UTF-8 bytes on WikiText-103 test and PubMed-20k test abstracts.
5. Fine-tune (mean-pool + linear head, full fine-tuning, 2 epochs) on SST-2 (20k train / 872 val), AG News
   (20k / 3,000 test), PubMed-RCT-20k (20k sentences / 3,000 test). 2 FT seeds per pre-trained model.
6. Statistics: per-condition mean ± sd over seeds; relative % change vs FREQ with bootstrap 95% CIs over test items
   (resampling items, averaging seeds); paired permutation tests vs FREQ, Holm correction.

## Baselines
FREQ (standard BPE) is the reference; RAND-ORDER/REV-ORDER are same-vocabulary controls.

## Expected Outcomes
Hypothesis supported if FREQ ≥ alternatives on general, DOMAIN > FREQ on PubMed, RAND within 2–5%, REV-LEARN worst.
Plausible alternative: differences are tiny (<1–2%) for application-order changes, large for REV-LEARN purely through
compression (more tokens per byte, shorter byte context).

## Timeline
Setup/data 30 min; BPE implementation + tokenizers 45 min; LM training (12 runs, shared GPUs) ~1.5–2.5 h; fine-tuning
~1 h; analysis + report 1 h.

## Potential Challenges
Shared GPUs (often ≤5 GB free): model sized to fit ~3 GB; queue with retries. Python BPE speed: restrict word dictionary
to pre-tokens with count ≥ 2. Unequal token counts per tokenizer → report both equal-data and compute statistics.

## Success Criteria
All six tokenizers trained, 12 LMs pre-trained, BPB + 3 fine-tuning tasks evaluated with CIs, hypothesis parts
explicitly accepted/rejected in REPORT.md.
