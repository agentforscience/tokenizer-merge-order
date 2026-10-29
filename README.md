# Tokenizer Merge-Order Sensitivity

This project is a controlled study of how the order of BPE merges affects small language models. It compares six
byte-level BPE tokenizers (8k vocabulary): standard frequency, domain-weighted, random, and reverse-frequency
learning order, plus random and reversed *application priority* of the standard merges. Each tokenizer is used to
pre-train identical 23M-parameter GPT models on the same 200 MB of Wikipedia and PubMed text. The models are then
scored by held-out bits per byte and by fine-tuning on SST-2, AG News and PubMed-RCT.

## Key findings
- **LM quality depends strongly on merge order.** Every alternative to standard frequency BPE is significantly worse
  in bits per byte: +1.8% to +5.8% on general text and +2.8% to +9.8% on biomedical text (all Holm-adjusted p ≤ 0.03).
  Reverse-frequency learning is the worst.
- **Merge priority alone matters.** Reordering the *same* merges, with the same vocabulary, costs +1.8–5.7% BPB,
  because long tokens stop firing (59–79% of the vocabulary goes unused). Compression rank-predicts BPB
  (Spearman ρ = −1.0 on general text).
- **Fine-tuned classification does not depend on merge order.** All conditions are within −0.8% to +1.6% of
  standard BPE, and none is significant. Even a near-character-level tokenizer is within 1%.
- **Domain-weighted merges help the domain only slightly:** −0.8% domain BPB, +0.4% general BPB, and no gain on
  domain classification. This is far below the hypothesised 5–12%.

See [REPORT.md](REPORT.md) for the full results, statistics and limitations.

## Reproduce
```bash
uv venv --python 3.12 && source .venv/bin/activate && uv sync   # deps in pyproject.toml
export NUMPY_MADVISE_HUGEPAGE=0        # avoids a numpy hugepage allocation stall on some hosts
# PubMed RCT parquet files go in datasets/raw/ (see src/prepare_data.py; HF: armanc/pubmed-rct20k,
# pietrolesci/pubmed-200k-rct). WikiText-103, SST-2 and AG News are read from the HF hub cache.
python src/prepare_data.py             # corpora + classification sets
python src/train_tokenizers.py         # 6 tokenizers (+ HF BpeTrainer sanity check)
python src/extra_tokenizers.py         # robustness: second random draws
python src/encode_and_intrinsics.py    # encode corpora, tokenizer metrics
python src/gpu_queue.py logs/jobs_lm2.txt   # pre-train (or: python src/train_lm.py FREQ 0)
python src/gpu_queue.py logs/jobs_ft.txt    # per-doc BPB + fine-tuning (eval_docs.py, finetune.py)
python src/analyze.py && python src/robustness.py
```
Each LM run takes 18–57 min and each fine-tuning run about 1 min, on a shared RTX A6000 slice (about 3 GB).

## Layout
- `src/`: BPE trainer (`bpe.py`), data, training, evaluation, analysis and queue scripts
- `results/`:
  - `tokenizers/`: tokenizers
  - `lm/`: LM metrics, per-document BPB and logs (checkpoints are not in git)
  - `ft/`: per-run fine-tuning predictions
  - `table_*.csv`, `summary.json`: aggregated tables
- `figures/`: BPB, accuracy, relative-change, compression and training-curve plots
- `planning.md`: the research plan; `logs/`: execution logs
