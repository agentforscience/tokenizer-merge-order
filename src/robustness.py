"""Robustness: (a) second random draw of RAND_ORDER / RAND_LEARN tokenizers (pre-train seed 0 only) vs the first
draw; (b) compute per condition; (c) Spearman correlation between compression and BPB degradation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
lm = pd.read_csv(R / "table_lm_runs.csv")
ft = pd.read_csv(R / "table_ft_runs.csv")
intr = json.load(open(R / "intrinsics.json"))
rows = []
for base in ["RAND_ORDER", "RAND_LEARN"]:
    for t in [base, base + "_t1"]:
        m = lm[(lm.tokenizer == t) & (lm.seed == 0)].iloc[0]
        f = ft[(ft.tokenizer == t) & (ft.pretrain_seed == 0)].groupby("task").acc.mean()
        rows.append({"tokenizer": t, "bytes_per_token_general": intr[t]["eval_general_bytes_per_token"],
                     "bpb_general": m.eval_general_bpb, "bpb_domain": m.eval_domain_bpb,
                     **{f"acc_{k}": v for k, v in f.items()}})
fr = lm[(lm.tokenizer == "FREQ") & (lm.seed == 0)].iloc[0]
rob = pd.DataFrame(rows)
rob["bpb_general_rel_vs_FREQ_s0_pct"] = 100 * (rob.bpb_general / fr.eval_general_bpb - 1)
rob["bpb_domain_rel_vs_FREQ_s0_pct"] = 100 * (rob.bpb_domain / fr.eval_domain_bpb - 1)
rob.to_csv(R / "table_robustness_tokenizer_seed.csv", index=False)
print(rob.round(4).to_string())

comp = lm.groupby("tokenizer").agg(train_tokens=("train_tokens", "mean"), train_min=("train_sec", lambda s: s.mean() / 60),
                                   steps=("steps", "mean")).reset_index()
comp["rel_compute_vs_FREQ"] = comp.train_tokens / comp[comp.tokenizer == "FREQ"].train_tokens.iloc[0]
comp.to_csv(R / "table_compute.csv", index=False)
print(comp.round(2).to_string())

b = pd.read_csv(R / "table_bpb.csv")
out = {}
for split in ["general", "domain"]:
    s = b[b.split == f"eval_{split}"]
    x = [intr[t][f"eval_{split}_bytes_per_token"] for t in s.tokenizer]
    rho, p = spearmanr(x, s.bpb_mean)
    out[split] = {"spearman_bytes_per_token_vs_bpb": rho, "p": p}
    # same-vocabulary family only (FREQ, RAND_ORDER, REV_ORDER)
print(out)
json.dump(out, open(R / "compression_bpb_corr.json", "w"), indent=1)
