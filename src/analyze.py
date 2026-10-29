"""Aggregate results, run statistics vs FREQ, write tables (results/*.csv, results/summary.json) and figures/.

Statistics
* BPB: per-document bits/bytes, averaged over pre-training seeds; relative change vs FREQ = ratio of corpus BPB - 1;
  95% CI by paired bootstrap over documents (10k resamples); two-sided paired sign-flip permutation test on per-doc
  BPB differences (10k permutations). The final (partially evaluated) document of each split is dropped.
* Fine-tuning: per-item correctness averaged over the 4 runs (2 pre-train x 2 FT seeds); relative accuracy change vs
  FREQ with paired bootstrap CI over test items; sign-flip permutation test on per-item differences.
* Holm correction over all (condition x metric) tests vs FREQ.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
NAMES = ["FREQ", "DOMAIN", "RAND_LEARN", "REV_LEARN", "RAND_ORDER", "REV_ORDER"]
TASKS = ["sst2", "agnews", "pubmed"]
B = 10000
rng = np.random.default_rng(0)


def perm_test(diff, n=B):
    diff = np.asarray(diff, float)
    obs = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, len(diff)))
    null = np.abs((signs * diff).mean(1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (n + 1))


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1, run)
    return adj


def load_lm():
    rows, docs = [], {}
    for d in sorted((RES / "lm").glob("*_s*")):
        if not (d / "metrics.json").exists():
            continue
        m = json.load(open(d / "metrics.json"))
        rows.append(m)
        if (d / "doc_bpb.json").exists():
            docs[(m["tokenizer"], m["seed"])] = json.load(open(d / "doc_bpb.json"))
    return pd.DataFrame(rows), docs


def load_ft():
    rows = []
    for f in sorted((RES / "ft").glob("*_s*/*_ft*.json")):
        rows.append(json.load(open(f)))
    return rows


def main():
    FIG.mkdir(exist_ok=True)
    intr = json.load(open(RES / "intrinsics.json"))
    it = pd.DataFrame({n: {k: v for k, v in intr[n].items() if k != "examples"} for n in NAMES}).T
    it.to_csv(RES / "table_intrinsics.csv")
    summary = {"intrinsics": it.to_dict(orient="index")}

    lm, docs = load_lm()
    lm.to_csv(RES / "table_lm_runs.csv", index=False)
    tests = []
    # ---------------- BPB ----------------
    bpb_rows = []
    for split in ["eval_general", "eval_domain"]:
        per = {}
        for n in NAMES:
            seeds = [s for (t, s) in docs if t == n]
            if not seeds:
                continue
            bits = np.mean([np.array(docs[(n, s)][split]["nll_bits"])[:-1] for s in seeds], 0)
            byts = np.array(docs[(n, seeds[0])][split]["bytes"])[:-1]
            per[n] = (bits, byts, seeds)
        if "FREQ" not in per:
            continue
        fb, fy, _ = per["FREQ"]
        for n, (bits, byts, seeds) in per.items():
            seed_bpb = lm[(lm.tokenizer == n)][f"{split}_bpb"].values
            corpus = bits.sum() / byts.sum()
            row = {"split": split, "tokenizer": n, "n_seeds": len(seeds), "bpb_mean": corpus,
                   "bpb_seed_sd": float(np.std(seed_bpb, ddof=1)) if len(seed_bpb) > 1 else np.nan,
                   "bpb_seeds": seed_bpb.tolist(), "n_docs": len(bits)}
            if n != "FREQ":
                # documents have (nearly) identical byte counts across tokenizers; use FREQ's for both ratios
                rel = corpus / (fb.sum() / fy.sum()) - 1
                idx = rng.integers(0, len(bits), size=(B, len(bits)))
                boot = (bits[idx].sum(1) / byts[idx].sum(1)) / (fb[idx].sum(1) / fy[idx].sum(1)) - 1
                p = perm_test(bits / byts - fb / fy)
                row.update(rel_vs_freq_pct=100 * rel, ci_lo=100 * np.percentile(boot, 2.5),
                           ci_hi=100 * np.percentile(boot, 97.5), p=p)
                tests.append({"metric": f"BPB {split}", "tokenizer": n, "p": p})
            bpb_rows.append(row)
    bpb = pd.DataFrame(bpb_rows)
    bpb.to_csv(RES / "table_bpb.csv", index=False)
    summary["bpb"] = bpb_rows

    # ---------------- fine-tuning ----------------
    ft = load_ft()
    ft_rows = []
    if ft:
        df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("preds", "labels")} for r in ft])
        df.to_csv(RES / "table_ft_runs.csv", index=False)
        for task in TASKS:
            corr = {}
            for n in NAMES:
                runs = [r for r in ft if r["tokenizer"] == n and r["task"] == task]
                if runs:
                    corr[n] = (np.mean([np.array(r["preds"]) == np.array(r["labels"]) for r in runs], 0), runs)
            if "FREQ" not in corr:
                continue
            fc = corr["FREQ"][0]
            for n, (c, runs) in corr.items():
                accs = [r["acc"] for r in runs]
                f1s = [r["macro_f1"] for r in runs]
                row = {"task": task, "tokenizer": n, "n_runs": len(runs), "acc_mean": float(np.mean(accs)),
                       "acc_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else np.nan,
                       "f1_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else np.nan,
                       "test_trunc_frac": runs[0]["test_trunc_frac"], "test_mean_tokens": runs[0]["test_mean_tokens"]}
                if n != "FREQ":
                    idx = rng.integers(0, len(c), size=(B, len(c)))
                    boot = c[idx].mean(1) / fc[idx].mean(1) - 1
                    p = perm_test(c - fc)
                    row.update(rel_vs_freq_pct=100 * (c.mean() / fc.mean() - 1), ci_lo=100 * np.percentile(boot, 2.5),
                               ci_hi=100 * np.percentile(boot, 97.5),
                               rel_f1_vs_freq_pct=100 * (np.mean(f1s) / np.mean([r["macro_f1"] for r in corr["FREQ"][1]]) - 1),
                               p=p)
                    tests.append({"metric": f"acc {task}", "tokenizer": n, "p": p})
                ft_rows.append(row)
    ftt = pd.DataFrame(ft_rows)
    ftt.to_csv(RES / "table_ft.csv", index=False)
    summary["ft"] = ft_rows

    if tests:
        tt = pd.DataFrame(tests)
        tt["p_holm"] = holm(tt.p.values)
        tt.to_csv(RES / "table_tests.csv", index=False)
        summary["tests"] = tt.to_dict(orient="records")
    json.dump(summary, open(RES / "summary.json", "w"), indent=1, default=float)

    # ---------------- figures ----------------
    colors = {"FREQ": "#4C72B0", "DOMAIN": "#55A868", "RAND_LEARN": "#C44E52", "REV_LEARN": "#8172B2",
              "RAND_ORDER": "#DD8452", "REV_ORDER": "#937860"}
    if len(bpb):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for a, split in zip(ax, ["eval_general", "eval_domain"]):
            s = bpb[bpb.split == split].set_index("tokenizer").reindex(NAMES).dropna(subset=["bpb_mean"])
            a.bar(s.index, s.bpb_mean, yerr=s.bpb_seed_sd.fillna(0), color=[colors[n] for n in s.index], capsize=3)
            a.set_title(f"Held-out bits per byte: {split.split('_')[1]} (lower = better)")
            a.set_ylabel("bits / byte")
            a.tick_params(axis="x", rotation=30)
            a.set_ylim(s.bpb_mean.min() * 0.9, s.bpb_mean.max() * 1.03)
        plt.tight_layout()
        plt.savefig(FIG / "bpb.png", dpi=150)
        plt.close()
    if len(ftt):
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for a, task in zip(ax, TASKS):
            s = ftt[ftt.task == task].set_index("tokenizer").reindex(NAMES).dropna(subset=["acc_mean"])
            a.bar(s.index, 100 * s.acc_mean, yerr=100 * s.acc_sd.fillna(0), color=[colors[n] for n in s.index],
                  capsize=3)
            a.set_title(f"{task} test accuracy (%)")
            a.set_ylim(100 * s.acc_mean.min() - 5, 100 * s.acc_mean.max() + 3)
            a.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(FIG / "ft_accuracy.png", dpi=150)
        plt.close()
        # relative change vs FREQ, all metrics, with CIs
        rel = []
        for r in bpb_rows:
            if r["tokenizer"] != "FREQ" and "rel_vs_freq_pct" in r:
                # sign flipped so that positive = better, like accuracy
                rel.append((r["tokenizer"], f"BPB {r['split'].split('_')[1]}", -r["rel_vs_freq_pct"], -r["ci_hi"], -r["ci_lo"]))
        for r in ft_rows:
            if r["tokenizer"] != "FREQ":
                rel.append((r["tokenizer"], f"acc {r['task']}", r["rel_vs_freq_pct"], r["ci_lo"], r["ci_hi"]))
        rel = pd.DataFrame(rel, columns=["tok", "metric", "v", "lo", "hi"])
        metrics = list(dict.fromkeys(rel.metric))
        fig, ax = plt.subplots(figsize=(12, 4.5))
        w = 0.8 / 5
        for k, n in enumerate([x for x in NAMES if x != "FREQ"]):
            s = rel[rel.tok == n].set_index("metric").reindex(metrics)
            xs = np.arange(len(metrics)) + (k - 2) * w
            ax.bar(xs, s.v, w, color=colors[n], label=n,
                   yerr=[s.v - s.lo, s.hi - s.v], capsize=2)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels(metrics)
        ax.set_ylabel("% change vs FREQ (positive = better)")
        ax.set_title("Relative performance vs standard frequency BPE (95% paired-bootstrap CI)")
        ax.legend(ncol=5, fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG / "relative_vs_freq.png", dpi=150)
        plt.close()
    # compression vs quality
    if len(bpb):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for a, split in zip(ax, ["general", "domain"]):
            s = bpb[bpb.split == f"eval_{split}"]
            for _, r in s.iterrows():
                x = intr[r.tokenizer][f"eval_{split}_bytes_per_token"]
                a.scatter(x, r.bpb_mean, color=colors[r.tokenizer], s=50)
                a.annotate(r.tokenizer, (x, r.bpb_mean), fontsize=8, xytext=(3, 3), textcoords="offset points")
            a.set_xlabel("bytes per token (compression)")
            a.set_ylabel("bits per byte")
            a.set_title(f"{split}: compression vs LM quality")
        plt.tight_layout()
        plt.savefig(FIG / "compression_vs_bpb.png", dpi=150)
        plt.close()
    # training curves (loss per byte)
    fig, ax = plt.subplots(figsize=(7, 4))
    for d in sorted((RES / "lm").glob("*_s0")):
        n = d.name.rsplit("_s", 1)[0]
        f = d / "train_log.jsonl"
        if not f.exists() or n not in colors:
            continue
        L = pd.read_json(f, lines=True)
        bpt = intr[n]["lm_train_bytes_per_token"]
        frac = L.step / L.step.max()
        ax.plot(frac, L.loss / np.log(2) / bpt, label=n, color=colors[n])
    ax.set_xlabel("fraction of the (same) training text seen")
    ax.set_ylabel("train loss, bits per byte (approx.)")
    ax.set_ylim(0.8, 3)
    ax.legend(fontsize=8)
    ax.set_title("Training curves (seed 0)")
    plt.tight_layout()
    plt.savefig(FIG / "train_curves.png", dpi=150)
    plt.close()
    print(it.round(3).to_string())
    print(bpb.round(4).to_string())
    print(ftt.round(4).to_string())
    if tests:
        print(tt.round(5).to_string())


if __name__ == "__main__":
    main()
