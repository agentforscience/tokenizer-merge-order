# Outline: Same Merges, Different Order

1. Intro: hook (tokenizer frozen into every model; merge order never varied) -> gap (no controlled study isolating order; learned vs applied order never separated) -> approach (6 tokenizers, 2 families, Fig 1) -> preview (BPB +1.8-5.8% gen / up to 9.8% dom, all sig; same-vocab reorder costs 1.8-5.7%; classification within 1.6%, n.s.; DOMAIN -0.8%) -> 4 contributions.
2. Related work: tokenizer comparisons (Bostrom&Durrett, Ali, Schmidt); compression & intrinsic metrics (Galle, Zouhar, Goldman); BPE variants/stochastic seg (Provilkov, Kudo); domain vocabularies (SciBERT, PubMedBERT, Gururangan).
3. Method: BPE formulation (learning rule sigma, application rank pi), Alg 1; six tokenizers (Table 1); data; model; metrics (BPB eq); statistics.
4. Results: intrinsics (Tab 2 + hypertension example) -> BPB (Tab 3, Fig 2) -> compression mechanism (Fig 3, rho=-1; Renyi rho=-0.71 n.s.) -> classification (Tab 4) -> robustness (Tab 5, seeds, compute) -> hypothesis verdicts (Tab 6).
5. Discussion: why compression; why fine-tuning robust; domain weighting small; practical guidance; limitations.
6. Conclusion.
Appendix: training curves, compute table, macro-F1, extra details.
