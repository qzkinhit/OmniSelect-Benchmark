# TabAICL

Fidelity label: **one-shot transfer. Iterative protocol reproduced separately**.

Source: Ma et al., Tab-AICL, arXiv 2026.

Strategy names: `tabpfn_coreset, tabpfn_margin, tabpfn_hybrid`. Implementation: [`method.py`](method.py).

The margin, coreset and hybrid acquisition rules applied once under the fixed budget on the tabular track, reading TabPFN probabilities from a seed context.

Deviations: Tab-AICL is round-based active labeling. Iterative_aulc.py reproduces that protocol on its original datasets and it is not compared head to head.

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
