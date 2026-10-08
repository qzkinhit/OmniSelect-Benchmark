# DSIR

Fidelity label: **official-implementation alignment**.

Source: Xie et al., Data Selection for Language Models via Importance Resampling, NeurIPS 2023.

Strategy names: `dsir`. Implementation: [`method.py`](method.py).

Hashed bigram importance weights toward the clean held-out reference, Gumbel top-k. On text the order is cut to the token budget.

Deviations: Aligned with the official data-selection package ordering (Spearman 0.85).

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
