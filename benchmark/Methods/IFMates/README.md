# IFMates

Fidelity label: **local implementation**.

Source: Koh and Liang, ICML 2017. Yu et al., MATES, NeurIPS 2024.

Strategy names: `if_mates`. Implementation: [`method.py`](method.py).

Influence Top-K over precomputed text influence scores.

Deviations: Text only. Shares the reference-perplexity influence of the text track.

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
