# D4

Fidelity label: **local implementation**.

Source: Tirumala et al., D4, NeurIPS 2023.

Strategy names: `d4`. Implementation: [`method.py`](method.py).

SemDeDup to 75% of the pool, then drop the records closest to their cluster prototype.

Deviations: Embeddings are the track representation.

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
