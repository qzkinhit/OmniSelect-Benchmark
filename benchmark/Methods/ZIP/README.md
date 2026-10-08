# ZIP

Fidelity label: **local implementation of the published greedy rule**.

Source: Yin et al., Entropy Law (ZIP), arXiv 2407.06645, 2024.

Strategy names: `zip`. Implementation: [`method.py`](method.py).

Greedy compression-ratio selection over text with a bit-exact incremental evaluator.

Deviations: Text only.

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
