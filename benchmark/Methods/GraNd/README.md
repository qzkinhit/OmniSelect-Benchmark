# GraNd

Fidelity label: **proxy (disclosed last-layer gradient norm)**.

Source: Paul et al., NeurIPS 2021.

Strategy names: `grand`. Implementation: [`method.py`](method.py).

Average over five early-training linear heads of ||p - onehot(y)|| sqrt(||phi||^2 + 1).

Deviations: Never a full-network GraNd. Classification only.

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
