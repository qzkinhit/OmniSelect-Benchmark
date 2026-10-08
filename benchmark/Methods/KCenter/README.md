# KCenter

Fidelity label: **local implementation (portfolio-internal)**.

Source: Sener and Savarese, ICLR 2018.

Strategy names: `kcenter`. Implementation: [`method.py`](method.py).

Farthest-point traversal from a seeded start.

Deviations: The active-learning loop is replaced by one-shot budgeted selection. k-center instantiates the controller's coverage signal and is not counted among the 11 compared strategies.

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
