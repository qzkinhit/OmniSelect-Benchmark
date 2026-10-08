# CCS

Fidelity label: **local implementation**.

Source: Zheng et al., Coverage-centric Coreset Selection, ICLR 2023.

Strategy names: `ccs`. Implementation: [`method.py`](method.py).

EL2N-binned stratified coverage: drop the hardest 10%, 50 equal-width strata, sparse strata first.

Deviations: The official pipeline uses AUM scores. An official released-implementation anchor was run separately and does not change this row's label.

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
