# FullData

Fidelity label: **reference row**.

Source: Training on the whole pool.

Strategy names: `full`. Implementation: [`method.py`](method.py).

Selects every record. It is the Full reference of the main table and the full-pool fit of the headroom precheck.

Deviations: none

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
