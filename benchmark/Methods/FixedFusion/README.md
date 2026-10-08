# FixedFusion

Fidelity label: **built-in**.

Source: Fixed-weight fusion of the three signals.

Strategy names: `mmdataselect` (the strategy id in run records), `fixed_fusion` on the text track. Implementation: [`method.py`](method.py).

Authenticity below the 0.25 quantile is gated, influence and redundancy are blended by a console with weights (0.5, 0.5), and the budget selector with diversity 0.5 selects k records.

Deviations: Canonical rows write -1e9 into gated records before min-max normalization, so passing records get near-identical importance and the diversity term dominates (fixes.fixed_fusion_gate = sentinel). The v2 preset selects within the passing records (finite_subset).

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
