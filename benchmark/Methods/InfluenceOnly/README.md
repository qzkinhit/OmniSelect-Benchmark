# InfluenceOnly

Fidelity label: **built-in (OmniSelect signal)**.

Source: OmniSelect influence channel.

Strategy names: `influence_only`. Implementation: [`method.py`](method.py).

Keeps the k records with the largest influence score. Classification tracks: log-probability of the observed label under a logistic probe fit on the influence reference sample (canonical: clean-tagged pool records, v2: records of V_con). Forecasting: negative absolute error of a reference DLinear. Text: negative reference-model loss.

Deviations: none

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
