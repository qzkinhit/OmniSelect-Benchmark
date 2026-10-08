# Density

Fidelity label: **local implementation**.

Source: Sachdeva et al., How to Train Data-Efficient LLMs, 2024.

Strategy names: `density (density_text on text)`. Implementation: [`method.py`](method.py).

Inverse-density Gumbel top-k with a 10-nearest-neighbour sparsity estimate.

Deviations: The kNN estimator replaces the source's LSH density estimator. Canonical vision, forecasting, TEP and tabular rows used seed 0 on every run seed (fixes.density_seed).

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
