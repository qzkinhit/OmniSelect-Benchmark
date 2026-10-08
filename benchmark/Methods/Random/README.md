# Random

Fidelity label: **exact (control)**.

Source: Uniform sampling without replacement.

Strategy names: `random`. Implementation: [`method.py`](method.py).

Draws k records from a permutation seeded by stable_seed(seed, 'select', 'random'). This is the random-allocation reference of Theorem 1 and a reference-eligible member on every track.

Deviations: none

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
