# Herding

Fidelity label: **local implementation**.

Source: Welling, Herding Dynamical Weights to Learn, ICML 2009 (DeepCore implementation).

Strategy names: `herding (herding_text on text)`. Implementation: [`method.py`](method.py).

Greedy mean matching on the track representation.

Deviations: Features are frozen embeddings or task features, not the source's kernel space.

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
