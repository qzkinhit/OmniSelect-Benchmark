# EL2N

Fidelity label: **qualitative protocol transfer**.

Source: Paul, Ganguli and Dziugaite, Deep Learning on a Data Diet, NeurIPS 2021.

Strategy names: `el2n`. Implementation: [`method.py`](method.py).

Error norm of the probe's class probabilities. Keeps the hardest k.

Deviations: Defined only for classification. The native ResNet-18 track scores from early checkpoints averaged over score runs. The frozen-feature tracks use a logistic probe.

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
