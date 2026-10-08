# DsDm

Fidelity label: **proxy-scale reduction**.

Source: Engstrom et al., DsDm: Model-Aware Dataset Selection with Datamodels, 2024.

Strategy names: `dsdm`. Implementation: [`method.py`](method.py).

Ridge datamodel over 12 to 20 random half-pool subsets scored on the construction split.

Deviations: Far fewer subset fits than the source. Not run on text or the native ResNet-18 protocol.

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
