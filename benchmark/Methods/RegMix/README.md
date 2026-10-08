# RegMix

Fidelity label: **proxy (never in tables)**.

Source: Liu et al., RegMix, ICLR 2025. Thrush et al., Perplexity Correlations, ICLR 2025.

Strategy names: `none registered`. Implementation: [`method.py`](method.py).

Domain-mixture regression and perplexity-correlation allocation probes on text.

Deviations: Group-level proxies. Not portfolio members.

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
