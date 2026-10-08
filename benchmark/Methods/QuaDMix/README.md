# QuaDMix

Fidelity label: **published-core transfer (quadmix_pub). Withdrawn proxy (quadmix)**.

Source: QuaDMix, arXiv 2504.16511, 2025.

Strategy names: `quadmix_pub, quadmix`. Implementation: [`method.py`](method.py).

Eqs. 1 to 3 with frozen parameters (lambda 100, omega 0.05, eta 1, epsilon 0.001), k-means domains where native labels are missing, Gumbel top-k without replacement.

Deviations: The 3,000-proxy-model LightGBM search is replaced by the frozen parameters. The style proxy produced duplicate ids and is withdrawn from every table. It is registered only to replay historical portfolios.

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
