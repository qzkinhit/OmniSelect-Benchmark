# SemDeDup

Fidelity label: **local implementation (portfolio-internal)**.

Source: Abbas et al., arXiv 2303.09540, 2023.

Strategy names: `semdedup`. Implementation: [`method.py`](method.py).

Within-cluster cosine deduplication with threshold 1 - 0.05, keeping the low-centroid-similarity member of each duplicate group.

Deviations: Web-scale clustering (50k clusters) is out of reach. The cluster count is min(64, n // 20).

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
