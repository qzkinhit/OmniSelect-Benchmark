# Coverage

Fidelity label: **built-in (OmniSelect signal)**.

Source: Clustered coverage.

Strategy names: `coreset`. Implementation: [`method.py`](method.py).

k-means with k clusters on the track representation and the record nearest each centroid. Empty clusters are filled deterministically so exactly k records return. The text track uses domain-wise farthest-first traversal of frozen LM embeddings (coverage_text).

Deviations: The native ResNet-18 protocol uses MiniBatchKMeans because k reaches 13,500 to 36,000.

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
