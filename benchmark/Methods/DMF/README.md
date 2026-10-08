# DMF

Fidelity label: **published-update transfer (dmf_pub). Proxy (dmf)**.

Source: Bai et al., Efficient Pretraining Data Selection via Multi-Actor Collaboration, ACL 2025.

Strategy names: `dmf_pub, dmf`. Implementation: [`method.py`](method.py).

dmf_pub: six rounds of Eqs. 6 to 8 over the authenticity, influence and redundancy channels with construction-split rewards and simplex projection. dmf: the multiplicative-weight proxy of the historical batches.

Deviations: Actor memories and the pretraining stack are not reproduced. dmf is removed from the unified portfolio. text_dmf.py holds the text console baseline.

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
