# AlignmentOnly

Fidelity label: **exact** (the rule is defined here).

Strategy name: `alignment_only`. Implementation: [`method.py`](method.py).

The single-signal rule of the alignment channel of protocol v2.1, the k records with the largest alignment, which is the cosine of the record's last-layer gradient at the warm-start parameters with the mean V_con gradient (GLISTER's round-0 gain normalized per record), or the LESS score on text. It is a reference-eligible member of the `unified_v21` portfolio on every track where the channel exists and is absent from the v2 portfolio.
