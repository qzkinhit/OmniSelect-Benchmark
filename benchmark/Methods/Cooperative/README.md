# Cooperative

Fidelity label: **local implementation** (the rule is defined here and in `omniselect/core/selection/cooperative.py`).

Strategy name: `coop_herding`. Implementation: [`method.py`](method.py).

The prespecified member of the cooperative candidates of protocol v2.2 ([docs/DESIGN.md](../../../docs/DESIGN.md)).
Three steps run on the pool only:

1. Gate: the ceil(1.3 k) records with the highest cleanliness score (the track's authenticity signal
   by default, `cooperative.gate_score`), ties by pool index. Token-budgeted text keeps the records of
   the score order inside 1.3 times each domain's token budget.
2. Vetoes in the selector's feature space (cosine distance on L2-normalized embeddings for vision, text
   and native, Euclidean distance on the window features for forecasting and on the standardized rows
   for TEP21 and Electricity): an admissible record is dropped when the robust z-score of its distance
   to the 10th nearest pool neighbour exceeds 3 (z = (d - median) / (1.4826 MAD) over the pool), and
   of each near-duplicate group (records linked when one is among the other's 10 neighbours at a
   distance below 0.1 times the median nearest-neighbour distance) only the cleanest admissible member
   stays. When fewer than k records remain, dropped records return in score order.
3. Herding (the rule of `benchmark/Methods/Herding`) selects k records inside the admissible set. On
   text the herding order is the token-budgeted per-domain rule of `herding_text`.

It is a reference-eligible member of the `unified_v22` portfolio on every track and is absent from
the v2 and v2.1 portfolios. The challenger grid of the same family (gate width in {1.15, 1.3, 1.5, 2.0,
none} times selector in {herding, kcenter, random, mmr}) is built by the driver and screened on V_con.
`scores.json` `selection_info` records the gate width, the selector, the admissible sizes, the drops
of each veto, the refill count and the seconds.
