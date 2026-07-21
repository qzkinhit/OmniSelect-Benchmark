# Published-method fidelity gate

Date: 2026-07-16

This gate separates original-system reproduction from a controlled implementation
under the unified protocol. A method is not promoted by wording alone.

## Status table

| Method | Published core | Local status | Required label | Remaining action |
|---|---|---|---|---|
| DSIR | Hashed n-gram importance ratio and sampling without replacement | Official package alignment is available. Rank correlation is 0.855 and top-50% overlap is 0.855 | official-implementation alignment under a shared pool | Preserve package version, target-corpus SHA, pool SHA, seed, and selected-id SHA |
| SemDeDup | K-means partition, within-cluster cosine duplicate threshold, retain the lowest centroid-similarity representative | Core rule and direction match. Cluster count and scale are reduced | reimplemented from the published method under our unified protocol | Record embedding model, cluster count, cosine threshold, and scale reduction |
| Density | Sample inversely to local density | Direction and sampling without replacement match. Web-scale KDE and LSH are replaced by a kNN density estimator | reimplemented with a kNN density-estimator substitution | Record k, metric, feature SHA, Gumbel sampling seed, and replacement rationale |
| QuaDMix | Domain-specific quality merging, within-domain quality rank, sigmoid expected-repeat sampling, proxy-model regression for parameter search | Current row is a quality-bucket and farthest-first proxy. It is not the published sampling function | QuaDMix-style proxy | Preserve the current result as a proxy. Implement a separate published-core version before using a stronger label |
| Dynamic fusion or DMF | Actor-memory EMA in Eq. 5, weighted actor score in Eq. 6, additive collaboration update in Eq. 8 | Current row uses multiplicative channel weights and local validation reward | dynamic-fusion proxy | Implement the additive published update, disclose replacement of actor memories and influence reward, then rerun only DMF and affected controller rows |
| GraNd | Expected full-network gradient norm across early training | Current row uses an expected last-layer gradient-norm proxy | GraNd proxy | Keep the proxy label. Do not claim full GraNd reproduction |

## Gate requirements

Each controlled implementation must store:

1. Paper identifier, equation or algorithm number, and quality direction.
2. Original sampling rule and the local fixed-budget adaptation.
3. Original data, model, estimator, and compute scale.
4. Every substituted component and why it is needed.
5. Pool, validation, test, feature, configuration, code, and selected-id hashes.
6. At least one mechanism test and three paper seeds.
7. Mean and standard deviation under the unified protocol.
8. A claim-limit field that forbids original absolute-number claims.

The paper-facing wording is:

`reimplemented from the published method under our unified protocol`

This wording is allowed only after the corresponding row above has passed its
mechanism test and three-seed artifact validation. QuaDMix and dynamic fusion remain
explicit proxies until their published-core reruns are complete.

## Existing mechanism tests

`pytest -q tests/test_baseline_fidelity.py` passed 8 of 8 tests on 2026-07-16.
These tests cover herding mean matching, k-center coverage radius, SemDeDup duplicate
removal, Density inverse-density behavior, distinct EL2N and GraNd rankings, CCS
difficulty coverage, dynamic-fusion preference adaptation, and the current
QuaDMix-style quality and dispersion behavior. The last two tests validate the
current proxies. They do not certify equation-level fidelity to the original systems.
