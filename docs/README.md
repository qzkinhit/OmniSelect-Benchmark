# Public documentation map

This directory intentionally contains only documents needed to understand, reproduce,
or extend the released benchmark. Internal development conversations, temporary review
notes, superseded theory drafts, and one-off server handoff logs are not part of the
public research artifact.

| File | Purpose |
|---|---|
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Exact runners, metrics, seeds, and protocol boundaries |
| [`ARTIFACTS_INDEX.md`](ARTIFACTS_INDEX.md) | What is in Git, what must be downloaded, and SHA/license status |
| [`dataset_provenance.md`](dataset_provenance.md) | Source/version/label/split provenance |
| [`baseline_fidelity_ledger.md`](baseline_fidelity_ledger.md) | Official vs reimplemented baseline fidelity tiers |
| [`master_coverage_matrix.md`](master_coverage_matrix.md) | Method × task applicability and coverage |
| [`full_paper_coverage_ledger.md`](full_paper_coverage_ledger.md) | Paper-facing coverage summary |
| [`ablation_master.md`](ablation_master.md) | Measured ablations and evidence level |
| [`architecture.md`](architecture.md) | Package and runner architecture |
| [`THEORY.md`](THEORY.md) | Formal problem statement and proofs |
| [`ccs_anchor_protocol.md`](ccs_anchor_protocol.md) | CCS released-implementation anchor protocol |
| [`CONTRIBUTING_DATASETS.md`](CONTRIBUTING_DATASETS.md) | Add a dataset or modality |
| [`results_schema.json`](results_schema.json) | Machine-readable result schema |

Advanced protocol evidence retained because it is referenced by the runners or canonical
tables: [`frozen_baseline_fidelity_gates.md`](frozen_baseline_fidelity_gates.md),
[`published_method_fidelity_gate.md`](published_method_fidelity_gate.md),
[`quadmix_styleproxy_invalidation.md`](quadmix_styleproxy_invalidation.md), and
[`zip_protocol_review.md`](zip_protocol_review.md).

Dataset license snapshots and byte-level checksums are under
[`provenance_evidence/`](provenance_evidence/). Dataset download instructions live in
[`../data/README.md`](../data/README.md); result navigation lives in
[`../results_canonical/README.md`](../results_canonical/README.md).
