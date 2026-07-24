# Fixed-primary text portfolio registration

This folder is the paper-facing registration for the complete five-domain text
portfolio completed on 2026-07-24. The raw result JSON is deliberately kept
verbatim so its source artifact hash remains independently checkable.

All eight applicable candidates share the recorded pool, downstream model,
token cap, initial state, training order, and evaluation path. The controller
retains the Herding reference after fixed fusion fails confirmation; therefore
OmniSelect and Herding have the same geometric-mean PPL.

The raw result JSON is small enough for Git. The full reproducibility bundle,
selected corpus records, and eight checkpoints are about 2.3 GiB and are
intentionally excluded from this repository and from the AAAI upload package.
They are retained as a local archive with the server-issued SHA-256 manifests.

To validate a downloaded full bundle after it has been moved, supply its run
directory explicitly:

    python scripts/validate_text_seed0_primary.py result \
      --result /path/to/run/results.json \
      --artifact-root /path/to/run \
      --output /tmp/validated_result.json

The result JSON contains original server checkpoint paths to preserve its source
hash. The validator resolves those paths beneath the supplied artifact root.
