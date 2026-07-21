# Method-v2 diagnostic (2026-07-16, before any v2 code change)

Per the keep-if-wins discipline: diagnose first (baseline misconfig / signal discriminability
/ stratified budget / under-training-saturation / seed noise), only add a module if it wins on
3 seeds without unacceptable regression. Source: controller_current_canonical_v5.json.

## Central finding: metric is stable, the PICK is not

Across 21 configs x 3 seeds, the controller's selected strategy differs seed-to-seed on 18 of
21 configs (only ts-ETTh2, vision-drop-infl, vision-drop-red pick a stable strategy class). Yet
the test-metric standard deviation is small (mostly 0.002-0.030). Examples:

| config | test mean +- sd | picks across 3 seeds |
|---|---|---|
| tep-base | 0.4146 +- 0.0032 | fuse / vote_ensemble / dmf |
| vision-base | 0.4315 +- 0.0043 | dmf / auth2_only / auth_only |
| ts-ETTh1 | 1.0040 +- 0.0122 | dmf / quadmix / quadmix |
| tabular-base | 0.8734 +- 0.0126 | density / tabpfn_hybrid / kcenter |

Interpretation: many portfolio candidates are near-tied on the validation split, so seed noise
in the validation probe flips which near-equal candidate wins. Because the winners are near-tied,
the downstream test metric barely moves - the instability is benign for the reported number.

## What this rules OUT as v2 levers

- NOT baseline misconfiguration: fidelity curve (Tier-1) passed; baselines behave as published.
- NOT under-training/saturation: 3-seed sd is small and stable; no divergence.
- NOT a metric problem: the controller's test metric is already competitive and low-variance.

## The one honest v2 lever: pick-stability (robustness, not metric)

The defensible improvement is to make the SELECTION itself stable across seeds (a reviewer can
ask "why does your controller pick a different strategy each seed?"), WITHOUT changing the test
outcome. Candidate mechanism (no tuning): aggregate the validation adjudication across the V1/V2
folds (or seed-repeats) and apply the Thm-4 paired-difference margin so the controller only
deviates from a stable reference when the gain is significant beyond paired noise; among
statistically-tied candidates, pick a fixed canonical (e.g. the reference baseline) rather than
the raw argmax. Expected effect: pick-stability up, test metric unchanged within seed noise.

## keep-if-wins gate for v2

Promote a v2 mechanism ONLY if, on the same seeds / same init / same data order:
  (a) pick-stability strictly increases (more configs pick a stable strategy class), AND
  (b) NO key-domain test regression beyond seed noise (|delta| <= existing sd) on vision/tep/
      tabular/ts base arms.
Otherwise it stays a documented negative result / ablation, not overwriting current canonical.

## Honest bottom line

The current method's reported METRICS are already stable and audited-clean. The only genuine
improvement space is selection-robustness, which is benign for the numbers. This is a
"nice-to-have robustness" change, not a metric win - the "keep only if it wins" bar for a
metric improvement is NOT met by the diagnostic. Decision on whether to spend GPU on a
pick-stability v2 vs lock the current fully-audited results for the AAAI deadline is a scope
call for the user.
