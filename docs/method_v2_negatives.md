# method-v2 negative results ledger (2026-07-16)

Keep-if-wins protocol: a candidate is promoted only if it is picked by the controller AND
improves metrics beyond seed noise with no regression on key arms. Everything else is
archived here. Canonical (method-v1) stays untouched.

## Attempt #1: diversity-regularized vote-ensemble (METHOD_V2=1)

Design: vote scores from the V1-top-3 fed to BudgetSelector with lam=0.4, injecting coverage
into the vote consensus. Run: vision/tep/tabular x 3 seeds, RUN_ID=method-v2, same seeds as
canonical. Log: experiments/method_v2_compare_3seed.log.

Verdict: NOT PICKED by any of the 9 (modality,seed) runs (grep 'div(v2)' picked count = 0);
final metrics identical-or-within-noise vs canonical. NEGATIVE. Consistent with
docs/method_v2_diagnostic.md: the honest ceiling is candidate synthesis; the metric is stable
(sd 0.002-0.03) while pick order is the unstable part.

## Attempt #2: complementarity-aware vote (METHOD_V2C=1) — design flaw, deterministic no-op

Codex audit (CODEX_AUDIT_20260716_1230 item 1) identified BEFORE metrics were read that this
variant cannot help by construction: when overlap<thr it appends the already-present
vote_ensemble(top3) selection; otherwise it appends the already-present best member's
selection. Either way the candidate SET is a duplicate, so max validation score over
candidates cannot increase. Server seed0 confirmed: vision v2c val = vote val = 0.4525;
tabular both 0.8636437; TEP v2c fusion-identical.

3-seed test metrics (RUN_ID=method-v2c, log experiments/method_v2c_compare_3seed.log) — all
within canonical seed noise, as forced by the duplication:

| arm | s0 | s1 | s2 |
|---|---|---|---|
| vision test_acc | 0.4340 | 0.4350 | 0.4255 |
| tep macroF1 | 0.4126 | 0.4191 | 0.4120 |
| tabular auc | 0.8758 | 0.8569 | 0.8876 |

Verdict: NEGATIVE (deterministic duplicate, not an empirical loss). Lesson recorded: a future
complementarity variant must produce a genuinely different selected-ID set (minimum bar:
different selected-set SHA) and complementarity should be measured on validation error types /
harmful mis-selections, not the top-3 intersection-over-union.

Note: the TEP rows above also re-exposed the uncalibrated operating point (FDR 0.98 / FAR 0.98
at seed0) — tracked separately as the TEP threshold-calibration fix; not a v2c artifact.

## Standing protocol for any further method-v2 attempt (frozen, per audit item 6)

The test sets consulted in attempts #1/#2 are now development data. Any next attempt must:
design and promote on train/V1/V2 only; freeze candidates + thresholds + promotion rule
before running; same seed/init/data-order/budget; then ONE final evaluation on untouched
data (shadow test or reserved seeds); no further structure edits against the same test after
that. If no stable 3-seed gain, method-v1 stands and the negative result is reported.
