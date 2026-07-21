# Ablation master table (2026-07-16, rev3) — every design component with its MEASURED conclusion

Each row: what is ablated, the evidence log, the one-line conclusion AS MEASURED, and an
evidence_level grading (audit item 6):
  **DCA** = direct controlled ablation (remove/vary the component, same seeds, measured delta)
  **DER** = derived comparison (evidence from related runs, not a dedicated remove-one run)
  **QUAL** = qualitative/mechanistic explanation only
Only DCA rows support causal component claims. DER/QUAL rows are marked as needing a direct
run against the FROZEN final method (with 3-seed mean+-std/CI, exit codes, result paths)
before any "component is necessary" sentence enters the paper.

| # | Component ablated | Setting | Evidence | evidence_level | Conclusion (as measured) |
|---|---|---|---|---|---|
| 1 | Authenticity-as-gate vs as-ranking | gate on/off | method chapter + CIFAR-10/100 flip | **DER** (needs direct run) | Ranking by authenticity contracts the distribution and hurts coverage (DemandClean over-repair analog). Pending a dedicated gate-vs-rank 3-seed run before causal wording. |
| 2 | Switch margin tau | tau in {0, 0.015, 0.03} | ablation_suite.log | **DCA** | Picks and results bit-for-bit unchanged on vision+TEP; the single hyperparameter is insensitive, so the gate is not a tuned knob. Safe. |
| 3 | Validation size n_v | n_v in {200,400,800,1600} | ablation_suite.log | **DCA** | Controller 0.423->0.434 monotone, seed std 0.007->0.001; below pure auth at tiny n_v, clearly above at large n_v. Confirms Thm 4's n_v prediction. |
| 4 | V1/V2 construction/adjudication split | on/off | split_protocol_3seed.log | **DCA** | Without the split the winner's-curse inflates image 0.432->real 0.428; the split restores Thm 4's independence premise. |
| 5 | Self-improving candidates | on/off | split_protocol + canonical | **DER** (needs 3-seed diff+CI) | Adopted on some process/time seeds (+0.002 TEP), tie elsewhere; a dedicated on/off 3-seed delta with CI is still required before claiming "free upside". |
| 6 | Noise-injection ratio | 20% / 40% / 60% | noise_ratio_ablation_3seed.log | **DCA** | Controller stays best-or-tied and never worst across ratios; the pool ratio is not a tuned advantage. |
| 7 | Channel-drop (reduced-portfolio) | drop auth / infl / red | channel_drop_ablation_3seed.log | **DCA** | Dropping any one channel degrades the controller; each channel carries non-redundant signal. |
| 8 | Successive-halving prescreen | SH on/off | sh_controller_vision.log | **DCA** | Same strategy + same test number as full-fidelity on 3 seeds, fusion-grid evals 42->4. Optional efficiency, guarantee intact. |
| 9 | Validation contamination robustness | rho_v in {0,10,20,30}% | val_contamination.log | **DCA** | Image selection unchanged; TEP drifts <=0.004; ranking never flips (Prop 6 gap condition). |
| 10 | Regularized switch vs raw argmax | fallback on/off | canonical picks | **DER** (needs direct run) | Observed: on insignificant-gain modalities the controller falls back to the best reference (bit-identical). A dedicated raw-argmax arm is still required for the causal "prevents noise-driven switches" claim. |

Direct-run backlog before final freeze (rows 1, 5, 10): gate-vs-rank arm, self-improving
on/off 3-seed delta with CI, raw-argmax arm. To run against the frozen final method together
with the component-interaction pass (audit item 6), then this table graduates from draft.

## Negative-result ablations (documented, NOT in main results)

| Variant | Verdict | Log/doc |
|---|---|---|
| GRPO/ES policy search | bit-for-bit tie all 5 datasets -> DROP | docs/negative_result_grpo_search.md |
| Rolling / recent-val | lost on target ETTh1 + fresh seeds -> DROP | docs/negative_result_recent_val.md |
| Robust worst-segment adjudication | lost on ETTh1 + held-out -> DROP | docs/negative_result_robust_adjudication.md |
| method-v2 diversity-regularized vote-ensemble | not picked by any seed, metric in noise -> DROP | docs/method_v2_diagnostic.md + method_v2_compare log |
| method-v2c complementarity-aware vote | (running) keep-if-wins pending | method_v2c_compare log |

Every negative result is kept as an ablation: it proves the corresponding design choice
(measured selection over search, fresh-split over rolling, average over worst-segment, base
vote-ensemble over diversity-regularized) is the right one, not an omission.
