# AuthenticityOnly

Fidelity label: **built-in (OmniSelect signal)**.

Source: OmniSelect authenticity channel.

Strategy names: `auth_only, auth2_only, auth3_only, auth_bottom`. Implementation: [`method.py`](method.py).

auth_only keeps the k records with the highest kNN label agreement on the track representation. auth2_only uses min(rank(out-of-fold probe agreement), rank(kNN agreement)). On forecasting it is min(rank(lag-1 autocorrelation), rank(kNN inlierness)). auth3_only adds the kNN inlier arm (process and tabular). auth_bottom keeps the k lowest-authenticity records and is the rank-inverting control.

Deviations: Authenticity-only is reference-eligible and appears in the leaderboards. It is not an independent main-table row.

Standalone run: `benchmark/MethodsRunScript/` (see its README). Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
