# CleanOracle

Fidelity label: **diagnostic**.

Source: selection by the injector's corruption tags (no paper).

Strategy name: `clean_oracle`. Implementation: [`method.py`](method.py).

The budget is a uniform draw (default_rng(seed)) from the records whose corruption tag is `high`. Corrupted records fill the budget only when the clean ones run out. On the token-budgeted text track the full order (clean first) is cut to the budget by the track. The row reads the tags, which no other strategy does, so it is never a portfolio member (`DIAGNOSTIC_ONLY` in `omniselect/core/portfolio/membership.py`) and its candidate role in the run record is `diagnostic`. It is a method row of the R5 queue.

Standalone run: `benchmark/MethodsRunScript/run_clean_oracle/`.
