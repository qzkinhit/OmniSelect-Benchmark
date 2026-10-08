# Controller package

`omniselect` holds the controller and its configuration, and nothing that loads a dataset or a
model. The public entry points are `OmniSelect` and `AdjudicationTask` in
[api/omniselect.py](api/omniselect.py) and `OmniSelectConfig` in [config/config.py](config/config.py).

| Call | Role |
|---|---|
| `OmniSelectConfig.preset(protocol, **overrides)` | Resolved configuration of `canonical`, `v2`, `v2_1` or `v2_2` (the protocol of the paper). Dotted keys override single fields |
| `OmniSelect(config, grid).adjudicate(task)` | Algorithm 1 on one pool. Returns an `Election` with every candidate, the precheck and the gate decision |
| `adjudicate(...)` in `core/adjudication/controller.py` | The same algorithm as a function of channel scores, references and utility callbacks |
| `AdaptiveController` | The 2026-07 controller interface, implemented by `adjudicate` under the canonical preset |
| `select_pool(records, budget)` | Signal fusion and budgeted selection without a learner |

| Package | Responsibility |
|---|---|
| `core/signals` | Authenticity, influence, redundancy, kNN agreement and novelty, gradient alignment, learned cleanliness |
| `core/selection` | Budget selector, fusion grid, cooperative candidates, relaxed quadratic selection, console blending, text token orders |
| `core/adjudication` | Validation splits, train-once cache, screening, synthesis, precheck, controller |
| `core/gates` | Margin, argmax, Hoeffding LCB, paired bootstrap and betting e-process gates, bounded per-unit surrogates |
| `core/portfolio` | Strategy registry, per-track membership tables with exclusion reasons, candidate roles |
| `store` | Run-record writer, metrics, recompute, replay, batch index |
| `tools`, `utils` | Pairing RNG helpers, manifests, hashing, I/O, logging |
| `tests` | CPU tests of every module and small end-to-end cells |

The controller never sees a learner. Tracks pass utility callbacks per split and per-unit
surrogates, and the driver in `tracks/common/experiment.py` connects them to the train-once cache
and writes the run record ([docs/RUN_RECORD.md](../docs/RUN_RECORD.md)). Selection methods register
themselves from `benchmark/Methods/`. [docs/CODE_MAP.md](../docs/CODE_MAP.md) gives a reading order.
