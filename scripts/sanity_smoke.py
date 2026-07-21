"""Pure-CPU end-to-end sanity smoke (no torch, no downloads, seconds).

Builds a tiny standardized pool (general/math/code with planted near-duplicates),
runs the full system path (signals -> Multi-Actor fusion -> budget selection), and
contrasts four strategies the way the paper's Instance 1 does:

    All  vs  Random  vs  Influence-only(Top-K)  vs  OmniSelect

Prints a diagnostics table and writes a manifest, then asserts basic invariants.
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from mmdataselect.api import select_pool  # noqa: E402
from mmdataselect.budget import Budget  # noqa: E402
from mmdataselect.fusion.console import MultiActorConsole  # noqa: E402
from mmdataselect.selectors.base import TopKSelector  # noqa: E402
from mmdataselect.signals import InfluenceSignal, RedundancySignal, set_redundancy  # noqa: E402
from mmdataselect.utils.manifest import write_manifest  # noqa: E402
from tools.standardize.make_demo import build  # noqa: E402


def _row(name, records, idx, influence):
    sel = [records[i] for i in idx]
    return {
        "strategy": name,
        "n": len(idx),
        "set_redundancy": round(set_redundancy(sel), 3),
        "mean_influence": round(float(np.mean(influence[idx])) if len(idx) else 0.0, 3),
    }


def main() -> int:
    records = build(60)
    n = len(records)
    k = Budget("fraction", 0.5).resolve(n)

    # shared influence scores (model-free CPU proxy) for fair contrast
    console = MultiActorConsole([("redundancy", RedundancySignal()), ("influence", InfluenceSignal())])
    influence = console.actor_scores(records)[console.names.index("influence")]

    rng = np.random.default_rng(0)
    rows = [
        _row("All (no selection)", records, list(range(n)), influence),
        _row("Random", records, list(rng.choice(n, size=k, replace=False)), influence),
        _row("Influence-only (Top-K)", records, TopKSelector().select(records, influence, k), influence),
    ]
    res = select_pool(records, Budget("fraction", 0.5), lam=0.5, method="greedy", seed=0)
    rows.append(_row("OmniSelect", records, res.selected_idx, influence))

    print(f"\nPool: {n} records | budget k={k} | actor weights={res.weights}\n")
    print(f"{'strategy':<26}{'n':>4}{'set_redundancy↓':>18}{'mean_influence↑':>18}")
    for r in rows:
        print(f"{r['strategy']:<26}{r['n']:>4}{r['set_redundancy']:>18}{r['mean_influence']:>18}")

    out_dir = os.path.join(_REPO, "outputs", "sanity_smoke")
    write_manifest(
        out_dir,
        experiment_id="sanity_smoke",
        method="mmdataselect",
        n_total=n,
        selected_ids=res.selected_ids,
        selected_rows=[records[i].to_dict() for i in res.selected_idx],
        extra={"diagnostics": res.diagnostics, "actor_weights": res.weights},
    )

    # invariants
    assert res.diagnostics["n_selected"] == k, "selected count must equal the budget"
    assert os.path.exists(os.path.join(out_dir, "manifests", "manifest.json")), "manifest must be written"
    mm = rows[-1]["set_redundancy"]
    infl_only = rows[2]["set_redundancy"]
    print(
        f"\nSANITY OK | OmniSelect set_redundancy={mm} vs Influence-only={infl_only} "
        f"(diversity term {'reduces' if mm <= infl_only else 'does not reduce'} redundancy)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
