#!/usr/bin/env python3
"""Rank all displayed methods on their common task intersection.

Means remain unrounded; exact ties receive average ranks. The reported ranking
is descriptive and includes explicitly labelled task adaptations.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from scipy.stats import rankdata

ROOT = Path.cwd()
MAIN_FILE = ROOT / "paper_main_stats.json"
NOCOOP_FILE = ROOT / "paper_nocoop_stats.json"
OUTPUT = ROOT / "table_method_ranks.json"
COLS = ["C-100", "C-100N", "C-10", "ETTh1", "ETTm1", "ETTh2", "CSTR", "Steam", "TEP21", "Elec", "Text"]
LOWER = {"ETTh1", "ETTm1", "ETTh2", "CSTR", "Steam", "Text"}
METHODS = [
    ("random", "Random"), ("influence_only", "Infl-only"), ("coreset", "Coverage"),
    ("mmdataselect", "Fixed fusion"), ("auth_only", "Auth-only"), ("herding", "Herding"),
    ("el2n", "EL2N / EL2N-adapt"), ("grand", "GraNd / GraNd-adapt"), ("ccs", "CCS / CCS-adapt"), ("density", "Density"),
    ("quadmix_pub", "QuaDMix-pub"), ("dmf_pub", "DMF-pub"),
    ("nocoop", "OmniSelect w/o cooperation"), ("omni", "OmniSelect"),
]
TEXT_ALIAS = {"herding": "herding_text", "density": "density_text", "coreset": "coverage_text"}


def hashed(path):
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def task_ranks(values, lower):
    """Friedman-style ranks from unrounded means, exact ties use average ranks."""
    signed = [v if lower else -v for v in values.values()]
    return dict(zip(values, map(float, rankdata(signed, method="average"))))


def main():
    M = json.loads(MAIN_FILE.read_text())["tasks"]
    N = json.loads(NOCOOP_FILE.read_text())["tasks"]
    values = {}
    for task in COLS:
        values[task] = {}
        for key, _ in METHODS:
            source = N if key == "nocoop" else M
            raw_key = "mmds_adapt" if key in {"omni", "nocoop"} else key
            rows = source.get(task, {}).get("rows", {})
            if task == "Text" and raw_key not in rows:
                raw_key = TEXT_ALIAS.get(raw_key, raw_key)
            row = rows.get(raw_key)
            if row is None:
                continue
            value = row["mean"]
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite observed mean {task}/{key}")
            seed_values = row.get("per_seed", {})
            if seed_values and abs(mean(seed_values.values()) - value) > 1e-12:
                raise ValueError(f"Stored mean disagrees with seed data {task}/{key}")
            values[task][key] = value
        if values[task] and "nocoop" in values[task]:
            if sorted(M[task]["seeds"]) != sorted(N[task]["seeds"]):
                raise ValueError(f"Main/no-cooperation seed scope differs on {task}")
    completed = [task for task in COLS if values[task]]
    common = [task for task in completed if len(values[task]) == len(METHODS)]
    if not common:
        raise ValueError("No completed tasks shared by all 14 methods")
    per_task = {task: task_ranks(values[task], task in LOWER) for task in completed}
    average = {key: mean(per_task[task][key] for task in common) for key, _ in METHODS}
    # Competition ranks retain genuine ties, without using table row order.
    overall = dict(zip(average, map(int, rankdata(list(average.values()), method="min"))))
    rows = [{"method": key, "display_name": label, "circle_rank": overall[key],
             "mean_task_rank": average[key], "ranking_tasks": common,
             "all_available_tasks": [task for task in completed if key in values[task]],
             "per_common_task_rank": {task: per_task[task][key] for task in common}}
            for key, label in METHODS]
    # Keep this alternative only to expose why it is not the recommended
    # all-method ordering: both task mix and competitor count vary by method.
    conditional = {key: mean(per_task[task][key] for task in completed if key in per_task[task])
                   for key, _ in METHODS}
    out = {
        "recommended_scheme": "mean rank on the common completed-task intersection",
        "ranking_tasks": common, "n_methods": len(METHODS), "n_ranking_tasks": len(common),
        "method_to_circle_rank": overall, "rows_in_manuscript_order": rows,
        "rows_sorted_by_rank": sorted(rows, key=lambda row: row["circle_rank"]),
        "conventions": {"within_task": "unrounded seed mean; metric direction respected; average ranks for exact ties",
                        "aggregation": "arithmetic mean of ranks over the same tasks for every method",
                        "circle_rank": "ascending average rank; competition rank for ties",
                        "missing": "never zero-filled or assigned a worst rank; all methods use the common intersection",
                        "no_cooperation": "included as one of the 14 ranked methods",
                        "oracle_and_verdict": "not methods and not ranked",
                        "inference": "descriptive label; no new significance claim"},
        "scope_limit": "All methods use the same task set. Task adaptations and independent text runs contribute descriptive mean values, without adding paired inference against historical OmniSelect.",
        "not_recommended_available_case_average": {
            "mean_ranks": conditional,
            "competitors_per_task": {task: len(values[task]) for task in completed},
            "why_not": "Use the shared-task intersection even if future task coverage changes.",
        },
        "caption_en": f"Circled numbers order all {len(METHODS)} methods by mean within-task rank across their {len(common)} shared tasks; exact ties receive average ranks. These are descriptive ranks, including task adaptations.",
        "caption_zh": f"圆圈数字按全部{len(METHODS)}种方法在共同覆盖的{len(common)}个任务上的平均秩排序；单个任务内并列取平均秩。排名为包含任务适配结果的描述统计。",
        "provenance": [hashed(MAIN_FILE), hashed(NOCOOP_FILE),
                       hashed(Path(__file__).resolve().parent / "make_main_table.py")],
    }
    OUTPUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    for row in rows:
        print(f"{row['display_name']:28s} {row['circle_rank']:2d}  mean rank {row['mean_task_rank']:.3f}")
    print("Shared tasks:", ", ".join(common))
    print("Recomputed all displayed method ranks. Wrote", OUTPUT)


if __name__ == "__main__":
    main()
