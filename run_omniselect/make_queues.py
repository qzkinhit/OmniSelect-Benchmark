"""Prepare local run queues, using the reported templates when no protocol or seed override is given.

The checked-in templates contain the paper's reported multi-seed runs and recorded settings.
Custom seed or protocol options use the existing experiment presets. Generated files default to
results_and_logs/generated_queues; --output-dir (or --out) chooses another directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omniselect.store.run_record import RunRecord

# (track, dataset, default learner). Small learners first, then text, then native ResNet-18.
SMALL_TASKS = [
    ("vision", "cifar100", "clip_vitb32"), ("vision", "cifar100n", "clip_vitb32"),
    ("vision", "cifar10_clip", "clip_vitb32"), ("timeseries", "ETTh1", "dlinear"),
    ("timeseries", "ETTm1", "dlinear"), ("timeseries", "ETTh2", "dlinear"),
    ("timeseries", "daisy_cstr", "dlinear"), ("timeseries", "daisy_steamgen", "dlinear"),
    ("process", "tep21", "mlp"), ("tabular", "electricity", "tabpfn"),
]
HEAVY_TASKS = [("text", "five_domain", "smollm2_135m"), ("native", "cifar10", "resnet18_scratch")]
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SEEDS = json.loads((REPO_ROOT / "results/paper/seeds_reported.json").read_text())["seeds"]
TASK_LABELS = {
    ("vision", "cifar100"): "C-100", ("vision", "cifar100n"): "C-100N",
    ("vision", "cifar10_clip"): "C-10-CLIP", ("native", "cifar10"): "C-10",
    ("timeseries", "ETTh1"): "ETTh1", ("timeseries", "ETTm1"): "ETTm1",
    ("timeseries", "ETTh2"): "ETTh2", ("timeseries", "daisy_cstr"): "CSTR",
    ("timeseries", "daisy_steamgen"): "Steam", ("process", "tep21"): "TEP21",
    ("tabular", "electricity"): "Elec", ("text", "five_domain"): "Text",
}


def task_seeds(track: str, dataset: str, override: list[int] | None) -> list[int]:
    return REPORT_SEEDS[TASK_LABELS[track, dataset]] if override is None else override


DEFAULT_LEARNER = {track: learner for track, _, learner in SMALL_TASKS + HEAVY_TASKS}

PAPER_BATCHES = {
    "main_v2_3": ("v2_2", "main experiment of the paper, class-conditional influence on classification tracks"),
    "ablation_v2_3": ("v2", "ablation without cooperative candidates, class-conditional influence on classification tracks"),
}
BATCHES = (*PAPER_BATCHES, "main_adapt_v2_2", "robustness")
# Queue key -> batch directory of the run record.
RECORD_BATCH = {
    "main_v2_3": "main",
    "ablation_v2_3": "main_nc",
    "main_adapt_v2_2": "main_adapt",
    "robustness": "robustness",
}
ADAPT_TASKS = [t for t in SMALL_TASKS if t[0] == "timeseries"] + [HEAVY_TASKS[0]]
ADAPT_METHODS = "random,el2n_adapt,grand_adapt,ccs_adapt"

# The native track selects with herding, k-center and k-means on the accelerator (identical records
# for herding and k-center, full-batch Lloyd k-means for coverage, docs/REPRODUCING.md).
NATIVE_FLAGS = " --track-set selection_device=cuda"

INJECTED = [t for t in SMALL_TASKS if t[1] != "cifar100n"]
LEARNER_SWAPS = [
    ("vision", "cifar100", ("dinov2_vits14", "resnet18_scratch")), ("process", "tep21", ("rf", "cnn1d")),
    ("timeseries", "ETTh1", ("chronos_tiny", "chronos_small")),
    ("tabular", "electricity", ("xgboost",)), ("text", "five_domain", ("smollm2_360m",)),
]
FINALISTS = "--set store.per_unit_scope=finalists"
# v2.3: class-conditional influence. Only the classification tracks read these fields in this way; the forecasting
# tracks also read influence_ref_n, so the flags are never added to them.
CLS_TRACKS = {"vision", "process", "tabular", "native"}
V23_FLAGS = " --track-set influence_within_class=true --track-set influence_ref_n=1000000"


def seed_range(spec: str) -> list[int]:
    """Seeds of a spec such as '0-9' or '0,1,2'."""
    out: list[int] = []
    for part in spec.replace(" ", "").split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def driver_line(batch: str, protocol: str, track: str, dataset: str, learner: str, seed: int, root: str,
                extra: str = "") -> str:
    """One driver command with its cell path and job name."""
    cell = RunRecord.cell_path(root, batch, track, dataset, learner, seed)
    learner_flag = "" if learner == DEFAULT_LEARNER[track] else f" --learner {learner}"
    cmd = (f"python -m tracks.common.experiment --track {track} --dataset {dataset}{learner_flag} --seed {seed} "
           f"--protocol {protocol} --batch {batch} --out {root}{extra}")
    tag = batch.replace("/", "_")
    return f"{cmd}   # cell={cell} name={tag}__{track}__{dataset}__{learner}__s{seed}"


def build_paper(batch: str, seeds: list[int] | None, root: str, heavy_seeds: list[int] | None = None) -> list[str]:
    """Lines of one paper batch. Every candidate keeps its per-unit files (store.per_unit_scope=all)."""
    protocol, text = PAPER_BATCHES[batch]
    out = [f"# Custom {batch}: protocol {protocol}, {text}.",
           "# Unspecified seeds use the per-task lists in results/paper/seeds_reported.json."]
    record_batch = RECORD_BATCH[batch]
    for track, dataset, learner in SMALL_TASKS + HEAVY_TASKS:
        extra = NATIVE_FLAGS if track == "native" else ""
        if batch.endswith("v2_3") and track in CLS_TRACKS:
            extra += V23_FLAGS
        for seed in task_seeds(track, dataset, heavy_seeds if (track, dataset, learner) in HEAVY_TASKS else seeds):
            out.append(driver_line(record_batch, protocol, track, dataset, learner, seed, root, extra))
    return out


def build_robustness(protocol: str, seeds: list[int] | None, root: str, v23: bool = True) -> list[str]:
    """Robustness cells: corruption ratio grid, validation contamination, unseen mechanisms, learner swaps.

    Each condition is a sub-batch ``robustness/<condition>`` so that conditions of the same task and
    seed get distinct cells. Per-unit files are kept for finalists only.
    """
    out = [f"# Custom robustness: protocol {protocol}; unspecified seeds use the per-task reported lists.",
           "# corruption ratio grid 0.0 to 0.6 (the track default is 0.4)"]

    def add(condition: str, track: str, dataset: str, learner: str, flags: str) -> None:
        extra = f" {flags} {FINALISTS}".replace("  ", " ").rstrip()
        if v23 and track in CLS_TRACKS:
            extra += V23_FLAGS
        for seed in task_seeds(track, dataset, seeds):
            out.append(driver_line(f"robustness/{condition}", protocol, track, dataset, learner, seed, root,
                                   extra))

    for ratio in ("0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"):
        for track, dataset, learner in (SMALL_TASKS[0], SMALL_TASKS[8], SMALL_TASKS[3]):
            add(f"ratio_{ratio}", track, dataset, learner, f"--set robustness.injection_ratio={ratio}")
    out.append("# validation-label contamination: symmetric 0.1 to 0.5, natural (CIFAR-N labels), prior shift 0.5")
    conditions = [(f"val_sym_{r}", f"--set robustness.val_noise_kind=symmetric --set robustness.val_noise_rate={r}")
                  for r in ("0.1", "0.2", "0.3", "0.4", "0.5")]
    conditions += [("val_natural", "--set robustness.val_noise_kind=natural"),
                   ("val_prior_shift_0.5",
                    "--set robustness.val_noise_kind=prior_shift --set robustness.val_noise_rate=0.5")]
    for name, flags in conditions:
        for track, dataset, learner in (SMALL_TASKS[0], SMALL_TASKS[8]):
            if name == "val_natural" and track != "vision":
                continue
            add(name, track, dataset, learner, flags)
    out.append("# unseen corruption mechanisms on every task with injected corruption")
    for track, dataset, learner in INJECTED:
        add("unseen", track, dataset, learner, "--set robustness.mechanism_set=unseen")
    out.append("# learner swaps")
    for track, dataset, learners in LEARNER_SWAPS:
        for learner in learners:
            add("learner", track, dataset, learner, "")
    return out


def build_adapt(root: str, seeds: list[int] | None, heavy_seeds: list[int] | None) -> list[str]:
    """EL2N, GraNd and CCS task adaptations on forecasting and text, run beside Random without the controller."""
    out = ["# main_adapt: EL2N/GraNd/CCS task adaptations on forecasting and text with their own Random.",
           "# Standalone rows, not members of the adjudicated portfolio."]
    for track, dataset, learner in ADAPT_TASKS:
        for seed in task_seeds(track, dataset, heavy_seeds if track == "text" else seeds):
            out.append(driver_line(RECORD_BATCH["main_adapt_v2_2"], "v2_2", track, dataset, learner, seed, root,
                                   f" --standalone --methods {ADAPT_METHODS}"))
    return out


def build(batch: str, *, seeds: str | None = None, root: str = "results_and_logs", protocol: str = "v2_2",
          robustness_seeds: str | None = None, heavy_seeds: str | None = None, v23: bool = True) -> list[str]:
    """Reported template by default; explicit seed/protocol overrides create a custom preset run."""
    if batch not in BATCHES:
        raise ValueError(f"unknown batch {batch!r}; expected one of {BATCHES}")
    if seeds is None and heavy_seeds is None and robustness_seeds is None and protocol == "v2_2" and v23:
        lines = (Path(__file__).resolve().parent / "queues" / f"{batch}.txt").read_text().splitlines()
        return [line.replace("--out results_and_logs", f"--out {root}")
                    .replace("cell=results_and_logs/", f"cell={root}/") for line in lines]
    small = seed_range(seeds) if seeds is not None else None
    heavy = seed_range(heavy_seeds) if heavy_seeds is not None else None
    if batch in PAPER_BATCHES:
        return build_paper(batch, small, root, heavy)
    if batch == "main_adapt_v2_2":
        return build_adapt(root, small, heavy)
    robust = seed_range(robustness_seeds) if robustness_seeds is not None else small
    return build_robustness(protocol, robust, root, v23)


def main(argv: list[str] | None = None) -> int:
    """Write local queue copies; custom seed/protocol options generate preset-based runs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", "--output-dir", type=Path, default=REPO_ROOT / "results_and_logs/generated_queues")
    parser.add_argument("--batches", nargs="+", default=list(PAPER_BATCHES), choices=BATCHES)
    parser.add_argument("--results-root", default="results_and_logs", help="output root written into every line")
    parser.add_argument("--seeds", default=None, help="custom small-learner seeds; default: reported seeds of each task")
    parser.add_argument("--heavy-seeds", default=None, help="custom text/native seeds; default: reported task seeds")
    parser.add_argument("--protocol", default="v2_2", help="protocol of the robustness batch")
    parser.add_argument("--robustness-seeds", default=None, help="custom robustness seeds; defaults to --seeds or reported task seeds")
    parser.add_argument("--robustness-v2-2-influence", action="store_true",
                        help="write the robustness batch with the earlier influence reference instead of v2.3")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for batch in args.batches:
        lines = build(batch, seeds=args.seeds, root=args.results_root, protocol=args.protocol,
                      robustness_seeds=args.robustness_seeds, heavy_seeds=args.heavy_seeds,
                      v23=not args.robustness_v2_2_influence)
        path = args.out / f"{batch}.txt"
        path.write_text("\n".join(lines) + "\n")
        print(f"wrote {path} ({sum(not x.startswith('#') for x in lines)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
