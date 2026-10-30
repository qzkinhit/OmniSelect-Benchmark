"""Compare all rebuilt tables and figure statistics with their published values."""
import json
import math
import sys
from pathlib import Path

TABLES = (
    "paper_main_stats.json", "paper_nocoop_stats.json", "paper_main_agg.json", "paper_nocoop_agg.json",
    "paper_seeds.json", "table_method_ranks.json", "scale_up_stats.json", "main_table_body.tex",
    "paper_numbers.json", "adoption_numbers.json", "robust_cells_view.json", "robustness_summary.json",
    "validation_noise_plot_data.json", "abl_v23_summary.json", "validation_size.json",
    "paired_diagnostics.json", "table4_admission.json", "adjudication_by_task.json",
)


def differences(expected, actual, path):
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            yield f"{path}: keys differ (missing={sorted(expected.keys() - actual.keys())}, extra={sorted(actual.keys() - expected.keys())})"
        for key in expected.keys() & actual.keys():
            yield from differences(expected[key], actual[key], f"{path}/{key}")
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            yield f"{path}: length {len(expected)} != {len(actual)}"
        for i, (a, b) in enumerate(zip(expected, actual)):
            yield from differences(a, b, f"{path}/{i}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if not math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12):
            yield f"{path}: {expected} != {actual}"
    elif expected != actual:
        yield f"{path}: {expected!r} != {actual!r}"


def main():
    pub, new = map(Path, sys.argv[1:3])
    errors = []
    for name in TABLES:
        if not (pub / name).is_file() or not (new / name).is_file():
            errors.append(f"missing {name}")
            continue
        a, b = (p.joinpath(name).read_text() for p in (pub, new))
        if name.endswith('.json'):
            errors.extend(differences(json.loads(a), json.loads(b), name))
        elif a != b:
            errors.append(f"differs {name}")
    for error in errors[:50]:
        print(error)
    print(f"{len(errors)} differences" if errors else f"all {len(TABLES)} rebuilt tables match")
    sys.exit(bool(errors))


if __name__ == '__main__':
    main()
