import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_seed0_wrapper_is_valid_bash():
    subprocess.run(
        ["bash", "-n", str(ROOT / "scripts/run_text_seed0_primary.sh")],
        check=True,
    )


def test_seed0_wrapper_freezes_primary_contract():
    source = (ROOT / "scripts/run_text_seed0_primary.sh").read_text()
    for token in (
        "SEED=0",
        "PASSES=2",
        "STRATIFY=1",
        "INFL_KIND=pplq",
        "TRAIN_MODE=finetune",
        "REPORT_ALL_CANDIDATES=1",
        "LMEVAL_ALL_CANDIDATES=0",
        "SAVE_ALL_CANDIDATE_MODELS=1",
        "influence_only",
        "coverage_text",
        "fixed_fusion",
        "herding_text",
        "density_text",
        "quadmix_pub",
        "dmf_pub",
        'SKIPPED_METHODS="el2n,grand,ccs"',
    ):
        assert token in source
    for excluded in ("el2n_text", "grand_text", "ccs_text", "dsir", "perpcorr"):
        assert excluded not in source
