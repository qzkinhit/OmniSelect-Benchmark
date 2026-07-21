#!/usr/bin/env bash
# Create a local virtualenv and install OmniSelect (editable) with all extras.
#
#   bash scripts/setup_env.sh            # venv at .venv, full [train,eval,dev] stack
#   VENV=.venv311 PYTHON=python3.11 bash scripts/setup_env.sh
#   EXTRAS="" bash scripts/setup_env.sh  # core-only (enough for import + CPU smoke)
#
# After it finishes:
#   source .venv/bin/activate
#   python scripts/sanity_smoke.py
set -euo pipefail

# Repo root = parent of this script's directory (scripts/setup_env.sh -> <repo>).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
cd "${REPO}"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"
# Default to the full local workstation stack; override with EXTRAS="" for core-only.
EXTRAS="${EXTRAS:-[train,eval,dev]}"

echo "==> repo:    ${REPO}"
echo "==> python:  ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo "==> venv:    ${VENV}"
echo "==> extras:  ${EXTRAS:-<core only>}"

# 1) Create the virtualenv if it does not already exist.
if [ ! -d "${VENV}" ]; then
  "${PYTHON}" -m venv "${VENV}"
fi

# 2) Activate and upgrade the installer toolchain.
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# 3) Editable install of the package (+ optional extras).
python -m pip install -e ".${EXTRAS}"

echo
echo "==> done. Activate with:  source ${VENV}/bin/activate"
echo "==> sanity check:         python scripts/sanity_smoke.py"
