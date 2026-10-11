#!/bin/bash
# Copy ONLY Paper 2 aggregates into share/paper2_run_<date>/ (safety-scanned; no row-level files).
set -u
ROOT="${ROOT:-reruns_paper2}"; OUT="share/paper2_run_$(date +%Y%m%d)"
mkdir -p "$ROOT/env"
pip freeze > "$ROOT/env/pip_freeze.txt" 2>/dev/null; git rev-parse HEAD > "$ROOT/env/commit.txt" 2>/dev/null
python -m src.package_share --root "$ROOT" --out "$OUT" || exit 1
echo; echo "Check: git status --short   (only $OUT/ should be new)"
