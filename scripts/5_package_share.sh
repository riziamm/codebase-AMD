#!/bin/bash
# Step E: copy ONLY aggregate results into share/ (safe to push).  Predictions are OFF by default.
#   bash scripts/5_package_share.sh            # aggregates only
#   PREDS=1 bash scripts/5_package_share.sh    # + holdout predictions (y_true,y_prob only; 18 rows/model, no IDs)
set -u
ROOT="${ROOT:-reruns_eyegrouped}"
OUT="share/run_$(date +%Y%m%d)"
mkdir -p "$ROOT/env"
pip freeze > "$ROOT/env/pip_freeze.txt" 2>/dev/null
git rev-parse HEAD > "$ROOT/env/commit.txt" 2>/dev/null
nvidia-smi -L > "$ROOT/env/gpu.txt" 2>/dev/null
FLAG=""; [ "${PREDS:-0}" = "1" ] && FLAG="--include_preds"
python -m src.package_share --root "$ROOT" --out "$OUT" $FLAG
echo; echo "Next:  git status   (only share/ should appear)"
echo "       git add share/ && git commit -m 'results' && git push origin eye-grouped-rerun"
