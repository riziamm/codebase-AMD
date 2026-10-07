#!/bin/bash
# Step D: gather all holdout predictions (ML + DL) and compute the final metrics table.
# Then copy the aggregate results into share/ for pushing.
set -u
PRED="reruns_eyegrouped/preds_holdout"
rm -rf "$PRED"
python -m src.collect_preds \
  --dl_runs reruns_eyegrouped/dl/EG_bin_* \
  --ml_dirs reruns_eyegrouped/ml_eval \
  --out "$PRED"

python src/evaluate_all.py --pred_dir "$PRED" \
  --manifest reruns_eyegrouped/split/split_manifest.csv \
  --out reruns_eyegrouped/results_holdout.csv

mkdir -p share/results
cp reruns_eyegrouped/results_holdout*.csv share/results/
echo "DONE. Results copied to share/results/  (aggregates only)"
