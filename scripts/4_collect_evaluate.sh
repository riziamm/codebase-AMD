#!/bin/bash
# Step D: gather all holdout predictions (ML + DL) and compute the final metrics table.
# Then copy the aggregate results into share/ for pushing.
set -u
ROOT="${ROOT:-reruns_eyegrouped}"   # output root (pilot uses ROOT=reruns_pilot)
PRED="$ROOT/preds_holdout"
rm -rf "$PRED"
python -m src.collect_preds \
  --dl_runs $ROOT/dl/EG_bin_* \
  --ml_dirs $ROOT/ml_eval \
  --out "$PRED"

python src/evaluate_all.py --pred_dir "$PRED" --n_boot "${NBOOT:-2000}" \
  --manifest reruns_eyegrouped/split/split_manifest.csv \
  --ref_pattern "${REF:-EG_bin_all_tuned_s42__CNNTransformer_parallel}" \
  --out $ROOT/results_holdout.csv

if [ "$ROOT" = "reruns_eyegrouped" ]; then
  mkdir -p share/results
  cp $ROOT/results_holdout*.csv share/results/
  echo "DONE. Results copied to share/results/  (aggregates only)"
else
  echo "DONE (pilot). Results in $ROOT/results_holdout.csv  (not copied to share/)"
fi
