#!/bin/bash
# Step A: turn the frozen holdout CSV into test_test.pkl for EVERY experiment in the given batch folders.
# Usage:  bash scripts/1_create_holdout.sh reruns_eyegrouped/ml_batches/batch_*
# (no args = all batches under reruns_eyegrouped/ml_batches)
set -u
HOLDOUT_CSV="reruns_eyegrouped/split/test_mpod.csv"
BATCH_DIRS=("$@")
[ ${#BATCH_DIRS[@]} -eq 0 ] && BATCH_DIRS=(reruns_eyegrouped/ml_batches/batch_*)
[ -f "$HOLDOUT_CSV" ] || { echo "ERROR: $HOLDOUT_CSV not found"; exit 1; }

n=0
for B in "${BATCH_DIRS[@]}"; do
  for EXP in "$B"/experiment_*; do
    [ -d "$EXP" ] || continue
    echo ">> $EXP"
    python -m src.main --mode preprocess_holdout \
      --holdout_csv_path "$HOLDOUT_CSV" --training_report_dir "$EXP" \
      && [ -f "$EXP/data/test_test.pkl" ] && n=$((n+1)) || echo "   !! FAILED: $EXP"
  done
done
echo "DONE: test_test.pkl created for $n experiment(s)"
