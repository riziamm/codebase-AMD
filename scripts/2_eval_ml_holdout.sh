#!/bin/bash
# Step B: evaluate every saved model of every experiment on its own test_test.pkl.
# Writes per-row predictions to $ROOT/ml_eval/predictions/
# Usage:  bash scripts/2_eval_ml_holdout.sh $ROOT/ml_batches/batch_*
set -u
ROOT="${ROOT:-reruns_eyegrouped}"   # output root (pilot uses ROOT=reruns_pilot)
OUT="$ROOT/ml_eval"
BATCH_DIRS=("$@")
[ ${#BATCH_DIRS[@]} -eq 0 ] && BATCH_DIRS=($ROOT/ml_batches/batch_*)

n=0
for B in "${BATCH_DIRS[@]}"; do
  for EXP in "$B"/experiment_*; do
    PKL="$EXP/data/test_test.pkl"
    MODELS=("$EXP"/models/*.pkl)
    if [ ! -f "$PKL" ]; then echo "skip (no test_test.pkl): $EXP"; continue; fi
    if [ ! -f "${MODELS[0]}" ]; then echo "skip (no models): $EXP"; continue; fi
    echo ">> $EXP  (${#MODELS[@]} models)"
    python -m src.main --mode evaluate --model_paths "${MODELS[@]}" \
      --eval_data_path "$PKL" --report_dir "$OUT" && n=$((n+1)) || echo "   !! FAILED: $EXP"
  done
done
echo "DONE: evaluated $n experiment(s). Prediction files:"
ls "$OUT"/predictions/ 2>/dev/null | wc -l
