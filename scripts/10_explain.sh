#!/bin/bash
# SHAP explanations of the out-of-fold models (RF = main text; zone-hybrid = supplement; LR = sanity check).
# Tasks with signal only (advanced, any). Resumable per run (skips finished). ~30-60 min. GPU used for the hybrid.
#   nohup bash scripts/10_explain.sh > reruns_eyegrouped/logs/explain.log 2>&1 &
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1
ROOT="${ROOT:-reruns_eyegrouped}"; SPLIT=reruns_eyegrouped/split
DATA="--data $SPLIT/train_mpod.csv $SPLIT/test_mpod.csv"
for T in ${TASKS:-advanced any}; do
  for M in rf lr hybrid_zone; do
    O=$ROOT/explain/${T}_${M}
    if [ -f $O/stability.json ]; then echo "[skip] $T $M"; continue; fi
    EXTRA="--repeats ${EXPL_R:-10}"; [ $M = hybrid_zone ] && EXTRA="--repeats ${EXPL_R_DL:-5} --nsamples ${NSAMPLES:-200}"
    echo "== $(date +%T) explain $T $M"
    python -m src.explain --model $M --task $T $DATA --out $O $EXTRA >> $ROOT/logs/explain_${T}_${M}.log 2>&1 || echo "!! FAILED $T $M"
    grep -a "DONE" $ROOT/logs/explain_${T}_${M}.log | tail -1
  done
done
echo "ALL DONE $(date). Next: bash scripts/5_package_share.sh && bash scripts/sync.sh 'explain'"
