#!/bin/bash
# FULL repeated nested eye-grouped CV (primary: eye-grouped; sensitivity: subject-grouped).
# Resumable: run it again after any interruption (finished models are skipped, permutations resume).
#   nohup bash scripts/7_rcv_full.sh > reruns_eyegrouped/logs/rcv_full.log 2>&1 &
#   grep -a "DONE\|==" reruns_eyegrouped/logs/rcv_full.log        # progress
# PAR=1 (default) runs CPU models and GPU models at the same time; PAR=0 runs them one after another.
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # A6000 (Blackwell unsupported by this PyTorch)
export OMP_NUM_THREADS=1
ROOT="${ROOT:-reruns_eyegrouped}"
SPLIT=reruns_eyegrouped/split
EP="${RCV_EPOCHS:-80}"; TREES="${RCV_TREES:-300}"; K="${RCV_K:-5}"; R="${RCV_R:-10}"; P="${RCV_PERM:-200}"; PD="${RCV_PERM_DL:-100}"; PR="${RCV_PERM_REPEATS:-2}"
NJ=$(nproc 2>/dev/null || echo 8); [ "$NJ" -gt 16 ] && NJ=16
DATA="--data $SPLIT/train_mpod.csv $SPLIT/test_mpod.csv"
mkdir -p "$ROOT/logs"
python -c "import torch;print('DL GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')" 2>&1 | grep "DL GPU"

TASKS="${TASKS:-any early advanced}"   # any = AREDS 2-4 vs 1 (paper task); early = 2 vs 1; advanced = 3-4 vs 1
cpu_phase () {
  for T in $TASKS; do for G in eye subject; do for M in lr rf; do
    TAG="${G}_${T}_all_allrings"; echo "== $(date +%T) $M $TAG"
    python -m src.repeated_cv run --model $M --task $T $DATA --out $ROOT/repeated_cv/$TAG --group_by $G --k $K --repeats $R \
      --n_perm $P --perm_repeats $PR --n_jobs $NJ --rf_trees $TREES >> $ROOT/logs/rcv_${M}_${TAG}.log 2>&1 || echo "!! FAILED $M $TAG"
    grep -a "DONE\|skip" $ROOT/logs/rcv_${M}_${TAG}.log | tail -1
  done; done; done
}
gpu_phase () {
  for T in $TASKS; do for G in eye subject; do for M in hybrid_zone hybrid_orig; do
    TAG="${G}_${T}_all_allrings"; echo "== $(date +%T) $M $TAG"
    python -m src.repeated_cv run --model $M --task $T $DATA --out $ROOT/repeated_cv/$TAG --group_by $G --k $K --repeats $R \
      --n_perm $PD --perm_repeats $PR --dl_epochs $EP >> $ROOT/logs/rcv_${M}_${TAG}.log 2>&1 || echo "!! FAILED $M $TAG"
    grep -a "DONE\|skip" $ROOT/logs/rcv_${M}_${TAG}.log | tail -1
  done; done; done
}
if [ "${PAR:-1}" = "1" ]; then cpu_phase & gpu_phase & wait; else cpu_phase; gpu_phase; fi

for D in $ROOT/repeated_cv/*_*_all_allrings $ROOT/repeated_cv/eye_*_*; do
  [ -d "$D" ] || continue
  python -m src.repeated_cv aggregate --out $D $DATA > $D/aggregate.log 2>&1 || echo "!! aggregate failed: $D"
done
python - <<PY
import pandas as pd, glob
t = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("$ROOT/repeated_cv/*/summary.csv"))], ignore_index=True)
t = t[["task","groups","rings","group_by","model","n_eyes","AUC","AUC_lo","AUC_hi","p_above","p_below","verdict"]]
t.to_csv("$ROOT/repeated_cv/OVERVIEW.csv", index=False)
pd.set_option("display.width", 220); print(t.round(3).to_string(index=False))
PY
echo; echo "ALL DONE $(date). Next: bash scripts/5_package_share.sh  then git add share/ && git commit && git push"
