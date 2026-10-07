#!/bin/bash
# FULL repeated nested eye-grouped CV (primary: eye-grouped; sensitivity: subject-grouped).
# Resumable: just run it again after any interruption (finished models are skipped, permutations resume).
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

cpu_phase () {
  for G in eye subject; do for M in lr rf; do
    echo "== $(date +%T) $M $G"
    python -m src.repeated_cv run --model $M $DATA --out $ROOT/repeated_cv/$G --group_by $G --k $K --repeats $R \
      --n_perm $P --perm_repeats $PR --n_jobs $NJ --rf_trees $TREES >> $ROOT/logs/rcv_${M}_${G}.log 2>&1 || echo "!! FAILED $M $G"
    grep -a "DONE" $ROOT/logs/rcv_${M}_${G}.log | tail -1
  done; done
}
gpu_phase () {
  for G in eye subject; do for M in hybrid_zone hybrid_orig; do
    echo "== $(date +%T) $M $G"
    python -m src.repeated_cv run --model $M $DATA --out $ROOT/repeated_cv/$G --group_by $G --k $K --repeats $R \
      --n_perm $PD --perm_repeats $PR --dl_epochs $EP >> $ROOT/logs/rcv_${M}_${G}.log 2>&1 || echo "!! FAILED $M $G"
    grep -a "DONE" $ROOT/logs/rcv_${M}_${G}.log | tail -1
  done; done
}
if [ "${PAR:-1}" = "1" ]; then cpu_phase & gpu_phase & wait; else cpu_phase; gpu_phase; fi

for G in eye subject; do
  echo; echo "######## AGGREGATE: $G-grouped ########"
  python -m src.repeated_cv aggregate --out $ROOT/repeated_cv/$G $DATA 2>&1 | tail -16
done
echo; echo "ALL DONE $(date). Next: bash scripts/5_package_share.sh  then git add share/ && git commit && git push"
