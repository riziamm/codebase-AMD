#!/bin/bash
# Diagnostics (model-free) + stage-specific and ring-specific repeated nested CV.  CPU only. Resumable.
#   nohup bash scripts/8_diagnostics.sh > reruns_eyegrouped/logs/diagnostics.log 2>&1 &
#   grep -a "==\|DONE" reruns_eyegrouped/logs/diagnostics.log          # progress
# Optional inputs (from: python -m src.mat_to_csv ... --out data/export):
#   data/export/zone_rings.csv  (ring map from d.p.ring)    data/export/ofa44.csv  (44-region OFA)
set -u
export OMP_NUM_THREADS=1
ROOT="${ROOT:-reruns_eyegrouped}"
SPLIT=reruns_eyegrouped/split
DATA="--data $SPLIT/train_mpod.csv $SPLIT/test_mpod.csv"
R="${RCV_R:-10}"; P="${RCV_PERM:-200}"; TREES="${RCV_TREES:-300}"; NB="${DIAG_BOOT:-1000}"
NJ=$(nproc 2>/dev/null || echo 8); [ "$NJ" -gt 16 ] && NJ=16
mkdir -p "$ROOT/logs"
EXTRA=""
[ -f data/export/zone_rings.csv ] && EXTRA="$EXTRA --rings data/export/zone_rings.csv"
[ -f data/export/ofa44.csv ] && EXTRA="$EXTRA --ofa44 data/export/ofa44.csv"

echo "== $(date +%T) A-C univariate diagnostics"
python -m src.diagnostics $DATA --out $ROOT/diagnostics --n_boot $NB $EXTRA > $ROOT/logs/diag_univariate.log 2>&1 \
  && echo "   DONE diagnostics -> $ROOT/diagnostics/REPORT.md" || echo "!! FAILED diagnostics (see $ROOT/logs/diag_univariate.log)"

run () {  # run <task> <model> <groups> <rings>
  local T=$1 M=$2 G=$3 RG=$4
  local TAG="eye_${T}_$( [ "$G" = all ] && echo all || echo ${G//,/-} )_$( [ "$RG" = all ] && echo allrings || echo ${RG//,/-} )"
  echo "== $(date +%T) $TAG $M"
  python -m src.repeated_cv run --model $M --task $T --groups $G --rings $RG $DATA --out $ROOT/repeated_cv/$TAG \
    --group_by eye --repeats $R --n_perm $P --perm_repeats 2 --n_jobs $NJ --rf_trees $TREES >> $ROOT/logs/rcv_$TAG.log 2>&1 \
    || echo "!! FAILED $TAG $M"
  grep -a "DONE" $ROOT/logs/rcv_$TAG.log | tail -1
}
# D1: stage-specific (all features)
for T in early advanced; do for M in lr rf; do run $T $M all all; done; done
# D2: ring-specific (outer-periphery hypothesis), all three tasks
for T in any early advanced; do for RG in centre middle outer; do run $T lr all $RG; done; done
# D3: functional (OFA) only vs structural (MPOD) only
for T in early advanced; do run $T lr Del,Amp all; run $T lr mean,median,std,iqr,idr,skew,kurt all; done

echo; echo "######## AGGREGATE ########"
for D in $ROOT/repeated_cv/eye_*_*; do
  [ -d "$D" ] || continue
  python -m src.repeated_cv aggregate --out $D $DATA > $D/aggregate.log 2>&1 || echo "!! aggregate failed: $D"
done
python - <<PY
import pandas as pd, glob
fs = sorted(glob.glob("$ROOT/repeated_cv/*/summary.csv"))
t = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
t = t[["task","groups","rings","group_by","model","n_eyes","AUC","AUC_lo","AUC_hi","p_above","p_below","verdict"]]
t.to_csv("$ROOT/repeated_cv/OVERVIEW.csv", index=False)
pd.set_option("display.width", 220); print(t.round(3).to_string(index=False))
PY
echo; echo "ALL DONE $(date). Next: bash scripts/5_package_share.sh  then git add share/ && git commit && git push"
