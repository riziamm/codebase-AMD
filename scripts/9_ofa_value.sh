#!/bin/bash
# Pre-specified OFA analyses (docs/ANALYSIS_PLAN_OFA.md): A incremental value of Del/Amp over MPOD,
# B leave-one-group-out importance, C structure-function coupling, D functional signal in early AMD.
# CPU only, resumable (finished runs are skipped). ~3 h.
#   nohup bash scripts/9_ofa_value.sh > reruns_eyegrouped/logs/ofa_value.log 2>&1 &
#   grep -a "==\|DONE\|skip" reruns_eyegrouped/logs/ofa_value.log | tail
set -u
export OMP_NUM_THREADS=1
ROOT="${ROOT:-reruns_eyegrouped}"; RCV=$ROOT/repeated_cv; OUT=$ROOT/ofa_value
SPLIT=reruns_eyegrouped/split; DATA="--data $SPLIT/train_mpod.csv $SPLIT/test_mpod.csv"
R="${RCV_R:-10}"; P="${RCV_PERM:-200}"; TREES="${RCV_TREES:-300}"; NB="${CMP_BOOT:-2000}"
TASKS="${TASKS:-any early advanced}"; MODELS="${MODELS:-lr rf}"
NJ=$(nproc 2>/dev/null || echo 8); [ "$NJ" -gt 16 ] && NJ=16
MPOD="mean,median,std,iqr,idr,skew,kurt"; BASE="mean median std iqr idr skew kurt Del Amp"
mkdir -p "$ROOT/logs" "$OUT"
tag () { echo "eye_${1}_${2//,/-}_allrings"; }

run () {  # run <task> <model> <groups> <n_perm>
  local T=$1 M=$2 G=$3 NP=$4 TG; TG=$(tag $1 "$3")
  echo "== $(date +%T) $TG $M (perm=$NP)"
  python -m src.repeated_cv run --model $M --task $T --groups "$G" $DATA --out $RCV/$TG --group_by eye \
    --repeats $R --n_perm $NP --perm_repeats 2 --n_jobs $NJ --rf_trees $TREES >> $ROOT/logs/rcv_${M}_${TG}.log 2>&1 \
    || echo "!! FAILED $TG $M"
  grep -a "DONE\|skip" $ROOT/logs/rcv_${M}_${TG}.log | tail -1
}

for T in $TASKS; do for M in $MODELS; do
  NPL=0; [ "$M" = lr ] && NPL=$P                  # permutation p-values from LR for single-modality sets
  run $T $M all $NPL                               # baseline (normally already done -> skipped)
  run $T $M "$MPOD" $NPL                           # A: structure only
  for G in "$MPOD,Del" "$MPOD,Amp" "$MPOD,coupling" "all,coupling"; do run $T $M "$G" 0; done    # A, C
  for G in Del Amp "Del,Amp" coupling; do run $T $M "$G" $NPL; done                             # A, C, D
  for g in $BASE; do run $T $M "all,-$g" 0; done                                                 # B: leave-one-group-out
done; done

echo; echo "######## PAIRED COMPARISONS (same eyes, same splits) ########"
rm -f $OUT/COMPARISONS.csv
for T in $TASKS; do for M in $MODELS; do
  PAIRS=( "$(tag $T "$MPOD,Del")|$(tag $T "$MPOD")|A: MPOD+Del vs MPOD"
          "$(tag $T "$MPOD,Amp")|$(tag $T "$MPOD")|A: MPOD+Amp vs MPOD"
          "$(tag $T all)|$(tag $T "$MPOD")|A: MPOD+Del+Amp vs MPOD"
          "$(tag $T all)|$(tag $T "$MPOD,Del")|A: adding Amp to MPOD+Del"
          "$(tag $T all)|$(tag $T "$MPOD,Amp")|A: adding Del to MPOD+Amp"
          "$(tag $T "$MPOD,coupling")|$(tag $T "$MPOD")|C: MPOD+coupling vs MPOD"
          "$(tag $T "all,coupling")|$(tag $T all)|C: all+coupling vs all" )
  for g in $BASE; do PAIRS+=( "$(tag $T all)|$(tag $T "all,-$g")|B: importance of $g (all minus $g)" ); done
  python -m src.compare_sets --root $RCV --model $M --out $OUT/COMPARISONS.csv --n_boot $NB --pairs "${PAIRS[@]}" \
    > $ROOT/logs/compare_${T}_${M}.log 2>&1 || echo "!! compare failed $T $M"
done; done

for D in $RCV/eye_*_*; do [ -d "$D" ] && python -m src.repeated_cv aggregate --out $D $DATA > $D/aggregate.log 2>&1; done
python - <<PY
import pandas as pd, glob
t = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("$RCV/*/summary.csv"))], ignore_index=True)
t = t[["task","groups","rings","group_by","model","n_eyes","AUC","AUC_lo","AUC_hi","p_above","p_below","verdict"]]
t.to_csv("$RCV/OVERVIEW.csv", index=False)
c = pd.read_csv("$OUT/COMPARISONS.csv")
pd.set_option("display.width", 230); pd.set_option("display.max_colwidth", 45)
md = ["# OFA value analyses (pre-specified; docs/ANALYSIS_PLAN_OFA.md)", "",
      "dAUC = A - B, paired on the same eyes and CV splits; 95% CI = eye-clustered bootstrap.", ""]
for T in c.task.unique():
    md += [f"## task: {T}", "~~~", c[c.task == T][["model","comparison","AUC_A","AUC_B","dAUC","lo","hi","conclusion"]].round(3).to_string(index=False), "~~~", ""]
s = t[t.groups.isin(["Del","Amp","Del,Amp","coupling","mean,median,std,iqr,idr,skew,kurt","all"]) & (t.rings=="all")]
md += ["## Single-modality and baseline sets (permutation p from LR)", "~~~",
       s[["task","groups","model","AUC","AUC_lo","AUC_hi","p_above","verdict"]].sort_values(["task","groups","model"]).round(3).to_string(index=False), "~~~"]
open("$OUT/REPORT.md","w").write("\n".join(md)+"\n"); print("\n".join(md))
PY
echo; echo "ALL DONE $(date). Next: bash scripts/5_package_share.sh && bash scripts/sync.sh 'ofa value'"
