#!/bin/bash
# Paper 1 age check (docs/ANALYSIS_PLAN_AGE.md): age vs MPOD vs MPOD + age, same splits as Paper 1. CPU, resumable, ~2 h.
#   nohup bash scripts/12_age_check.sh > reruns_eyegrouped/logs/age_check.log 2>&1 &
# Needs data/export/demographics.csv (python -m src.mat_to_csv ... --demographics demographics_raw.csv).
set -u
export OMP_NUM_THREADS=1
ROOT="${ROOT:-reruns_eyegrouped}"; RCV=$ROOT/repeated_cv; OUT=$ROOT/age_check
SPLIT=reruns_eyegrouped/split; DATA="--data $SPLIT/train_mpod.csv $SPLIT/test_mpod.csv"
DEMO="${DEMO:-data/export/demographics.csv}"
R="${RCV_R:-10}"; P="${RCV_PERM:-200}"; TREES="${RCV_TREES:-300}"; NB="${CMP_BOOT:-2000}"
TASKS="${TASKS:-advanced any early}"; MODELS="${MODELS:-rf lr}"; GROUPINGS="${GROUPINGS:-subject eye}"
NJ=$(nproc 2>/dev/null || echo 8); [ "$NJ" -gt 16 ] && NJ=16
MPOD="mean,median,std,iqr,idr,skew,kurt"
[ -f "$DEMO" ] || { echo "ERROR: $DEMO not found"; exit 1; }
mkdir -p "$ROOT/logs" "$OUT"
tag () { echo "${1}_${2}_${3//,/-}_allrings"; }      # <grouping>_<task>_<groups>

run () {  # run <grouping> <task> <model> <groups> <n_perm>
  local GB=$1 T=$2 M=$3 G=$4 NP=$5 TG; TG=$(tag $1 $2 "$4")
  echo "== $(date +%T) $TG $M (perm=$NP)"
  python -m src.repeated_cv run --model $M --task $T --groups "$G" --group_by $GB $DATA --demographics $DEMO \
    --out $RCV/$TG --repeats $R --n_perm $NP --perm_repeats 2 --n_jobs $NJ --rf_trees $TREES >> $ROOT/logs/rcv_${M}_${TG}.log 2>&1 \
    || echo "!! FAILED $TG $M"
  grep -a "DONE\|skip" $ROOT/logs/rcv_${M}_${TG}.log | tail -1
}

for T in $TASKS; do for GB in $GROUPINGS; do for M in $MODELS; do
  NPL=0; [ "$M" = lr ] && NPL=$P
  run $GB $T $M age $NPL
  run $GB $T $M "$MPOD" $NPL
  run $GB $T $M "$MPOD,age" 0
  run $GB $T $M all 0
  run $GB $T $M "all,age" 0
done; done; done

echo; echo "######## PAIRED COMPARISONS ########"
rm -f $OUT/COMPARISONS_AGE.csv
for T in $TASKS; do for GB in $GROUPINGS; do for M in $MODELS; do
  python -m src.compare_sets --root $RCV --model $M --out $OUT/COMPARISONS_AGE.csv --n_boot $NB --pairs \
    "$(tag $GB $T "$MPOD,age")|$(tag $GB $T age)|[$GB] MPOD+age vs age (PRIMARY if subject, RF)" \
    "$(tag $GB $T "$MPOD,age")|$(tag $GB $T "$MPOD")|[$GB] MPOD+age vs MPOD" \
    "$(tag $GB $T "$MPOD")|$(tag $GB $T age)|[$GB] MPOD vs age" \
    "$(tag $GB $T "all,age")|$(tag $GB $T age)|[$GB] all+age vs age" \
    "$(tag $GB $T "all,age")|$(tag $GB $T all)|[$GB] all+age vs all" > $ROOT/logs/compare_age_${T}_${GB}_${M}.log 2>&1 \
    || echo "!! compare failed $T $GB $M"
done; done; done

python - <<PY
import pandas as pd, json, glob
rows = []
for f in glob.glob("$RCV/*_*_*/*/summary.json"):
    s = json.load(open(f))
    if s.get("groups") in ("age", "$MPOD", "$MPOD,age", "all", "all,age") and s.get("rings", "all") == "all":
        rows.append(dict(grouping=s["group_by"], task=s.get("task"), model=s["model"], set=s["groups"],
                         AUC=s["auc_mean"], lo=s["auc_ci_lo"], hi=s["auc_ci_hi"], p=s["p_above"]))
t = pd.DataFrame(rows).replace({"set": {"$MPOD": "MPOD", "$MPOD,age": "MPOD+age", "all,age": "all+age"}})
t = t[t.model.isin("$MODELS".split())].sort_values(["grouping","task","model","set"])
t.to_csv("$OUT/AUC_BY_SET.csv", index=False)
c = pd.read_csv("$OUT/COMPARISONS_AGE.csv")
pd.set_option("display.width", 230); pd.set_option("display.max_colwidth", 60)
md = ["# Age check (pre-specified: docs/ANALYSIS_PLAN_AGE.md)", "", "## AUC by feature set", "~~~", t.round(3).to_string(index=False), "~~~", "",
      "## Paired comparisons (dAUC = A - B; 95% CI eye-clustered bootstrap)", "~~~",
      c[["task","model","comparison","AUC_A","AUC_B","dAUC","lo","hi","conclusion"]].round(3).to_string(index=False), "~~~"]
prim = c[c.comparison.str.startswith("[subject] MPOD+age vs age") & (c.model == "rf")]
md += ["", "## PRIMARY (RF, participant-grouped, MPOD+age vs age)", "~~~", prim[["task","dAUC","lo","hi","conclusion"]].round(3).to_string(index=False), "~~~"]
open("$OUT/REPORT.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
PY
echo; echo "ALL DONE $(date). Next: bash scripts/5_package_share.sh && bash scripts/sync.sh 'age check'"
