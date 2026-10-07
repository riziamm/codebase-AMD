#!/bin/bash
# PILOT for the repeated nested CV (~5-10 min). Nothing here goes into the paper.
#   nohup bash scripts/6_rcv_pilot.sh > rcv_pilot.log 2>&1 &
#   tail -n 25 rcv_pilot.log        # wait for the CHECKLIST: all PASS
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1
SPLIT=reruns_eyegrouped/split
OUT=reruns_pilot/repeated_cv/eye
R="${RCV_R:-2}"; P="${RCV_PERM:-6}"; PD="${RCV_PERM_DL:-4}"; EP="${RCV_EPOCHS:-80}"; TREES="${RCV_TREES:-100}"
NJ=$(nproc 2>/dev/null || echo 4); [ "$NJ" -gt 16 ] && NJ=16
[ -f $SPLIT/train_mpod.csv ] && [ -f $SPLIT/test_mpod.csv ] || { echo "ERROR: $SPLIT/*.csv missing (run src/make_split.py)"; exit 1; }
rm -rf reruns_pilot/repeated_cv; mkdir -p "$OUT"
python -c "import torch;print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')" 2>&1 | grep GPU
COMMON="--data $SPLIT/train_mpod.csv $SPLIT/test_mpod.csv --out $OUT --group_by eye --k 5 --repeats $R --perm_repeats 1 --chunk 3 --n_jobs $NJ --n_boot 200 --rf_trees $TREES --dl_epochs $EP"
for M in lr rf; do python -m src.repeated_cv run --model $M --n_perm $P $COMMON > "$OUT/$M.log" 2>&1; echo "== $M done"; done
for M in hybrid_zone hybrid_orig; do python -m src.repeated_cv run --model $M --n_perm $PD $COMMON > "$OUT/$M.log" 2>&1; echo "== $M done"; done
python -m src.repeated_cv aggregate --out $OUT --data $SPLIT/train_mpod.csv $SPLIT/test_mpod.csv --n_boot 200 > "$OUT/aggregate.log" 2>&1

ok () { if eval "$2"; then echo "  PASS  $1"; else echo "  FAIL  $1"; fi; }
echo; echo "=========== CHECKLIST ==========="
ok "all 4 models finished"         "[ \$(ls $OUT/*/summary.json 2>/dev/null | wc -l) -eq 4 ]"
ok "summary.csv has 4 rows"        "[ \$(tail -n +2 $OUT/summary.csv 2>/dev/null | wc -l) -eq 4 ]"
ok "AUCs are finite (0..1)"        "python -c \"import pandas as pd;d=pd.read_csv('$OUT/summary.csv');assert d.AUC.between(0,1).all()\""
ok "null files complete"           "python -c \"import pandas as pd;[pd.read_csv('$OUT/%s/null.csv'%m) for m in ['lr','rf','hybrid_zone','hybrid_orig']]\""
ok "paired.csv written"            "[ -s $OUT/paired.csv ]"
ok "same 58 eyes everywhere"       "python -c \"import pandas as pd;d=pd.read_csv('$OUT/summary.csv');assert (d.n_eyes==58).all()\""
ok "no Traceback in logs"          "! grep -l Traceback $OUT/*.log"
echo "================================="
echo "If any FAIL: send me the matching *.log in $OUT/"
