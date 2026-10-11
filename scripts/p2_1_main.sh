#!/bin/bash
# Paper 2 main analyses P2 (visit-to-visit stability) and P1 (functional validity). REQUIRES the plan to be signed off
# ("Status: SIGNED OFF" in docs/ANALYSIS_PLAN_PAPER2.md). CPU only, resumable. ~20-40 min with 8-16 cores.
#   nohup bash scripts/p2_1_main.sh > reruns_paper2/logs/p2_1.log 2>&1 &
#   grep -a "==\|DONE\|STOP\|Error" reruns_paper2/logs/p2_1.log | tail
set -u
export OMP_NUM_THREADS=1
EXPORT="${EXPORT:-data/export}"; ROOT="${ROOT:-reruns_paper2}"
BANDS="${BANDS44:-}"; B44=""; [ -n "$BANDS" ] && B44="--bands44 $BANDS"
UNS=""; [ "${ALLOW_UNSIGNED:-0}" = 1 ] && UNS="--allow_unsigned"   # synthetic tests ONLY
R="${P2_REPEATS:-20}"; NB="${P2_BOOT:-2000}"; TREES="${P2_TREES:-300}"
NJ=$(nproc 2>/dev/null || echo 8); [ "$NJ" -gt 16 ] && NJ=16
mkdir -p "$ROOT/logs"
echo "== $(date +%T) P2 visit-to-visit stability"
python -m src.p2_stability --export "$EXPORT" --out "$ROOT/p2_stability" $B44 --n_boot $NB $UNS || exit 1
echo "== $(date +%T) P1 functional validity (run)"
python -m src.p2_function run --export "$EXPORT" --out "$ROOT/p2_function" $B44 --repeats $R --k 5 --rf_trees $TREES --n_jobs $NJ $UNS || exit 1
echo "== $(date +%T) P1 aggregate"
python -m src.p2_function aggregate --export "$EXPORT" --out "$ROOT/p2_function" --n_boot $NB || exit 1
echo "ALL DONE $(date). Next: bash scripts/p2_package.sh && bash scripts/sync.sh 'p2 main'"
