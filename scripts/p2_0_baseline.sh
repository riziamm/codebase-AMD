#!/bin/bash
# Paper 2, step S0 (docs/ANALYSIS_PLAN_PAPER2.md): Table 1, age-confound check for Paper 1, follow-up counts.
# Contains no predictor-outcome association -> allowed BEFORE plan sign-off. CPU, ~1 min.
#   nohup bash scripts/p2_0_baseline.sh > reruns_paper2/logs/p2_0.log 2>&1 &
set -u
EXPORT="${EXPORT:-data/export}"; ROOT="${ROOT:-reruns_paper2}"; RAW="${RAW:-demographics_raw.csv}"
BANDS="${BANDS44:-}"; B44=""; [ -n "$BANDS" ] && B44="--bands44 $BANDS"
mkdir -p "$ROOT/logs"
[ -f "$EXPORT/demographics.csv" ] || { echo "STOP: $EXPORT/demographics.csv missing. Run: python -m src.mat_to_csv --mat <file> --out $EXPORT --demographics $RAW"; exit 1; }
python -m src.p2_baseline --export "$EXPORT" --out "$ROOT/p2_baseline" --raw "$RAW" $B44 --n_boot 2000
echo "ALL DONE $(date). Next: bash scripts/p2_package.sh && bash scripts/sync.sh 'p2 S0'"
