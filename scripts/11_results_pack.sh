#!/bin/bash
# Paper 1 results pack: tables (T1-T4), figures (incl. out-of-fold ROC, needs oof.csv -> run on the HPC), RESULTS_PACK.md.
#   bash scripts/11_results_pack.sh            (~2 min)
#   then: bash scripts/5_package_share.sh && bash scripts/sync.sh "results pack"
set -u
ROOT="${ROOT:-reruns_eyegrouped}"
python -m src.results_pack --root $ROOT --out $ROOT/results_pack 2>&1 | grep -v "DLClassification"
echo "Open: $ROOT/results_pack/RESULTS_PACK.md"
