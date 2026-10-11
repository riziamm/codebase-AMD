#!/bin/bash
# Paper 1 calibration + decision curves (EAAI R1-10) from existing out-of-fold predictions. No refitting. ~5-10 min.
#   bash scripts/13_calibration.sh
#   then: bash scripts/5_package_share.sh && bash scripts/sync.sh "calibration"
set -u
ROOT="${ROOT:-reruns_eyegrouped}"
python -m src.calibration --root $ROOT --tasks advanced any --groupings eye subject --n_boot ${CAL_BOOT:-1000}
echo "Open: $ROOT/calibration/CAL_REPORT.md"
