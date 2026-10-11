#!/bin/bash
# End-to-end test of the Paper 2 pipeline on SYNTHETIC data (exact .mat schema -> src.mat_to_csv -> S0, P2, P1,
# packaging + safety scan). Never touches real data. ~5 min.   bash tests/run_synthetic.sh
set -eu
T=$(mktemp -d /tmp/p2test.XXXX); echo "workdir $T"
python tests/make_synthetic_mat.py --out $T/syn --mode signal --seed 1
python -m src.mat_to_csv --mat $T/syn/synthetic.mat --out $T/export --demographics $T/syn/demographics_raw.csv
export EXPORT=$T/export ROOT=$T/reruns_paper2 RAW=$T/syn/demographics_raw.csv BANDS44=$T/syn/ofa44_layout.csv
bash scripts/p2_0_baseline.sh | tail -3
ALLOW_UNSIGNED=1 P2_REPEATS=3 P2_BOOT=200 P2_TREES=60 bash scripts/p2_1_main.sh | grep -a "==\|DONE\|PRIMARY\|Visit rule"
python -m src.package_share --root $ROOT --out $T/share_test | tail -2
echo "files packaged:"; find $T/share_test -type f | sed "s|$T/share_test/||" | sort | head -40
if find $T/share_test -name "oof*" | grep -q .; then echo "FAIL: row-level file packaged"; exit 1; fi
python - <<PY
import pandas as pd
c = pd.read_csv("$T/share_test/p2_function/CONTRASTS.csv"); p = c[c.id == "P1"].iloc[0]
assert p.conclusion == "A adds (dR2 > 0)", p.conclusion       # planted Del effect must be found
print("PRIMARY on planted signal:", round(p.dR2, 3), p.conclusion)
PY
grep -q "VisitLast_year" $T/share_test/p2_baseline/visits.csv || { echo "FAIL: visit years missing"; exit 1; }
if grep -qE "[0-9]{2}-[A-Za-z]{3}-20(18|19|20)" -r $T/share_test; then echo "FAIL: exact visit date packaged"; exit 1; fi
echo "08-Mar-2019" > $T/reruns_paper2/p2_baseline/leak_test.csv
if python -m src.package_share --root $T/reruns_paper2 --out $T/share_leak >/dev/null 2>&1; then
  echo "FAIL: packager did not block a patient-era date"; exit 1; fi
echo "date guard OK (visit years only; leaked date blocked)"
echo "SYNTHETIC TEST PASSED"
