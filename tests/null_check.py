"""Quick false-positive check for P1 on NULL synthetic data (no OFA effect). Not a full calibration."""
import subprocess, sys, numpy as np
from src.p2_data import eye_table
from src.p2_function import feature_sets, run_one, r2_rows, boot_index
py = sys.executable; hits = 0; seeds = range(11, 17)
for s in seeds:
    d = f"/tmp/p2null{s}"
    subprocess.run([py, "tests/make_synthetic_mat.py", "--out", d, "--mode", "null", "--seed", str(s)], check=True, capture_output=True)
    subprocess.run([py, "-m", "src.mat_to_csv", "--mat", f"{d}/synthetic.mat", "--out", f"{d}/export",
                    "--demographics", f"{d}/demographics_raw.csv"], check=True, capture_output=True)
    E, _ = eye_table(f"{d}/export"); S = feature_sets(E)
    _, y, PA = run_one(E, S["AS+MPOD+OFA"], "VAlc", "ridge", 5, 5, 2027, 100, 4)
    dd, _, PB = run_one(E, S["AS+MPOD"], "VAlc", "ridge", 5, 5, 2027, 100, 4)
    subj = dd.Subject.values
    bs = [(r2_rows(y[i], PA[:, i]) - r2_rows(y[i], PB[:, i])).mean() for i in boot_index(subj, 500, 1)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); obs = (r2_rows(y, PA) - r2_rows(y, PB)).mean()
    hits += lo > 0
    print(f"seed {s}: dR2={obs:+.3f} [{lo:+.3f}, {hi:+.3f}]  {'FALSE POSITIVE' if lo > 0 else 'ok'}")
print(f"false positives: {hits}/{len(seeds)}")
