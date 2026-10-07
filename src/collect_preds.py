#!/usr/bin/env python3
"""
collect_preds.py — gather holdout predictions from DL and ML runs into one folder
for src/evaluate_all.py. Output CSVs contain y_true, y_prob, y_pred only (no features).

DL : <run>/post_analysis/metrics/<Model>_final_best_holdout_eval_classification_report.json
ML : <eval_report_dir>/predictions/<batch>__<experiment>__<model>__test_test.csv (from `--mode evaluate` on test_test.pkl)

Usage
  python -m src.collect_preds --dl_runs reports/dl_pipeline/EG_* \
      --ml_dirs reports/final_evaluation_eg --out reruns_eyegrouped/preds_holdout
Each output file is named <run_name>__<model>.csv so every run stays traceable.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def from_dl(run_dir, out):
    run_dir = Path(run_dir)
    n = 0
    for f in run_dir.glob("post_analysis/metrics/*_final_best_holdout_eval_classification_report.json"):
        d = json.load(open(f))
        p = np.asarray(d["y_prob"], dtype=float)
        if p.ndim == 2 and p.shape[1] == 1:
            p = p[:, 0]
        elif p.ndim == 2:  # multiclass: keep max prob; evaluate_all is for the binary task
            p = p.max(axis=1)
        model = f.name.replace("_final_best_holdout_eval_classification_report.json", "")
        df = pd.DataFrame({"y_true": np.asarray(d["y_true"]).astype(int), "y_prob": p,
                           "y_pred": np.asarray(d["y_pred"]).astype(int)})
        dst = out / f"{run_dir.name}__{model}.csv"
        df.to_csv(dst, index=False); n += 1
        print(f"[DL] {dst.name}  rows={len(df)}")
    return n


def from_ml(ml_dir, out):
    ml_dir = Path(ml_dir)
    n = 0
    for f in ml_dir.glob("**/predictions/*__test_test.csv"):
        dst = out / f"ML__{f.stem.replace('__test_test', '')}.csv"
        pd.read_csv(f).to_csv(dst, index=False); n += 1
        print(f"[ML] {dst.name}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl_runs", nargs="*", default=[])
    ap.add_argument("--ml_dirs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = sum(from_dl(r, out) for r in a.dl_runs) + sum(from_ml(m, out) for m in a.ml_dirs)
    print(f"collected {n} prediction files into {out}")


if __name__ == "__main__":
    main()
