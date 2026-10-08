#!/usr/bin/env python3
"""
package_share.py — copy ONLY aggregate, shareable outputs into share/<tag>/ (safe to git push).

Copies
  results      results_holdout*.csv
  split        split_report.txt, split_manifest.sha256        (counts only)
  env          pip_freeze.txt, versions.txt, commit.txt, gpu.txt  (if present)
  ML           all_configurations.json, experiment_config.json, metrics/*.json,
               batch summary CSVs, figures *.png
  DL           experiment_config.json, metrics/**/*.json, tuning CSVs, figures *.png
  (opt-in)     --include_preds : holdout prediction CSVs (y_true,y_prob,y_pred ONLY; no IDs/features)

Safety guards (a file that fails is SKIPPED and listed)
  * JSON: per-row arrays (y_true, y_pred, y_prob, y_proba) are stripped unless --include_preds
  * CSV : refused if it has feature/ID columns (Subject, Eye, *_region*, *_Z*) or >200 rows
          (prediction CSVs: only y_true/y_prob/y_pred, <=40 rows)
  * never copies .pkl .npy .npz .pt .pth .html (force plots embed feature values) .log
  * files >5 MB are skipped

Usage
  python -m src.package_share --root reruns_eyegrouped --out share/run_$(date +%Y%m%d)
"""
import argparse, hashlib, json, shutil
from pathlib import Path
import pandas as pd

DROP_KEYS = {"y_true", "y_pred", "y_prob", "y_proba"}
BAD_COLS = ("subject", "eye", "repeat")
BAD_SUBSTR = ("_region", "_Z")
MAX_MB = 5.0


def strip_json(obj):
    if isinstance(obj, dict):
        return {k: strip_json(v) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, list):
        return [strip_json(x) for x in obj]
    return obj


OOF_COLS = {"cv_rep", "row_index", "eye_code", "y_true", "y_prob"}


def csv_ok(path, pred_mode=False, max_rows=200):
    if Path(path).name == "oof.csv":  # repeated-CV out-of-fold scores: pseudonymous eye_code, no features
        try:
            cols = set(pd.read_csv(path, nrows=5).columns)
        except Exception as e:
            return False, f"unreadable ({e})"
        return (cols <= OOF_COLS, "oof.csv has unexpected columns" if not cols <= OOF_COLS else "")
    try:
        df = pd.read_csv(path, nrows=500)
    except Exception as e:
        return False, f"unreadable ({e})"
    cols = [str(c) for c in df.columns]
    if pred_mode:
        if not set(c.lower() for c in cols) <= {"y_true", "y_prob", "y_pred"}:
            return False, f"prediction CSV has extra columns {cols}"
        return (len(df) <= 40, "too many rows" if len(df) > 40 else "")
    if any(c.lower() in BAD_COLS for c in cols) or any(any(b in c for b in BAD_SUBSTR) for c in cols):
        return False, "has ID/feature columns"
    if len(df) > max_rows:
        return False, f">{max_rows} rows"
    return True, ""


class Packager:
    def __init__(self, out, include_preds):
        self.out = Path(out); self.include_preds = include_preds
        self.copied, self.skipped, self.bytes = [], [], 0
        self.out.mkdir(parents=True, exist_ok=True)

    def _dst(self, src, root, sub):
        rel = src.relative_to(root)
        return self.out / sub / rel

    def add(self, src, root, sub):
        src = Path(src)
        if not src.is_file():
            return
        if src.stat().st_size > MAX_MB * 1e6:
            self.skipped.append((str(src), f">{MAX_MB} MB")); return
        suf = src.suffix.lower()
        if suf in {".pkl", ".npy", ".npz", ".pt", ".pth", ".html", ".log", ".joblib", ".h5"}:
            return
        dst = self._dst(src, root, sub)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if suf == ".json":
            try:
                data = json.load(open(src))
            except Exception as e:
                self.skipped.append((str(src), f"bad json ({e})")); return
            if not self.include_preds:
                data = strip_json(data)
            json.dump(data, open(dst, "w"), indent=1)
        elif suf == ".csv":
            pred = "preds_holdout" in src.parts
            ok, why = csv_ok(src, pred_mode=pred, max_rows=2000 if "diagnostics" in src.parts else 200)
            if not ok:
                self.skipped.append((str(src), why)); return
            shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        self.copied.append(str(dst)); self.bytes += dst.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="reruns_eyegrouped")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include_preds", action="store_true")
    ap.add_argument("--max_total_mb", type=float, default=80.0)
    a = ap.parse_args()
    root = Path(a.root)
    P = Packager(a.out, a.include_preds)

    for f in root.glob("results_holdout*.csv"):
        P.add(f, root, "results")
    for n in ("split_report.txt", "split_manifest.sha256"):
        P.add(root / "split" / n, root / "split", "split")
    for f in (root / "env").glob("*"):
        P.add(f, root / "env", "env")

    mlb = root / "ml_batches"
    for pat in ("*/all_configurations.json", "*/*.csv", "*/experiment_*/experiment_config.json",
                "*/experiment_*/metrics/*.json", "*/experiment_*/metrics/*.csv", "*/experiment_*/figures/*.png"):
        for f in mlb.glob(pat):
            P.add(f, mlb, "ml_batches")
    mle = root / "ml_eval"
    for pat in ("**/*.json", "**/*.png"):
        for f in mle.glob(pat):
            P.add(f, mle, "ml_eval")

    dl = root / "dl"
    for pat in ("*/experiment_config.json", "*/metrics/**/*.json", "*/post_analysis/metrics/**/*.json",
                "*/**/*.csv", "*/figures/**/*.png", "*/post_analysis/figures/**/*.png"):
        for f in dl.glob(pat):
            P.add(f, dl, "dl")

    for f in (root / "diagnostics").rglob("*"):
        if f.suffix.lower() in {".csv", ".md", ".png", ".json"}:
            P.add(f, root / "diagnostics", "diagnostics")
    rcv = root / "repeated_cv"
    P.add(rcv / "OVERVIEW.csv", rcv, "repeated_cv")
    for pat in ("*/summary.csv", "*/paired.csv", "*/REPORT.md", "*/*/summary.json", "*/*/null.csv"):
        for f in rcv.glob(pat):
            P.add(f, rcv, "repeated_cv")
    if a.include_preds:
        for f in (root / "preds_holdout").glob("*.csv"):
            P.add(f, root / "preds_holdout", "preds_holdout")
        for f in rcv.glob("*/*/oof.csv"):
            P.add(f, rcv, "repeated_cv")

    man = Path(a.out) / "MANIFEST.txt"
    lines = [f"files copied: {len(P.copied)}   total size: {P.bytes/1e6:.1f} MB   include_preds={a.include_preds}"]
    lines += [f"SKIPPED {s}: {w}" for s, w in P.skipped]
    man.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:1] + [f"SKIPPED: {len(P.skipped)} file(s) (see MANIFEST.txt)"]))
    if P.bytes / 1e6 > a.max_total_mb:
        print(f"WARNING: package is {P.bytes/1e6:.0f} MB (> {a.max_total_mb} MB). Consider deleting figures/ folders you don't need before pushing.")
    # final safety scan of the OUTPUT folder
    bad = [str(f) for f in Path(a.out).rglob("*.csv")
           if "preds_holdout" not in f.parts and not csv_ok(f, max_rows=2000 if "diagnostics" in f.parts else 200)[0]]
    bad += [str(f) for f in Path(a.out).rglob("*") if f.suffix.lower() in {".pkl", ".npy", ".npz", ".pt", ".html"}]
    if bad:
        raise SystemExit(f"SAFETY SCAN FAILED, remove before pushing: {bad[:5]}")
    print("SAFETY SCAN PASSED: no ID/feature columns, no pkl/npy/html in package.")


if __name__ == "__main__":
    main()
