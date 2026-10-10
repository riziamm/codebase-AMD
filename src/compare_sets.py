#!/usr/bin/env python3
"""
compare_sets.py: paired AUC differences between FEATURE SETS (same model, task, eyes and CV splits).

Each run folder (from src.repeated_cv) holds <model>/oof.csv. Two runs of the same task/group_by/seed
use identical outer splits, so their out-of-fold scores are paired eye by eye.
dAUC = AUC(A) - AUC(B), averaged over repeats; 95% CI from an eye-clustered bootstrap.

Usage
  python -m src.compare_sets --root reruns_eyegrouped/repeated_cv --model rf \
      --pairs "eye_advanced_mpod-Del_allrings|eye_advanced_mpod_allrings" ... --out table.csv
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load(folder, model):
    o = pd.read_csv(Path(folder) / model / "oof.csv")
    s = json.load(open(Path(folder) / model / "summary.json"))
    R = int(o.cv_rep.max()) + 1
    o = o.sort_values(["cv_rep", "row_index"])
    P = np.vstack([o[o.cv_rep == r].y_prob.values for r in range(R)])
    first = o[o.cv_rep == 0]
    return dict(P=P, y=first.y_true.values.astype(int), eye=first.eye_code.values, row=first.row_index.values,
                auc=s["auc_mean"], task=s.get("task"), group_by=s["group_by"], groups=s.get("groups"))


def paired(A, B, n_boot, seed):
    if not (np.array_equal(A["y"], B["y"]) and np.array_equal(A["row"], B["row"]) and np.array_equal(A["eye"], B["eye"])):
        raise SystemExit("runs are not paired (different rows/labels): check task and group_by")
    R = min(len(A["P"]), len(B["P"]))
    y = A["y"]
    obs = np.mean([roc_auc_score(y, A["P"][r]) - roc_auc_score(y, B["P"][r]) for r in range(R)])
    eyes = np.unique(A["eye"]); idx = {e: np.where(A["eye"] == e)[0] for e in eyes}
    rng = np.random.default_rng(seed); d = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[e] for e in rng.choice(eyes, len(eyes), replace=True)])
        yb = y[rows]
        if yb.min() == yb.max():
            continue
        d.append(np.mean([roc_auc_score(yb, A["P"][r][rows]) - roc_auc_score(yb, B["P"][r][rows]) for r in range(R)]))
    d = np.array(d)
    return obs, np.percentile(d, 2.5), np.percentile(d, 97.5), min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="'folderA|folderB[|label]' ; dAUC = A - B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rows = []
    for spec in a.pairs:
        parts = spec.split("|"); fa, fb = parts[0], parts[1]; label = parts[2] if len(parts) > 2 else f"{fa} - {fb}"
        pa, pb = Path(a.root) / fa, Path(a.root) / fb
        if not ((pa / a.model / "oof.csv").exists() and (pb / a.model / "oof.csv").exists()):
            print(f"  [missing] {label}"); continue
        A, B = load(pa, a.model), load(pb, a.model)
        d, lo, hi, p = paired(A, B, a.n_boot, a.seed)
        rows.append(dict(task=A["task"], model=a.model, comparison=label, set_A=A["groups"], set_B=B["groups"],
                         AUC_A=A["auc"], AUC_B=B["auc"], dAUC=d, lo=lo, hi=hi, boot_p=p,
                         conclusion=("A better" if lo > 0 else "B better" if hi < 0 else "no difference detected")))
    t = pd.DataFrame(rows)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out)
        t = pd.concat([old[~(old.model.eq(a.model) & old.comparison.isin(t.comparison) & old.task.isin(t.task))], t])
    t.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False) if rows else "no comparisons")


if __name__ == "__main__":
    main()
