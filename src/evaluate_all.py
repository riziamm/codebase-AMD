#!/usr/bin/env python3
"""
evaluate_all.py — compute identical metrics for every model from saved predictions.

Input: one CSV per model in --pred_dir, named <model_name>.csv, with columns
    row_index, y_true, y_prob        (binary: y_prob = P(AMD))
    optional: y_pred                 (else threshold 0.5)
Rows are joined to split_manifest.csv on row_index to recover eye_id.

CIs: eye-clustered percentile bootstrap (resample eyes; both repeats kept together).
Paired dAUC: same resampled eyes for both models.

Usage
  python evaluate_all.py --pred_dir reruns_eyegrouped/preds_holdout \
      --manifest reruns_eyegrouped/split/split_manifest.csv \
      --out reruns_eyegrouped/results_holdout.csv \
      --pairs hybrid_parallel:rf_std_smote rf_std_smote:lr_baseline
Outputs contain aggregate metrics only (safe to share).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                             precision_score, balanced_accuracy_score, accuracy_score)


def point_metrics(y, p, yhat):
    tn = int(((y == 0) & (yhat == 0)).sum()); fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum()); tp = int(((y == 1) & (yhat == 1)).sum())
    out = {
        "AUC": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "PPV": tp / (tp + fp) if tp + fp else np.nan,
        "NPV": tn / (tn + fn) if tn + fn else np.nan,
        "BalancedAcc": balanced_accuracy_score(y, yhat),
        "Accuracy": accuracy_score(y, yhat),
        "F1_AMD": f1_score(y, yhat, average="binary", zero_division=0),
        "F1_weighted": f1_score(y, yhat, average="weighted", zero_division=0),
        "F1_macro": f1_score(y, yhat, average="macro", zero_division=0),
    }
    out.update({"TP": tp, "FP": fp, "TN": tn, "FN": fn})
    return out


def eye_bootstrap(df, n_boot, seed):
    """Yield resampled dataframes, resampling eyes with replacement."""
    rng = np.random.default_rng(seed)
    eyes = df["eye_id"].unique()
    groups = {e: g for e, g in df.groupby("eye_id")}
    for _ in range(n_boot):
        pick = rng.choice(eyes, size=len(eyes), replace=True)
        yield pd.concat([groups[e] for e in pick], ignore_index=True)


def load_preds(pred_dir, manifest):
    full = pd.read_csv(manifest)
    man = full[["row_index", "eye_id", "Subject"]]
    ho = full[full.partition == "holdout"].reset_index(drop=True)
    preds = {}
    for f in sorted(Path(pred_dir).glob("*.csv")):
        d = pd.read_csv(f)
        if "row_index" not in d:
            # predictions saved in test_mpod.csv order: map by position and verify labels
            if len(d) != len(ho):
                raise ValueError(f"{f.name}: no row_index and length {len(d)} != holdout {len(ho)}")
            if not (d["y_true"].astype(int).values == ho["y_bin"].values).all():
                raise ValueError(f"{f.name}: y_true order does not match holdout manifest order")
            d["row_index"] = ho["row_index"].values
        d = d.merge(man, on="row_index", how="left", validate="one_to_one")
        if d["eye_id"].isna().any():
            raise ValueError(f"{f.name}: row_index not found in manifest")
        if "y_pred" not in d:
            d["y_pred"] = (d["y_prob"] >= 0.5).astype(int)
        preds[f.stem] = d
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pairs", nargs="*", default=[],
                    help="modelA:modelB pairs for paired dAUC (A minus B)")
    a = ap.parse_args()

    preds = load_preds(a.pred_dir, a.manifest)
    rows = []
    for name, d in preds.items():
        y, p, yhat = d.y_true.values.astype(int), d.y_prob.values, d.y_pred.values.astype(int)
        pt = point_metrics(y, p, yhat)
        boots = {k: [] for k in pt if k not in ("TP", "FP", "TN", "FN")}
        for b in eye_bootstrap(d, a.n_boot, a.seed):
            if b.y_true.nunique() < 2:
                continue
            m = point_metrics(b.y_true.values.astype(int), b.y_prob.values, b.y_pred.values.astype(int))
            for k in boots:
                boots[k].append(m[k])
        r = {"model": name, "n_rows": len(d), "n_eyes": d.eye_id.nunique(),
             "n_subjects": d.Subject.nunique()}
        for k, v in pt.items():
            r[k] = v
            if k in boots:
                arr = np.array(boots[k], dtype=float)
                arr = arr[~np.isnan(arr)]
                r[f"{k}_lo"], r[f"{k}_hi"] = (np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan))
        rows.append(r)
    res = pd.DataFrame(rows)

    # paired dAUC
    pair_rows = []
    for pair in a.pairs:
        A, B = pair.split(":")
        da, db = preds[A], preds[B]
        m = da[["row_index", "eye_id", "y_true", "y_prob"]].merge(
            db[["row_index", "y_prob"]], on="row_index", suffixes=("_A", "_B"))
        assert len(m) == len(da) == len(db), f"{pair}: models scored on different rows"
        obs = roc_auc_score(m.y_true, m.y_prob_A) - roc_auc_score(m.y_true, m.y_prob_B)
        diffs = []
        for b in eye_bootstrap(m, a.n_boot, a.seed):
            if b.y_true.nunique() < 2:
                continue
            diffs.append(roc_auc_score(b.y_true, b.y_prob_A) - roc_auc_score(b.y_true, b.y_prob_B))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        pair_rows.append({"comparison": f"{A} - {B}", "dAUC": obs, "dAUC_lo": lo, "dAUC_hi": hi,
                          "boot_p_two_sided": 2 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0))})

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    if pair_rows:
        pd.DataFrame(pair_rows).to_csv(out.with_name(out.stem + "_paired_dAUC.csv"), index=False)

    pd.set_option("display.width", 200)
    show = ["model", "n_eyes", "AUC", "AUC_lo", "AUC_hi", "Sensitivity", "Specificity",
            "BalancedAcc", "F1_AMD", "F1_weighted"]
    print(res[show].round(3).to_string(index=False))
    if pair_rows:
        print(pd.DataFrame(pair_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
