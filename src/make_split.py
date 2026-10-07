#!/usr/bin/env python3
"""
make_split.py — create ONE frozen, eye-grouped split for all experiments.

Design
  * Unit of grouping = eye (Subject x Eye). Both repeat sessions of an eye
    always land in the same partition and the same CV fold.
  * Because AREDS grade is constant within an eye, a stratified split of the
    58-eye table (then mapped back to the 116 rows) is exactly a stratified
    group split. Works on any scikit-learn version (incl. 0.24).
  * Stratified on the 4-class AREDS grade so one split serves binary and
    multiclass tasks and every stage is represented in every partition.

Usage
  python make_split.py --data mpod.csv --out reruns_eyegrouped/split \
      --holdout_folds 7 --cv_folds 3 5 --seed 42

Outputs (no feature values are written)
  split_manifest.csv : row_index, Subject, Eye, Repeat, eye_id, Y, y_bin,
                       partition, cv3_fold, cv5_fold ...
  split_report.txt   : counts + leakage assertions (safe to share: aggregates only)
  split_manifest.sha256
"""
import argparse, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ID_COLS = ["Subject", "Repeat", "Eye", "Y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout_folds", type=int, default=7,
                    help="1/holdout_folds of eyes go to holdout (7 -> ~15%%)")
    ap.add_argument("--cv_folds", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write_csvs", action="store_true",
                    help="also write train_mpod.csv / test_mpod.csv (same columns as input) for the existing pipeline")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.data)
    missing = [c for c in ID_COLS if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns: {missing}")

    meta = df[ID_COLS].copy()
    meta.insert(0, "row_index", np.arange(len(df)))
    meta["eye_id"] = meta["Subject"].astype(str) + "_" + meta["Eye"].astype(str)
    meta["y_bin"] = (meta["Y"] >= 2).astype(int)

    # --- eye-level table (label must be constant within eye) ---
    per_eye = meta.groupby("eye_id")["Y"].nunique()
    if (per_eye > 1).any():
        sys.exit(f"Inconsistent Y within eye(s): {per_eye[per_eye > 1].index.tolist()}")
    eyes = meta.groupby("eye_id").agg(Subject=("Subject", "first"), Y=("Y", "first")).reset_index()

    # --- holdout: one fold of a stratified K-fold over eyes ---
    skf = StratifiedKFold(n_splits=a.holdout_folds, shuffle=True, random_state=a.seed)
    _, ho_idx = next(iter(skf.split(eyes["eye_id"], eyes["Y"])))
    ho_eyes = set(eyes.loc[ho_idx, "eye_id"])
    meta["partition"] = np.where(meta["eye_id"].isin(ho_eyes), "holdout", "dev")

    # --- CV folds within development eyes ---
    dev_eyes = eyes[~eyes["eye_id"].isin(ho_eyes)].reset_index(drop=True)
    for k in a.cv_folds:
        col = f"cv{k}_fold"
        fold_of = {}
        skf_k = StratifiedKFold(n_splits=k, shuffle=True, random_state=a.seed)
        for f, (_, te) in enumerate(skf_k.split(dev_eyes["eye_id"], dev_eyes["Y"])):
            for e in dev_eyes.loc[te, "eye_id"]:
                fold_of[e] = f
        meta[col] = meta["eye_id"].map(fold_of).fillna(-1).astype(int)  # -1 = holdout

    # --- assertions ---
    dev_e = set(meta.loc[meta.partition == "dev", "eye_id"])
    assert not (dev_e & ho_eyes), "Eye in both dev and holdout"
    for k in a.cv_folds:
        col = f"cv{k}_fold"
        assert meta[meta.partition == "dev"].groupby("eye_id")[col].nunique().max() == 1, \
            f"Eye split across {col} folds"

    # --- report (aggregates only) ---
    L = []
    L.append(f"data={a.data}  rows={len(meta)}  eyes={meta.eye_id.nunique()}  "
             f"subjects={meta.Subject.nunique()}  seed={a.seed}")
    L.append(f"feature columns (first 3 / last 3): {list(df.columns[:3])} ... "
             f"{[c for c in df.columns if c not in ID_COLS][-3:]}")
    for part in ["dev", "holdout"]:
        m = meta[meta.partition == part]
        e = m.drop_duplicates("eye_id")
        L.append(f"\n[{part}] rows={len(m)} eyes={len(e)} subjects={m.Subject.nunique()}")
        L.append(f"  AREDS (eyes): {e.Y.value_counts().sort_index().to_dict()}")
        L.append(f"  AREDS (rows): {m.Y.value_counts().sort_index().to_dict()}")
        L.append(f"  binary (rows) 0/1: {m.y_bin.value_counts().sort_index().to_dict()}")
    ho_subj = meta.loc[meta.partition == "holdout"].drop_duplicates("eye_id")
    fellow_in_dev = 0
    fellow_discordant = 0
    for _, r in ho_subj.iterrows():
        fellow = meta[(meta.Subject == r.Subject) & (meta.eye_id != r.eye_id)]
        if len(fellow) and (fellow.partition == "dev").all():
            fellow_in_dev += 1
            if fellow.y_bin.iloc[0] != int(r.Y >= 2):
                fellow_discordant += 1
    L.append(f"\nHoldout eyes whose fellow eye is in dev: {fellow_in_dev} / {len(ho_subj)}"
             f"  (of which binary-discordant: {fellow_discordant})")
    for k in a.cv_folds:
        col = f"cv{k}_fold"
        d = meta[meta.partition == "dev"]
        L.append(f"\nCV k={k}: rows per fold {d[col].value_counts().sort_index().to_dict()}; "
                 f"binary positives per fold "
                 f"{d.groupby(col).y_bin.sum().to_dict()}")
    L.append("\nASSERTIONS PASSED: no eye in >1 partition; no eye in >1 CV fold.")

    man = out / "split_manifest.csv"
    meta.to_csv(man, index=False)
    sha = hashlib.sha256(man.read_bytes()).hexdigest()
    (out / "split_manifest.sha256").write_text(sha + "\n")
    L.append(f"manifest sha256: {sha}")
    if a.write_csvs:
        dev_rows = meta.loc[meta.partition == "dev", "row_index"].values
        ho_rows = meta.loc[meta.partition == "holdout", "row_index"].values
        df.iloc[dev_rows].to_csv(out / "train_mpod.csv", index=False)
        df.iloc[ho_rows].to_csv(out / "test_mpod.csv", index=False)
        L.append(f"wrote train_mpod.csv ({len(dev_rows)} rows) and test_mpod.csv ({len(ho_rows)} rows), "
                 f"same column layout as input; row order = manifest order")
    (out / "split_report.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
