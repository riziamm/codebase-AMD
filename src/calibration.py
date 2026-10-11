#!/usr/bin/env python3
"""
calibration.py: calibration and decision-curve analysis (answers EAAI R1-10) from the out-of-fold predictions
of the repeated nested CV (no refitting). Paper 1, all-feature models.

Per model x task x grouping (computed per repeat, then averaged; 95% CI = eye-clustered bootstrap):
  Brier score, scaled Brier (1 - Brier / Brier of the prevalence), calibration intercept (calibration-in-the-large,
  logit(p) as offset) and calibration slope (logistic recalibration), paired dBrier vs the random forest.
Figures: binned calibration curves (pooled over repeats) and decision curves (net benefit vs threshold).
Note: all models used class weighting (pos_weight / balanced), which shifts predicted risks upwards for the
minority class; calibration-in-the-large is therefore expected to deviate and is reported, not corrected.

Usage:  python -m src.calibration --root reruns_eyegrouped
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = ["lr", "rf", "hybrid_zone", "hybrid_orig"]
LABEL = {"lr": "Logistic regression", "rf": "Random forest", "hybrid_zone": "Zone-token hybrid", "hybrid_orig": "Original hybrid"}
COLOR = {"lr": "#7f8c8d", "rf": "#2e86c1", "hybrid_zone": "#d35400", "hybrid_orig": "#8e44ad"}
TASK = {"any": "Any AMD (AREDS 2-4 vs 1)", "advanced": "Intermediate/advanced (AREDS 3-4 vs 1)", "early": "Early AMD (AREDS 2 vs 1)"}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _newton(X, y, offset=None, iters=50):
    """Unpenalised logistic regression by Newton-Raphson (small, dependency-free)."""
    b = np.zeros(X.shape[1]); off = np.zeros(len(y)) if offset is None else offset
    for _ in range(iters):
        eta = X @ b + off; mu = 1 / (1 + np.exp(-eta)); W = mu * (1 - mu) + 1e-9
        H = X.T @ (X * W[:, None]); g = X.T @ (y - mu)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def cal_metrics(y, p):
    lp = _logit(p)
    brier = np.mean((p - y) ** 2); brier0 = np.mean((y.mean() - y) ** 2)
    slope = _newton(np.c_[np.ones_like(lp), lp], y)[1]
    intercept = _newton(np.ones((len(y), 1)), y, offset=lp)[0]
    return dict(brier=brier, scaled_brier=1 - brier / brier0 if brier0 > 0 else np.nan, cal_slope=slope, cal_intercept=intercept)


def load_oof(folder, model):
    f = Path(folder) / model / "oof.csv"
    if not f.exists():
        return None
    o = pd.read_csv(f).sort_values(["cv_rep", "row_index"])
    R = int(o.cv_rep.max()) + 1
    P = np.vstack([o[o.cv_rep == r].y_prob.values for r in range(R)])
    first = o[o.cv_rep == 0]
    return dict(P=P, y=first.y_true.values.astype(int), eye=first.eye_code.values)


def boot_rows(eye, B, seed):
    rng = np.random.default_rng(seed); eyes = np.unique(eye); idx = {e: np.where(eye == e)[0] for e in eyes}
    for _ in range(B):
        yield np.concatenate([idx[e] for e in rng.choice(eyes, len(eyes), replace=True)])


def summarise(D, B, seed):
    y = D["y"]
    point = pd.DataFrame([cal_metrics(y, p) for p in D["P"]]).mean()
    boots = []
    for rows in boot_rows(D["eye"], B, seed):
        yb = y[rows]
        if yb.min() == yb.max():
            continue
        boots.append(pd.DataFrame([cal_metrics(yb, p[rows]) for p in D["P"]]).mean())
    bt = pd.DataFrame(boots)
    return point, bt.quantile(0.025), bt.quantile(0.975)


def net_benefit(y, p, t):
    n = len(y); pos = p >= t
    tp = np.sum(pos & (y == 1)); fp = np.sum(pos & (y == 0))
    return tp / n - fp / n * t / (1 - t)


def figures(data, task, grouping, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    bins = np.linspace(0, 1, 6)
    for m, D in data.items():
        y = np.tile(D["y"], len(D["P"])); p = D["P"].ravel()
        idx = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
        mp = [p[idx == k].mean() for k in range(len(bins) - 1) if (idx == k).sum() >= 5]
        oy = [y[idx == k].mean() for k in range(len(bins) - 1) if (idx == k).sum() >= 5]
        a1.plot(mp, oy, "o-", color=COLOR[m], label=LABEL[m])
        th = np.linspace(0.05, 0.8, 31)
        nb = [np.mean([net_benefit(D["y"], pr, t) for pr in D["P"]]) for t in th]
        a2.plot(th, nb, color=COLOR[m], label=LABEL[m])
    a1.plot([0, 1], [0, 1], "k--", lw=0.8); a1.set_xlabel("Mean predicted probability (5 bins)"); a1.set_ylabel("Observed proportion with AMD")
    a1.set_title("Calibration (out-of-fold, pooled over repeats)", fontsize=9); a1.legend(fontsize=7)
    y0 = next(iter(data.values()))["y"]; prev = y0.mean(); th = np.linspace(0.05, 0.8, 31)
    a2.plot(th, prev - (1 - prev) * th / (1 - th), "k:", lw=1, label="Treat all")
    a2.axhline(0, color="k", lw=0.8, label="Treat none"); a2.set_ylim(-0.1, max(prev, 0.1) + 0.05)
    a2.set_xlabel("Threshold probability"); a2.set_ylabel("Net benefit"); a2.set_title("Decision curve", fontsize=9); a2.legend(fontsize=7)
    fig.suptitle(f"{TASK[task]} | {grouping}-grouped repeated nested CV", fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"calibration_dca_{task}_{grouping}.png", dpi=250, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="reruns_eyegrouped")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tasks", nargs="+", default=["advanced", "any"])
    ap.add_argument("--groupings", nargs="+", default=["eye", "subject"])
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    root = Path(a.root); out = Path(a.out or root / "calibration"); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in a.tasks:
        for g in a.groupings:
            folder = root / "repeated_cv" / f"{g}_{t}_all_allrings"
            data = {m: D for m in MODELS if (D := load_oof(folder, m)) is not None}
            if not data:
                print(f"[missing] {folder}"); continue
            for m, D in data.items():
                pt, lo, hi = summarise(D, a.n_boot, a.seed)
                r = dict(task=t, grouping=g, model=m)
                for k in pt.index:
                    r[k], r[f"{k}_lo"], r[f"{k}_hi"] = pt[k], lo[k], hi[k]
                if m != "rf" and "rf" in data and np.array_equal(D["y"], data["rf"]["y"]):
                    d = []
                    for rows_ in boot_rows(D["eye"], a.n_boot, a.seed):
                        yb = D["y"][rows_]
                        d.append(np.mean([np.mean((p[rows_] - yb) ** 2) for p in D["P"]]) -
                                 np.mean([np.mean((p[rows_] - yb) ** 2) for p in data["rf"]["P"]]))
                    obs = np.mean([np.mean((p - D["y"]) ** 2) for p in D["P"]]) - np.mean([np.mean((p - D["y"]) ** 2) for p in data["rf"]["P"]])
                    r["dBrier_vs_rf"], r["dBrier_lo"], r["dBrier_hi"] = obs, np.percentile(d, 2.5), np.percentile(d, 97.5)
                rows.append(r)
            figures(data, t, g, out)
            print(f"done {t} {g}: {list(data)}")
    T = pd.DataFrame(rows); T.to_csv(out / "CALIBRATION.csv", index=False)
    f = lambda v, l, h, d=3: f"{v:.{d}f} [{l:.{d}f}, {h:.{d}f}]"
    show = pd.DataFrame({"task": T.task, "grouping": T.grouping, "model": T.model.map(LABEL),
                         "Brier": [f(*x) for x in zip(T.brier, T.brier_lo, T.brier_hi)],
                         "scaled Brier": [f(*x, d=2) for x in zip(T.scaled_brier, T.scaled_brier_lo, T.scaled_brier_hi)],
                         "slope (ideal 1)": [f(*x, d=2) for x in zip(T.cal_slope, T.cal_slope_lo, T.cal_slope_hi)],
                         "intercept (ideal 0)": [f(*x, d=2) for x in zip(T.cal_intercept, T.cal_intercept_lo, T.cal_intercept_hi)],
                         "dBrier vs RF": [f(v, l, h) if pd.notna(v) else "-" for v, l, h in
                                          zip(T.get("dBrier_vs_rf", pd.Series(np.nan, index=T.index)), T.get("dBrier_lo", pd.Series(np.nan, index=T.index)),
                                              T.get("dBrier_hi", pd.Series(np.nan, index=T.index)))]})
    md = ["# Calibration and decision curves (out-of-fold, repeated nested CV)", "",
          "Brier: lower is better; scaled Brier: 1 = perfect, 0 = no better than predicting the prevalence.",
          "Slope < 1 = predictions too extreme (overfitting); intercept != 0 = systematic over/under-estimation",
          "(expected here: all models used class weighting). dBrier < 0 = better calibrated than the random forest.", "",
          "| " + " | ".join(show.columns) + " |", "|" + "---|" * len(show.columns)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in show.values]
    md += ["", "Figures: calibration_dca_<task>_<grouping>.png"]
    (out / "CAL_REPORT.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
