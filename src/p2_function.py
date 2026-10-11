#!/usr/bin/env python3
"""
p2_function.py: P1 of docs/ANALYSIS_PLAN_PAPER2.md. Do OFA measures explain visual acuity beyond age, sex and MPOD?

DESIGN (pre-specified; change only through a dated plan amendment)
  unit     : eye (58), features = mean of the two test sessions; outcome VAlc (primary) and VA (secondary)
  outer CV : repeated (R=20) 5-fold, SUBJECT-grouped (age/sex are subject-level; fellow eyes correlate),
             stratified on worst-eye AREDS {1, 2, 3+}; reuses src.repeated_cv.outer_splits. Same splits for
             every feature set of an outcome -> predictions are paired eye by eye.
  model    : ridge (primary; alpha tuned by inner 3-fold subject-grouped CV, MSE); random forest (sensitivity)
             imputation + scaling fitted inside every training fold. No deep learning (n = 58 eyes).
  metrics  : out-of-fold R^2 (primary), Spearman rho, MAE; averaged over repeats
  inference: subject-clustered bootstrap (B=2000) of the repeat-averaged metric; paired for differences
  decision : "adds" if the 95% CI of dR^2 excludes 0 (A > B); else "no difference detected"

  python -m src.p2_function run       --export data/export --out reruns_paper2/p2_function [--bands44 f.csv]
  python -m src.p2_function aggregate --export data/export --out reruns_paper2/p2_function
Resumable: a finished (outcome, set, model) is skipped. Row-level OOF stays on the HPC (oof_reg.csv).
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .p2_data import (MPOD_STATS, eye_table, cols_ring, cols_regions, write_run_info, require_signoff, log)
from .repeated_cv import outer_splits

OUTCOMES = ["VAlc", "VA"]


# ------------------------------------------------------------------ pre-specified sets and contrasts
def feature_sets(E):
    AS = ["Age", "Sex01"]
    MP, OF = cols_ring(MPOD_STATS), cols_ring(["Del", "Amp"])
    O44 = [c for c in E.columns if c.startswith(("Del2|", "Amp2|"))]
    S = {"AS": AS, "AS+MPOD": AS + MP, "AS+MPOD+OFA": AS + MP + OF, "AS+OFA": AS + OF,
         "AS+MPOD+Del": AS + MP + cols_ring(["Del"]), "AS+MPOD+Amp": AS + MP + cols_ring(["Amp"]),
         "AS+grade": AS + ["Y"], "AS+grade+OFA": AS + ["Y"] + OF,
         "AS+MPODz": AS + cols_regions(MPOD_STATS), "AS+MPODz+OFAz": AS + cols_regions(MPOD_STATS + ["Del", "Amp"])}
    if O44:
        S["AS+MPOD+OFA+OFA44"] = AS + MP + OF + O44
    return S


RUNS = [(o, s, "ridge") for o in OUTCOMES for s in
        ["AS", "AS+MPOD", "AS+MPOD+OFA", "AS+OFA", "AS+MPOD+Del", "AS+MPOD+Amp", "AS+grade", "AS+grade+OFA",
         "AS+MPOD+OFA+OFA44", "AS+MPODz", "AS+MPODz+OFAz"]] + \
       [(o, s, "rf") for o in OUTCOMES for s in ["AS+MPOD", "AS+MPOD+OFA"]]

# (id, role, outcome, model, set_A, set_B, question)   d = A - B
CONTRASTS = [
    ("P1", "PRIMARY", "VAlc", "ridge", "AS+MPOD+OFA", "AS+MPOD", "OFA adds to age+sex+MPOD for low-contrast VA"),
    ("S1", "secondary", "VA", "ridge", "AS+MPOD+OFA", "AS+MPOD", "OFA adds to age+sex+MPOD for VA"),
    ("S2a", "secondary", "VAlc", "ridge", "AS+OFA", "AS", "OFA adds to age+sex (VAlc)"),
    ("S2b", "secondary", "VA", "ridge", "AS+OFA", "AS", "OFA adds to age+sex (VA)"),
    ("S3a", "context", "VAlc", "ridge", "AS+MPOD", "AS", "MPOD adds to age+sex (VAlc)"),
    ("S3b", "context", "VA", "ridge", "AS+MPOD", "AS", "MPOD adds to age+sex (VA)"),
    ("S4a", "secondary", "VAlc", "ridge", "AS+grade+OFA", "AS+grade", "OFA adds beyond AREDS grade (VAlc)"),
    ("S4b", "secondary", "VA", "ridge", "AS+grade+OFA", "AS+grade", "OFA adds beyond AREDS grade (VA)"),
    ("S5a", "secondary", "VAlc", "ridge", "AS+MPOD+Del", "AS+MPOD", "Del alone adds (VAlc)"),
    ("S5b", "secondary", "VAlc", "ridge", "AS+MPOD+Amp", "AS+MPOD", "Amp alone adds (VAlc)"),
    ("S6a", "secondary", "VAlc", "ridge", "AS+MPOD+OFA+OFA44", "AS+MPOD+OFA", "44-region field adds to 20 (VAlc)"),
    ("S6b", "secondary", "VA", "ridge", "AS+MPOD+OFA+OFA44", "AS+MPOD+OFA", "44-region field adds to 20 (VA)"),
    ("SENS1", "sensitivity", "VAlc", "rf", "AS+MPOD+OFA", "AS+MPOD", "P1 with random forest"),
    ("SENS2", "sensitivity", "VAlc", "ridge", "AS+MPODz+OFAz", "AS+MPODz", "P1 at full 20-region resolution"),
]


# ------------------------------------------------------------------ models
def make_model(kind, seed, trees):
    pre = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    if kind == "ridge":
        return Pipeline(pre + [("m", Ridge())]), {"m__alpha": list(np.logspace(-3, 4, 15))}
    rf = RandomForestRegressor(n_estimators=trees, random_state=seed, n_jobs=1)
    return Pipeline(pre + [("m", rf)]), {"m__min_samples_leaf": [3, 5], "m__max_features": [0.33, 1.0]}


def _fit_predict(kind, X, y, g, tr, te, seed, trees):
    pipe, grid = make_model(kind, seed, trees)
    gs = GridSearchCV(pipe, grid, scoring="neg_mean_squared_error", cv=GroupKFold(n_splits=3), refit=True,
                      n_jobs=1, error_score="raise")
    gs.fit(X[tr], y[tr], groups=g[tr])
    return gs.predict(X[te])


def run_one(E, cols, outcome, kind, R, K, seed, trees, n_jobs):
    d = E[E[outcome].notna()].reset_index(drop=True)
    X = d[cols].to_numpy(float); y = d[outcome].to_numpy(float); g = d["Subject"].to_numpy()
    allnan = np.isnan(X).all(0)
    if allnan.any():
        raise SystemExit(f"columns entirely missing: {list(np.array(cols)[allnan])[:5]}")
    splits = outer_splits(d, K, R, seed, "subject")
    tasks = [(r, f) for r in range(R) for f in range(K)]
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(kind, X, y, g, splits[r][f][0], splits[r][f][1],
                                                        seed + 100 * r + f, trees) for r, f in tasks)
    P = np.full((R, len(y)), np.nan)
    for (r, f), p in zip(tasks, res):
        P[r, splits[r][f][1]] = p
    assert not np.isnan(P).any(), "some eyes never predicted out-of-fold"
    return d, y, P


# ------------------------------------------------------------------ metrics
def r2_rows(y, P):                       # P: (R, n) -> (R,)
    sst = ((y - y.mean()) ** 2).sum()
    return 1 - ((P - y[None, :]) ** 2).sum(1) / sst


def rho_rows(y, P):
    ry = rankdata(y); rp = np.apply_along_axis(rankdata, 1, P)
    ry = (ry - ry.mean()) / ry.std(); rp = (rp - rp.mean(1, keepdims=True)) / rp.std(1, keepdims=True)
    return (rp * ry[None, :]).mean(1)


def mae_rows(y, P):
    return np.abs(P - y[None, :]).mean(1)


METRICS = {"R2": r2_rows, "rho": rho_rows, "MAE": mae_rows}


def boot_index(subj, B, seed):
    rng = np.random.default_rng(seed)
    S = np.unique(subj); idx = {s: np.where(subj == s)[0] for s in S}
    return [np.concatenate([idx[s] for s in rng.choice(S, len(S), replace=True)]) for _ in range(B)]


def load_run(folder):
    o = pd.read_csv(folder / "oof_reg.csv").sort_values(["cv_rep", "row_index"])
    R = int(o.cv_rep.max()) + 1
    P = np.vstack([o[o.cv_rep == r].y_pred.values for r in range(R)])
    first = o[o.cv_rep == 0]
    return dict(P=P, y=first.y_true.values, subj=first.subject_code.values, row=first.row_index.values)


# ------------------------------------------------------------------ commands
def cmd_run(a):
    st = require_signoff(a.allow_unsigned)
    E, info = eye_table(a.export, a.bands44)
    S = feature_sets(E)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    write_run_info(out, a, a.export, extra=dict(data_info=info, signoff=st, sets={k: len(v) for k, v in S.items()}))
    log(f"eyes={info['n_eyes']} subjects={info['n_subjects']} sets={ {k: len(v) for k, v in S.items()} }")
    for outcome, sname, kind in RUNS:
        if sname not in S:
            log(f"skip {outcome}/{sname}/{kind}: set unavailable (no ofa44.csv)"); continue
        f = out / outcome / sname.replace("+", "_") / kind
        if (f / "summary.json").exists():
            log(f"skip {outcome}/{sname}/{kind} (done)"); continue
        f.mkdir(parents=True, exist_ok=True)
        log(f"== {outcome} / {sname} ({len(S[sname])} features) / {kind}")
        d, y, P = run_one(E, S[sname], outcome, kind, a.repeats, a.k, a.seed, a.rf_trees, a.n_jobs)
        subj_code = pd.factorize(d["Subject"])[0]                      # pseudonymous, HPC-only file
        pd.DataFrame(dict(cv_rep=np.repeat(np.arange(len(P)), len(y)), row_index=np.tile(np.arange(len(y)), len(P)),
                          subject_code=np.tile(subj_code, len(P)), y_true=np.tile(y, len(P)), y_pred=P.ravel())) \
            .to_csv(f / "oof_reg.csv", index=False)
        summ = dict(outcome=outcome, set=sname, model=kind, n_features=len(S[sname]), n_eyes=len(y),
                    n_subjects=int(d.Subject.nunique()), repeats=a.repeats, k=a.k, seed=a.seed,
                    **{m: float(fn(y, P).mean()) for m, fn in METRICS.items()})
        json.dump(summ, open(f / "summary.json", "w"), indent=1)
        log(f"   R2={summ['R2']:.3f} rho={summ['rho']:.3f} MAE={summ['MAE']:.3f}")
    log("DONE p2_function run")


def cmd_aggregate(a):
    out = Path(a.out)
    runs = {}
    for js in sorted(out.glob("*/*/*/summary.json")):
        s = json.load(open(js)); runs[(s["outcome"], s["set"], s["model"])] = (s, js.parent)
    if not runs:
        raise SystemExit("no finished runs")
    rows = []
    cache = {}
    for key, (s, folder) in sorted(runs.items()):
        R = cache.setdefault(key, load_run(folder))
        bi = boot_index(R["subj"], a.n_boot, a.seed)
        row = dict(outcome=s["outcome"], set=s["set"], model=s["model"], n_features=s["n_features"],
                   n_eyes=s["n_eyes"], n_subjects=s["n_subjects"])
        for m, fn in METRICS.items():
            bs = np.array([fn(R["y"][i], R["P"][:, i]).mean() for i in bi])
            row.update({m: s[m], f"{m}_lo": np.percentile(bs, 2.5), f"{m}_hi": np.percentile(bs, 97.5)})
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "SUMMARY.csv", index=False)

    crow = []
    for cid, role, outcome, kind, A, B, qtxt in CONTRASTS:
        ka, kb = (outcome, A, kind), (outcome, B, kind)
        if ka not in runs or kb not in runs:
            crow.append(dict(id=cid, role=role, question=qtxt, outcome=outcome, model=kind, set_A=A, set_B=B,
                             conclusion="NOT RUN")); continue
        RA, RB = cache[ka], cache[kb]
        if not (np.array_equal(RA["row"], RB["row"]) and np.allclose(RA["y"], RB["y"])
                and np.array_equal(RA["subj"], RB["subj"])):
            raise SystemExit(f"{cid}: runs not paired")
        Rn = min(len(RA["P"]), len(RB["P"]))
        bi = boot_index(RA["subj"], a.n_boot, a.seed + 1)
        row = dict(id=cid, role=role, question=qtxt, outcome=outcome, model=kind, set_A=A, set_B=B,
                   n_eyes=len(RA["y"]))
        for m, fn in METRICS.items():
            dobs = (fn(RA["y"], RA["P"][:Rn]) - fn(RB["y"], RB["P"][:Rn])).mean()
            bs = np.array([(fn(RA["y"][i], RA["P"][:Rn, i]) - fn(RB["y"][i], RB["P"][:Rn, i])).mean() for i in bi])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row.update({f"{m}_A": runs[ka][0][m], f"{m}_B": runs[kb][0][m], f"d{m}": dobs, f"d{m}_lo": lo,
                        f"d{m}_hi": hi, f"d{m}_boot_p": min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))})
        row["conclusion"] = ("A adds (dR2 > 0)" if row["dR2_lo"] > 0 else
                             "A worse (dR2 < 0)" if row["dR2_hi"] < 0 else "no difference detected")
        crow.append(row)
    C = pd.DataFrame(crow)
    C.to_csv(out / "CONTRASTS.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 48)
    show = [c for c in ["id", "role", "outcome", "model", "set_A", "set_B", "n_eyes", "R2_A", "R2_B", "dR2", "dR2_lo",
                        "dR2_hi", "drho", "drho_lo", "drho_hi", "conclusion"] if c in C.columns]
    md = ["# Paper 2, P1: functional validity (out-of-fold, subject-grouped repeated CV)", "",
          "R2 = out-of-fold R^2 (can be < 0); rho = Spearman(prediction, observed); 95% CI = subject-clustered "
          "bootstrap.", "Decision (plan): P1 is the only confirmatory test; all others are reported with CIs and no "
          "significance claims.", "",
          "## Contrasts (d = A - B, paired)", "~~~", C[show].round(3).to_string(index=False), "~~~", "",
          "## All runs", "~~~",
          summ[["outcome", "set", "model", "n_features", "n_eyes", "R2", "R2_lo", "R2_hi", "rho", "rho_lo", "rho_hi",
                "MAE"]].round(3).to_string(index=False), "~~~"]
    (out / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    log("DONE p2_function aggregate")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "aggregate"):
        p = sub.add_parser(name)
        p.add_argument("--export", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--bands44", default=None)
        p.add_argument("--seed", type=int, default=2027)
        if name == "run":
            p.add_argument("--repeats", type=int, default=20)
            p.add_argument("--k", type=int, default=5)
            p.add_argument("--rf_trees", type=int, default=300)
            p.add_argument("--n_jobs", type=int, default=8)
            p.add_argument("--allow_unsigned", action="store_true", help="synthetic tests only")
        else:
            p.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_aggregate(a)


if __name__ == "__main__":
    main()
