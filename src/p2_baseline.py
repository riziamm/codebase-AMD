#!/usr/bin/env python3
"""
p2_baseline.py: Step S0 of docs/ANALYSIS_PLAN_PAPER2.md (runs first; descriptive + safety checks).

  S0.1 Table 1 by AREDS grade (eyes, subjects, age, sex, VA, VAlc: median [IQR], missing counts)
  S0.2 AGE CONFOUND CHECK for Paper 1: AUC of age alone for any (2-4 vs 1), early (2 vs 1),
       advanced (3-4 vs 1); subject-clustered bootstrap CI. Pre-specified flag rule (plan section S0).
  S0.3 Age vs OFA total deviation and MPOD in normal eyes (TD should be ~age-independent; no acuity here).
       S0 contains no predictor-outcome association, so it may run before the plan is signed off.
  S0.4 Follow-up check: value counts of VisitN / VisitLast from demographics_raw.csv (counts only).
Outputs (aggregates only): table1.csv, age_auc.csv, age_corr.csv, visits.csv, REPORT.md, run_info.json

  python -m src.p2_baseline --export data/export --out reruns_paper2/p2_baseline [--raw demographics_raw.csv]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .p2_data import eye_table, write_run_info, log

TASKS = {"any (2-4 vs 1)": ([1, 2, 3, 4], [2, 3, 4]), "early (2 vs 1)": ([1, 2], [2]),
         "advanced (3-4 vs 1)": ([1, 3, 4], [3, 4])}


def q(x):
    x = pd.Series(x).dropna()
    return f"{x.median():.2f} [{x.quantile(.25):.2f}, {x.quantile(.75):.2f}]" if len(x) else "NA"


def subj_boot(E, B, seed):
    rng = np.random.default_rng(seed)
    S = E.Subject.unique(); idx = {s: np.where(E.Subject.values == s)[0] for s in S}
    for _ in range(B):
        yield np.concatenate([idx[s] for s in rng.choice(S, len(S), replace=True)])


def table1(E):
    rows = []
    for name, m in [("all", np.ones(len(E), bool))] + [(f"AREDS {g}", (E.Y == g).values) for g in (1, 2, 3, 4)]:
        e = E[m]; subj = e.drop_duplicates("Subject")
        rows.append(dict(group=name, n_eyes=len(e), n_subjects=subj.Subject.nunique(),
                         age_median_IQR=q(subj.Age), sex_counts_subjects=str(subj.Sex.value_counts().to_dict()),
                         VA_median_IQR=q(e.VA), VA_missing=int(e.VA.isna().sum()),
                         VAlc_median_IQR=q(e.VAlc), VAlc_missing=int(e.VAlc.isna().sum())))
    return pd.DataFrame(rows)


def age_auc(E, B, seed):
    rows = []
    for name, (keep, pos) in TASKS.items():
        e = E[E.Y.isin(keep)].reset_index(drop=True)
        y = e.Y.isin(pos).astype(int).values; a = e.Age.values
        auc = roc_auc_score(y, a)
        bs = [roc_auc_score(y[i], a[i]) for i in subj_boot(e, B, seed) if 0 < y[i].sum() < len(i)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        flag = "FLAG: age alone discriminates (report to Paper 1 authors)" if (auc >= 0.70 and lo > 0.5) else \
               "NOTE: possible age confounding" if lo > 0.5 else "no age signal detected"
        rows.append(dict(task=name, n_eyes=len(e), n_pos=int(y.sum()), AUC_age=auc, lo=lo, hi=hi, verdict=flag))
    return pd.DataFrame(rows)


def age_corr(E, B, seed):
    rows = []
    tests = []   # no predictor-outcome associations here: S0 runs BEFORE sign-off (age vs VA is reported in P1)
    nor = E[E.Y == 1].reset_index(drop=True)
    for c in ("Del|all", "Amp|all", "Del2|all", "Amp2|all", "mean|all"):
        if c in E:
            tests.append((c, nor, "normal eyes (AREDS 1)"))
    for c, e, who in tests:
        m = e[c].notna().values & e.Age.notna().values
        if m.sum() < 8:
            continue
        e2 = e[m].reset_index(drop=True)
        r = spearmanr(e2.Age, e2[c]).correlation
        bs = [spearmanr(e2.Age.values[i], e2[c].values[i]).correlation for i in subj_boot(e2, B, seed)]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rows.append(dict(variable=c, sample=who, n_eyes=int(m.sum()), spearman_age=r, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def visits(raw):
    if not raw or not Path(raw).exists():
        return pd.DataFrame()
    T = pd.read_csv(raw, usecols=lambda c: c in ("VisitN", "VisitLast"))   # SbjID is never read
    rows = []
    for c in T.columns:
        for v, n in T[c].value_counts(dropna=False).sort_index().items():
            rows.append(dict(variable=c, value=v, n_subjects=int(n)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default=None, help="demographics_raw.csv (only VisitN/VisitLast are read)")
    ap.add_argument("--bands44", default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2027)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    E, info = eye_table(a.export, a.bands44)
    log(f"eyes={info['n_eyes']} subjects={info['n_subjects']} AREDS={info['areds_per_eye']} sex={info['sex_coding']}")
    write_run_info(out, a, a.export, extra=dict(data_info=info))
    t1, aa, ac, vs = table1(E), age_auc(E, a.n_boot, a.seed), age_corr(E, a.n_boot, a.seed), visits(a.raw)
    t1.to_csv(out / "table1.csv", index=False); aa.to_csv(out / "age_auc.csv", index=False)
    ac.to_csv(out / "age_corr.csv", index=False)
    if len(vs):
        vs.to_csv(out / "visits.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
    md = ["# Paper 2 step S0: baseline and safety checks", "", f"data: {info}", "",
          "## S0.1 Table 1 (median [IQR]; age/sex at subject level)", "~~~", t1.to_string(index=False), "~~~", "",
          "## S0.2 Age-only discrimination (AUC of age; subject-clustered bootstrap 95% CI)", "~~~",
          aa.round(3).to_string(index=False), "~~~", "",
          "## S0.3 Spearman correlations with age", "~~~", ac.round(3).to_string(index=False), "~~~", "",
          "## S0.4 Follow-up fields (counts of subjects)", "~~~",
          vs.to_string(index=False) if len(vs) else "demographics_raw.csv not given", "~~~"]
    (out / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    log("DONE p2_baseline")


if __name__ == "__main__":
    main()
