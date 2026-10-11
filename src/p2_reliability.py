#!/usr/bin/env python3
"""
p2_reliability.py: P2 of docs/ANALYSIS_PLAN_PAPER2.md. Test-retest reliability of OFA (20 and 44 regions) vs MPOD.

ESTIMANDS (per measure = feature group x ring mean; session 1 vs session 2 of the same eye; n = 58 eyes)
  ICC(2,1)  two-way random, absolute agreement, single measure (src.diagnostics.icc21, reused)
  Sw        within-eye SD = sqrt(sum(d^2) / 2n), d = session2 - session1
  RC        repeatability coefficient = 2.77 Sw (95% of test-retest differences lie within +/- RC)
  bias      mean(d) (learning/fatigue effect), with 95% LoA = bias +/- 1.96 SD(d)
  RC/SDb    RC relative to the between-eye SD of eye means (scale-free; < 1 = change detectable within the
            spread of the cohort)
  per zone  ICC of every single zone, summarised as median [IQR] and % zones with ICC >= 0.75
STRATA    all eyes (primary), AREDS 1 and AREDS 2-4 (secondary)
CIs       subject-clustered bootstrap (B=2000; both eyes of a subject resampled together), percentile
CONTRASTS dICC = ICC(OFA measure) - ICC(MPOD mean), same ring, same bootstrap draws (paired)

  python -m src.p2_reliability --export data/export --out reruns_paper2/p2_reliability [--bands44 f.csv]
Outputs (aggregates only): reliability_measures.csv, reliability_zones.csv, reliability_contrasts.csv,
                           figures/icc_forest.png, REPORT.md, run_info.json
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .diagnostics import icc21
from .p2_data import (BASE, RING_NAME, load_sessions, load_rings, load_bands44, ring_summaries, ofa44_summaries,
                      write_run_info, require_signoff, log)

MPOD_REF = "mean"


def paired_sessions(sess, cols):
    """Return (A1, A2, Y, subj): arrays [n_eyes x n_cols] for repeat 1 and 2, eyes with both sessions."""
    s1 = sess[sess.Repeat == 1].set_index(["Subject", "Eye"]); s2 = sess[sess.Repeat == 2].set_index(["Subject", "Eye"])
    ix = s1.index.intersection(s2.index).sort_values()
    return (s1.loc[ix, cols].to_numpy(float), s2.loc[ix, cols].to_numpy(float), s1.loc[ix, "Y"].to_numpy(int),
            np.array([i[0] for i in ix]))


def stats_one(x1, x2):
    m = np.isfinite(x1) & np.isfinite(x2)
    x1, x2 = x1[m], x2[m]
    d = x2 - x1
    sw = np.sqrt((d ** 2).sum() / (2 * len(d)))
    sdb = np.std((x1 + x2) / 2, ddof=1)
    return dict(n=int(m.sum()), ICC=icc21(x1, x2), Sw=sw, RC=2.77 * sw, bias=d.mean(),
                LoA_lo=d.mean() - 1.96 * d.std(ddof=1), LoA_hi=d.mean() + 1.96 * d.std(ddof=1),
                SD_between=sdb, RC_over_SDb=2.77 * sw / sdb if sdb > 0 else np.nan)


def boot_draws(subj, B, seed):
    rng = np.random.default_rng(seed)
    S = np.unique(subj); idx = {s: np.where(subj == s)[0] for s in S}
    return [np.concatenate([idx[s] for s in rng.choice(S, len(S), replace=True)]) for _ in range(B)]


def forest(T, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = T[(T.stratum == "all")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.5, 0.22 * len(t) + 1.2))
    yy = np.arange(len(t))[::-1]
    col = ["tab:red" if f.startswith(("Del", "Amp")) else "tab:blue" for f in t.measure]
    ax.errorbar(t.ICC, yy, xerr=[t.ICC - t.ICC_lo, t.ICC_hi - t.ICC], fmt="none", ecolor=col, elinewidth=1)
    ax.scatter(t.ICC, yy, c=col, s=12, zorder=3)
    for v in (0.5, 0.75, 0.9):
        ax.axvline(v, color="0.8", lw=0.8, ls="--")
    ax.set_yticks(yy); ax.set_yticklabels(t.measure, fontsize=6.5)
    ax.set_xlabel("ICC(2,1), 95% CI (subject-clustered bootstrap)"); ax.set_xlim(0, 1)
    ax.set_title("Test-retest reliability (red: OFA, blue: MPOD)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bands44", default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--allow_unsigned", action="store_true", help="synthetic tests only")
    a = ap.parse_args()
    st = require_signoff(a.allow_unsigned)
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    mp, o44 = load_sessions(a.export)
    ring = load_rings(a.export); bands = load_bands44(a.bands44)
    sess = pd.concat([mp[["Subject", "Eye", "Repeat", "Y"]], ring_summaries(mp, ring), mp.iloc[:, :180]], axis=1)
    measures = [f"{b}|{r}" for b in BASE for r in ("centre", "middle", "outer", "all")]
    zone_cols = {}
    for b in BASE:
        for k in (1, 2, 3):
            zone_cols[f"{b}|{RING_NAME[k]}"] = [f"{b}_region{z}" for z in range(1, 21) if ring[z] == k]
        zone_cols[f"{b}|all"] = [f"{b}_region{z}" for z in range(1, 21)]
    if o44 is not None:
        s44 = ofa44_summaries(o44, bands)
        sess = pd.concat([sess, s44, o44.iloc[:, :88]], axis=1)
        measures += list(s44.columns)
        for b in ("Del2", "Amp2"):
            zone_cols[f"{b}|all"] = [f"{b}_region{z}" for z in range(1, 45)]
            if bands:
                for bd in sorted(set(bands.values())):
                    zone_cols[f"{b}|{bd}"] = [f"{b}_region{z}" for z in range(1, 45) if bands[z] == bd]
    write_run_info(out, a, a.export, extra=dict(signoff=st, n_measures=len(measures)))

    X1, X2, Y, subj = paired_sessions(sess, measures)
    log(f"eyes with both sessions: {len(Y)}  subjects: {len(np.unique(subj))}  measures: {len(measures)}")
    strata = {"all": np.ones(len(Y), bool), "AREDS 1": Y == 1, "AREDS 2-4": Y >= 2}
    rows, draws_icc = [], {}
    for sname, m in strata.items():
        bi = boot_draws(subj[m], a.n_boot, a.seed)
        for j, meas in enumerate(measures):
            x1, x2 = X1[m, j], X2[m, j]
            r = stats_one(x1, x2)
            bs = {k: [] for k in ("ICC", "RC", "bias")}
            for i in bi:
                s = stats_one(x1[i], x2[i])
                for k in bs:
                    bs[k].append(s[k])
            for k in bs:
                r[f"{k}_lo"], r[f"{k}_hi"] = np.nanpercentile(bs[k], [2.5, 97.5])
            if sname == "all":
                draws_icc[meas] = np.array(bs["ICC"])
            rows.append(dict(stratum=sname, measure=meas, **r))
        log(f"   stratum {sname}: n={int(m.sum())} done")
    T = pd.DataFrame(rows)
    T.to_csv(out / "reliability_measures.csv", index=False)

    # per-zone ICC (point estimates), summarised per group x ring / band
    Z1, Z2, _, _ = paired_sessions(sess, sorted({c for v in zone_cols.values() for c in v}))
    zc = sorted({c for v in zone_cols.values() for c in v}); pos = {c: i for i, c in enumerate(zc)}
    zr = []
    for meas, cols in zone_cols.items():
        icc = np.array([icc21(Z1[:, pos[c]], Z2[:, pos[c]]) for c in cols])
        zr.append(dict(measure=meas, n_zones=len(cols), zone_ICC_median=np.nanmedian(icc),
                       zone_ICC_q25=np.nanpercentile(icc, 25), zone_ICC_q75=np.nanpercentile(icc, 75),
                       pct_zones_ICC_ge_075=100 * np.nanmean(icc >= 0.75)))
    ZT = pd.DataFrame(zr)
    ZT.to_csv(out / "reliability_zones.csv", index=False)

    # pre-specified paired contrasts vs the MPOD reference (same ring, same draws)
    cr = []
    pairs = [(f"{f}|{r}", f"{MPOD_REF}|{r}") for f in ("Del", "Amp") for r in ("centre", "middle", "outer", "all")]
    if o44 is not None:
        pairs += [("Del2|all", "Del|all"), ("Amp2|all", "Amp|all")]
    for A, B in pairs:
        ia, ib = measures.index(A), measures.index(B)
        d_obs = icc21(X1[:, ia], X2[:, ia]) - icc21(X1[:, ib], X2[:, ib])
        dd = draws_icc[A] - draws_icc[B]
        lo, hi = np.nanpercentile(dd, [2.5, 97.5])
        cr.append(dict(measure_A=A, measure_B=B, dICC=d_obs, lo=lo, hi=hi,
                       conclusion="A more reliable" if lo > 0 else "B more reliable" if hi < 0 else "no difference detected"))
    CT = pd.DataFrame(cr)
    CT.to_csv(out / "reliability_contrasts.csv", index=False)
    forest(T, out / "figures" / "icc_forest.png")

    pd.set_option("display.width", 250)
    ta = T[T.stratum == "all"][["measure", "n", "ICC", "ICC_lo", "ICC_hi", "RC", "RC_lo", "RC_hi", "bias", "bias_lo",
                                "bias_hi", "RC_over_SDb"]]
    md = ["# Paper 2, P2: test-retest reliability (session 1 vs 2, eye level)", "",
          f"eyes={len(Y)}, subjects={len(np.unique(subj))}, bootstrap B={a.n_boot} (subject-clustered)", "",
          "## Ring-mean measures, all eyes", "~~~", ta.round(3).to_string(index=False), "~~~", "",
          "## Paired contrasts vs MPOD mean (same ring)", "~~~", CT.round(3).to_string(index=False), "~~~", "",
          "## Per-zone ICC", "~~~", ZT.round(3).to_string(index=False), "~~~", "",
          "## By stratum (ICC [95% CI])", "~~~",
          T.assign(ICC_CI=T.apply(lambda r: f"{r.ICC:.2f} [{r.ICC_lo:.2f}, {r.ICC_hi:.2f}]", axis=1))
           .pivot(index="measure", columns="stratum", values="ICC_CI").reindex(measures).to_string(), "~~~",
          "", "Figure: figures/icc_forest.png"]
    (out / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    log("DONE p2_reliability")


if __name__ == "__main__":
    main()
