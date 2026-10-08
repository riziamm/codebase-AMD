#!/usr/bin/env python3
"""
diagnostics.py: model-free checks of WHERE (if anywhere) the data separate AMD stages.
All outputs are aggregates (safe to share). Unit = eye (the 2 repeat sessions are averaged);
CIs = subject-clustered bootstrap (a subject's eyes are resampled together).

A  Stage-specific direction maps: AUC of AREDS-k vs AREDS-1 (k = 2, 3, 4, and 2-4 pooled) for every
   feature-group x ring and every single region. AUC > 0.5 = value HIGHER in disease.
   Plus Spearman rho of each feature-group x ring with AREDS grade 1-4 (monotonic trend).
B  Positive control (published OFA result, Rai et al. 2022): eye-level worst/mean OFA total deviation
   (Del max/mean, Amp min/mean) for each contrast.
C  Test-retest reliability: ICC(2,1) (absolute agreement) between Repeat 1 and 2, per feature-group x ring;
   fellow-eye Spearman correlation per feature-group.
Optional --ofa44: same as A/B for the 44-region wider-field OFA data (Del2/Amp2).

Usage
  python -m src.diagnostics --data reruns_eyegrouped/split/train_mpod.csv reruns_eyegrouped/split/test_mpod.csv \
      --out reruns_eyegrouped/diagnostics [--rings data/export/zone_rings.csv] [--ofa44 data/export/ofa44.csv]
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

BASE = ["mean", "median", "std", "iqr", "idr", "skew", "kurt", "Del", "Amp"]
DEFAULT_RING = {z: (1 if z <= 4 else 2 if z <= 12 else 3) for z in range(1, 21)}
RING_NAME = {1: "centre", 2: "middle", 3: "outer"}
CONTRASTS = {"early (2 vs 1)": [2], "intermediate (3 vs 1)": [3], "advanced (4 vs 1)": [4],
             "advanced (3-4 vs 1)": [3, 4], "any AMD (2-4 vs 1)": [2, 3, 4]}


def auc(score, label):
    """Mann-Whitney AUC; NaNs dropped."""
    m = ~np.isnan(score)
    s, y = score[m], label[m]
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def eye_level(df, feat_cols):
    g = df.groupby(["Subject", "Eye"])
    E = g[feat_cols].mean()
    E["Y"] = g["Y"].first()
    return E.reset_index()


def boot_subjects(E, B, seed):
    rng = np.random.default_rng(seed)
    subs = E["Subject"].unique()
    idx = {s: np.where(E["Subject"].values == s)[0] for s in subs}
    for _ in range(B):
        pick = rng.choice(subs, len(subs), replace=True)
        yield np.concatenate([idx[s] for s in pick])


def contrast_table(E, scores, B, seed, with_ci=True):
    """scores: dict name -> eye-level score array. Returns long table of AUCs per contrast."""
    rows = []
    Y = E["Y"].values
    boots = list(boot_subjects(E, B, seed)) if with_ci else []
    for cname, ks in CONTRASTS.items():
        keep = np.isin(Y, [1] + ks)
        lab = np.isin(Y, ks).astype(int)
        for name, sc in scores.items():
            a = auc(sc[keep], lab[keep])
            lo = hi = np.nan
            if with_ci:
                bs = []
                for b in boots:
                    kb = keep[b]
                    bs.append(auc(sc[b][kb], lab[b][kb]))
                bs = np.array(bs, float); bs = bs[~np.isnan(bs)]
                if len(bs):
                    lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(contrast=cname, feature=name, n_disease=int(lab[keep].sum()), n_normal=int((1 - lab[keep]).sum()),
                             AUC=a, lo=lo, hi=hi,
                             direction=("higher in disease" if a > 0.5 else "lower in disease") if not np.isnan(a) else "",
                             ci_excludes_05=bool(with_ci and not np.isnan(lo) and (lo > 0.5 or hi < 0.5))))
    return pd.DataFrame(rows)


def icc21(x1, x2):
    X = np.column_stack([x1, x2]); X = X[~np.isnan(X).any(1)]
    n, k = X.shape
    if n < 3:
        return np.nan
    gm = X.mean(); rm = X.mean(1); cm = X.mean(0)
    msr = k * ((rm - gm) ** 2).sum() / (n - 1)
    msc = n * ((cm - gm) ** 2).sum() / (k - 1)
    mse = ((X - rm[:, None] - cm[None, :] + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def heatmap(tab, contrast_list, path, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(contrast_list), figsize=(3.4 * len(contrast_list), 4.2), squeeze=False)
    for ax, c in zip(axes[0], contrast_list):
        t = tab[tab.contrast == c]
        M = np.full((len(BASE), 3), np.nan)
        for i, b in enumerate(BASE):
            for j, rg in enumerate([1, 2, 3]):
                v = t[t.feature == f"{b}|{RING_NAME[rg]}"]
                if len(v):
                    M[i, j] = v.AUC.values[0]
        im = ax.imshow(M, vmin=0.2, vmax=0.8, cmap="RdBu_r", aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    star = "*" if t[t.feature == f"{BASE[i]}|{RING_NAME[j+1]}"].ci_excludes_05.values[0] else ""
                    ax.text(j, i, f"{M[i, j]:.2f}{star}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(3)); ax.set_xticklabels(["centre", "middle", "outer"]); ax.set_yticks(range(len(BASE)))
        ax.set_yticklabels(BASE if ax is axes[0][0] else []); ax.set_title(c, fontsize=9)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="AUC (red = higher in disease)")
    fig.suptitle(title + "   (* = 95% CI excludes 0.5)", fontsize=10)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rings", default=None, help="zone_rings.csv (region, ring) from mat_to_csv")
    ap.add_argument("--ofa44", default=None, help="ofa44.csv from mat_to_csv (optional)")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(p) for p in a.data], ignore_index=True)
    ring = DEFAULT_RING
    if a.rings:
        r = pd.read_csv(a.rings); ring = dict(zip(r.region.astype(int), r.ring.astype(int)))
        if ring != DEFAULT_RING:
            print("NOTE: ring map from file differs from default 1-4/5-12/13-20; using the file.")
    feat = [f"{b}_region{z}" for b in BASE for z in range(1, 21)]
    E = eye_level(df, feat)
    print(f"eyes={len(E)} subjects={E.Subject.nunique()} AREDS per eye={E.Y.value_counts().sort_index().to_dict()}")

    # ---- A: aggregates (feature-group x ring, and whole field)
    agg = {}
    for b in BASE:
        for rg in (1, 2, 3):
            cols = [f"{b}_region{z}" for z in range(1, 21) if ring[z] == rg]
            agg[f"{b}|{RING_NAME[rg]}"] = E[cols].mean(1).values
        agg[f"{b}|all"] = E[[f"{b}_region{z}" for z in range(1, 21)]].mean(1).values
    A = contrast_table(E, agg, a.n_boot, a.seed)
    rho = []
    for name, sc in agg.items():
        m = ~np.isnan(sc)
        r_, p_ = spearmanr(sc[m], E.Y.values[m])
        rho.append(dict(feature=name, spearman_rho_vs_AREDS=r_, p=p_))
    A = A.merge(pd.DataFrame(rho), on="feature", how="left")
    A.to_csv(out / "A_stage_direction_rings.csv", index=False)
    Z = contrast_table(E, {c: E[c].values for c in feat}, 0, a.seed, with_ci=False)
    Z["ring"] = Z.feature.str.extract(r"region(\d+)")[0].astype(int).map(ring).map(RING_NAME)
    Z.to_csv(out / "A_stage_direction_regions.csv", index=False)
    heatmap(A, ["early (2 vs 1)", "intermediate (3 vs 1)", "advanced (4 vs 1)", "any AMD (2-4 vs 1)"],
            out / "figures" / "A_stage_direction_heatmap.png", "Stage-specific univariate AUC by feature x ring (eye level)")

    # direction-reversal summary: early vs advanced on opposite sides of 0.5
    ring_rows = A[A.feature.str.contains(r"\|(centre|middle|outer)$")]
    piv = ring_rows.pivot(index="feature", columns="contrast", values="AUC")
    sig = ring_rows.pivot(index="feature", columns="contrast", values="ci_excludes_05")
    opp = ((piv["early (2 vs 1)"] - .5) * (piv["advanced (4 vs 1)"] - .5)) < 0
    # a reversal counts only if BOTH contrasts have CIs excluding 0.5, in opposite directions
    rev = piv[opp & sig["early (2 vs 1)"].astype(bool) & sig["advanced (4 vs 1)"].astype(bool)]

    # ---- B: positive control (OFA total deviations)
    pc = {"Del mean TD": E[[f"Del_region{z}" for z in range(1, 21)]].mean(1).values,
          "Del worst (max) TD": E[[f"Del_region{z}" for z in range(1, 21)]].max(1).values,
          "Amp mean TD": E[[f"Amp_region{z}" for z in range(1, 21)]].mean(1).values,
          "Amp worst (min) TD": E[[f"Amp_region{z}" for z in range(1, 21)]].min(1).values}
    for rg in (1, 2, 3):
        cols = [z for z in range(1, 21) if ring[z] == rg]
        pc[f"Del mean TD {RING_NAME[rg]}"] = E[[f"Del_region{z}" for z in cols]].mean(1).values
        pc[f"Amp mean TD {RING_NAME[rg]}"] = E[[f"Amp_region{z}" for z in cols]].mean(1).values
    Bt = contrast_table(E, pc, a.n_boot, a.seed)
    Bt.to_csv(out / "B_positive_control_OFA.csv", index=False)

    # ---- C: reliability
    rel = []
    R1 = df[df.Repeat == df.Repeat.min()].set_index(["Subject", "Eye"]).sort_index()
    R2 = df[df.Repeat == df.Repeat.max()].set_index(["Subject", "Eye"]).sort_index()
    for b in BASE:
        for rg in (1, 2, 3):
            cols = [f"{b}_region{z}" for z in range(1, 21) if ring[z] == rg]
            icc_ring = icc21(R1[cols].mean(1).values, R2[cols].mean(1).values)
            icc_zone = np.nanmedian([icc21(R1[c].values, R2[c].values) for c in cols])
            rel.append(dict(feature=b, ring=RING_NAME[rg], ICC_ring_mean=icc_ring, ICC_median_per_zone=icc_zone))
    C = pd.DataFrame(rel)
    fe = []
    for b in BASE:
        cols = [f"{b}_region{z}" for z in range(1, 21)]
        L = E[E.Eye == 1].set_index("Subject")[cols].mean(1); Rr = E[E.Eye == 2].set_index("Subject")[cols].mean(1)
        j = L.index.intersection(Rr.index)
        fe.append(dict(feature=b, fellow_eye_spearman=spearmanr(L[j], Rr[j])[0]))
    C = C.merge(pd.DataFrame(fe), on="feature")
    C.to_csv(out / "C_reliability_ICC.csv", index=False)

    # ---- optional 44-region OFA
    extra = ""
    if a.ofa44:
        o = pd.read_csv(a.ofa44)
        oc = [c for c in o.columns if c.startswith(("Del2_", "Amp2_"))]
        E44 = eye_level(o, oc)
        s44 = {"Del2 mean TD": E44[[c for c in oc if c.startswith("Del2_")]].mean(1).values,
               "Del2 worst (max) TD": E44[[c for c in oc if c.startswith("Del2_")]].max(1).values,
               "Amp2 mean TD": E44[[c for c in oc if c.startswith("Amp2_")]].mean(1).values,
               "Amp2 worst (min) TD": E44[[c for c in oc if c.startswith("Amp2_")]].min(1).values}
        contrast_table(E44, s44, a.n_boot, a.seed).to_csv(out / "D_ofa44_summary.csv", index=False)
        contrast_table(E44, {c: E44[c].values for c in oc}, 0, a.seed, with_ci=False).to_csv(out / "D_ofa44_regions.csv", index=False)
        extra = "\n## 44-region OFA (wider field)\nSee D_ofa44_summary.csv and D_ofa44_regions.csv.\n"

    # ---- report
    def top(t, c, n=8):
        x = t[t.contrast == c].assign(dist=lambda d: (d.AUC - .5).abs()).sort_values("dist", ascending=False).head(n)
        return x[["feature", "AUC", "lo", "hi", "direction", "ci_excludes_05"]].round(3).to_string(index=False)
    md = [f"# Diagnostics  (eyes={len(E)}, subjects={E.Subject.nunique()}, bootstrap B={a.n_boot}, subject-clustered)", "",
          "## A. Strongest feature x ring effects per contrast (AUC>0.5 = higher in disease)"]
    for c in CONTRASTS:
        md += ["", f"### {c}", "```", top(A[A.feature.str.contains(r'\|(centre|middle|outer)$')], c), "```"]
    md += ["", f"### Significant direction reversals (early vs advanced, both CIs exclude 0.5, opposite sides): {len(rev)} of {len(piv)} feature x ring cells",
           "```", rev[["early (2 vs 1)", "advanced (4 vs 1)"]].round(3).to_string() if len(rev) else "none", "```",
           "", "## B. Positive control: OFA total deviations", "```",
           Bt[["contrast", "feature", "AUC", "lo", "hi", "ci_excludes_05"]].round(3).to_string(index=False), "```",
           "", "## C. Test-retest reliability ICC(2,1)  (<0.5 poor, 0.5-0.75 moderate, 0.75-0.9 good, >0.9 excellent)", "```",
           C.round(3).to_string(index=False), "```", extra,
           "", "Figure: figures/A_stage_direction_heatmap.png"]
    (out / "REPORT.md").write_text("\n".join(md) + "\n")
    json.dump(dict(eyes=len(E), subjects=int(E.Subject.nunique()), n_boot=a.n_boot, ring_map=ring), open(out / "config.json", "w"), indent=1)
    print("\n".join(md[:40]))
    print(f"\nwrote {out}/REPORT.md, A_*.csv, B_*.csv, C_*.csv, figures/")


if __name__ == "__main__":
    main()
