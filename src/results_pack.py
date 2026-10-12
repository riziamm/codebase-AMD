#!/usr/bin/env python3
"""
results_pack.py: Paper 1 (ML: Health) tables + figures from the repeated-CV, OFA-value, SHAP and diagnostics outputs.

Inputs (all under --root, default reruns_eyegrouped):
  repeated_cv/<folder>/<model>/summary.json (+ oof.csv for ROC curves; only on the HPC)
  ofa_value/COMPARISONS.csv   explain/<task>_<model>/   diagnostics/
Outputs (--out, default <root>/results_pack): aggregates only
  tables/T1_performance.csv|md   T2_feature_sets.csv|md   T3_ofa_value_paired.csv|md   T4_models.csv|md
  figures/F_model_comparison.png  F_feature_sets.png  F_leave_one_group_out.png  F_roc_<task>.png
          F_zone_overall_<task>_rf_<layout>_<scale>.png (all-feature, MPOD | OFA side by side)
          + copies of the SHAP figures under manuscript names
  RESULTS_PACK.md: key numbers as sentences + manuscript mapping
Usage:  python -m src.results_pack --root reruns_eyegrouped
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd

from .explain import zone_geometry

BASE = ["mean", "median", "std", "iqr", "idr", "skew", "kurt", "Del", "Amp"]
MPOD = "mean,median,std,iqr,idr,skew,kurt"
TASK_LABEL = {"any": "Any AMD (AREDS 2-4 vs 1)", "early": "Early AMD (AREDS 2 vs 1)", "advanced": "Intermediate/advanced (AREDS 3-4 vs 1)"}
MODEL_LABEL = {"lr": "Logistic regression", "rf": "Random forest", "hybrid_zone": "Zone-token CNN-Transformer",
               "hybrid_orig": "CNN-Transformer (original)"}
SET_LABEL = {"all": "MPOD + OFA (all 9)", MPOD: "MPOD only (7)", "Del,Amp": "OFA only (Del + Amp)", "Del": "OFA delay only",
             "Amp": "OFA sensitivity only", "coupling": "Structure-function coupling only"}
TASKS = ["any", "early", "advanced"]


def ci(a, lo, hi, d=2):
    return f"{a:.{d}f} [{lo:.{d}f}-{hi:.{d}f}]" if pd.notna(a) else "-"


def pfmt(p):
    return "-" if pd.isna(p) else ("<0.01" if p < 0.01 else f"{p:.2f}")


def load_runs(rcv):
    rows = []
    for f in sorted(Path(rcv).glob("*/*/summary.json")):
        s = json.load(open(f))
        rows.append(dict(folder=f.parent.parent.name, model=s["model"], task=s.get("task", "any"), groups=s.get("groups", "all"),
                         rings=s.get("rings", "all"), group_by=s["group_by"], n_eyes=s["n_eyes"], AUC=s["auc_mean"],
                         lo=s["auc_ci_lo"], hi=s["auc_ci_hi"], p=s["p_above"], perm_n=s.get("perm_n", 0),
                         sens=s["sensitivity"], spec=s["specificity"], n_features=s.get("n_features")))
    return pd.DataFrame(rows)


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values]
    return "\n".join(out)


def save(df, path, title):
    df.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".md").write_text(f"### {title}\n\n" + md_table(df) + "\n")


# ------------------------------------------------------------------ tables
def table_performance(R):
    m = R[(R.groups == "all") & (R.rings == "all")]
    rows = []
    for t in TASKS:
        for mod in ["lr", "rf", "hybrid_zone", "hybrid_orig"]:
            e = m[(m.task == t) & (m.model == mod) & (m.group_by == "eye")]
            s = m[(m.task == t) & (m.model == mod) & (m.group_by == "subject")]
            if e.empty and s.empty:
                continue
            e = e.iloc[0] if len(e) else None; s = s.iloc[0] if len(s) else None
            rows.append({"Task": TASK_LABEL[t], "Model": MODEL_LABEL[mod],
                         "AUC [95% CI] (eye-grouped)": ci(e.AUC, e.lo, e.hi) if e is not None else "-",
                         "p (perm.)": pfmt(e.p) if e is not None else "-",
                         "Sens / Spec at 0.5": f"{e.sens:.2f} / {e.spec:.2f}" if e is not None else "-",
                         "AUC [95% CI] (subject-grouped)": ci(s.AUC, s.lo, s.hi) if s is not None else "-",
                         "p (perm.) ": pfmt(s.p) if s is not None else "-",
                         "n eyes": int(e.n_eyes if e is not None else s.n_eyes)})
    return pd.DataFrame(rows)


def table_feature_sets(R):
    e = R[(R.group_by == "eye")]
    rows = []
    for t in TASKS:
        for g, lab in SET_LABEL.items():
            for mod in ["lr", "rf"]:
                x = e[(e.task == t) & (e.groups == g) & (e.rings == "all") & (e.model == mod)]
                if len(x):
                    x = x.iloc[0]
                    rows.append({"Task": TASK_LABEL[t], "Feature set": lab, "Model": MODEL_LABEL[mod],
                                 "AUC [95% CI]": ci(x.AUC, x.lo, x.hi), "p (perm.)": pfmt(x.p) if x.perm_n else "-"})
        for rg in ["centre", "middle", "outer"]:
            x = e[(e.task == t) & (e.groups == "all") & (e.rings == rg) & (e.model == "lr")]
            if len(x):
                x = x.iloc[0]
                rows.append({"Task": TASK_LABEL[t], "Feature set": f"{rg.capitalize()} ring only (all 9 features)",
                             "Model": MODEL_LABEL["lr"], "AUC [95% CI]": ci(x.AUC, x.lo, x.hi), "p (perm.)": pfmt(x.p) if x.perm_n else "-"})
    return pd.DataFrame(rows)


def table_paired(C):
    if C is None or C.empty:
        return pd.DataFrame()
    C = C.copy()
    C["Task"] = C.task.map(TASK_LABEL); C["Model"] = C.model.map(MODEL_LABEL)
    C["dAUC [95% CI]"] = [ci(a, b, c, 3) for a, b, c in zip(C.dAUC, C.lo, C.hi)]
    C["AUC A / B"] = [f"{a:.2f} / {b:.2f}" for a, b in zip(C.AUC_A, C.AUC_B)]
    return C[["Task", "Model", "comparison", "AUC A / B", "dAUC [95% CI]", "conclusion"]].rename(columns={"comparison": "Comparison (A - B)"})


def table_models():
    rows = [dict(Model=MODEL_LABEL["lr"], Tuned="C in {0.001, 0.01, 0.1, 1} (inner 3-fold, AUC)", Notes="L2, class-balanced, liblinear"),
            dict(Model=MODEL_LABEL["rf"], Tuned="max_depth in {3, None} x min_samples_leaf in {1, 3}", Notes="300 trees, balanced_subsample"),
            dict(Model=MODEL_LABEL["hybrid_zone"], Tuned="none (pre-specified)", Notes="20 zone tokens x 9 features; zone + ring embeddings; 1 encoder layer (d=16, 2 heads); Conv1d over zones (16 ch); AdamW lr 3e-3, wd 1e-2, 80 epochs, batch 16, pos_weight = n_neg/n_pos"),
            dict(Model=MODEL_LABEL["hybrid_orig"], Tuned="none (configuration of the original study)", Notes="parallel CNN + Transformer on a single 180-d token; d=16, 2 heads, 1 layer, dropout 0.3")]
    try:
        from .repeated_cv import build_dl
        for r, k in zip(rows[2:], ["hybrid_zone", "hybrid_orig"]):
            r["Parameters"] = f"{sum(p.numel() for p in build_dl(k).parameters()):,}"
    except (Exception, SystemExit):   # torch / dl_pipeline_gen not importable -> values from the study run
        rows[2]["Parameters"], rows[3]["Parameters"] = "3,265", "602,993"
    rows[0]["Parameters"] = "181"; rows[1]["Parameters"] = "-"
    return pd.DataFrame(rows)[["Model", "Parameters", "Tuned", "Notes"]]


# ------------------------------------------------------------------ figures (one style: src/figstyle.py)
def _get(R, t, mod, g="all", rg="all", gb="eye"):
    x = R[(R.task == t) & (R.model == mod) & (R.groups == g) & (R.rings == rg) & (R.group_by == gb)]
    return x.iloc[0] if len(x) else None


def fig3_performance(R, path):
    """Fig 3: (a) AUC by model, (b) feature-set and ring ablation (LR); intermediate/advanced and any AMD."""
    from . import figstyle as fs
    import matplotlib.pyplot as plt
    fs.apply()
    tasks = [("advanced", "Intermediate/advanced vs normal"), ("any", "Any AMD vs normal")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [4, 6]})
    models = ["lr", "rf", "hybrid_zone", "hybrid_orig"]
    V = [[getattr(_get(R, t, m), "AUC", np.nan) for m in models] for t, _ in tasks]
    L = [[getattr(_get(R, t, m), "lo", np.nan) for m in models] for t, _ in tasks]
    H = [[getattr(_get(R, t, m), "hi", np.nan) for m in models] for t, _ in tasks]
    fs.auc_bars(a1, [fs.MODEL_SHORT[m] for m in models], [n for _, n in tasks], V, L, H, [fs.BAR, fs.BAR_LIGHT])
    a1.set_title("(a) Models (all features)", fontsize=8.5, loc="left")
    sets = [("all", "all", "MPOD+\nOFA"), (MPOD, "all", "MPOD\nonly"), ("Del,Amp", "all", "OFA\nonly"),
            ("all", "centre", "Centre\nring"), ("all", "middle", "Middle\nring"), ("all", "outer", "Outer\nring")]
    V = [[getattr(_get(R, t, "lr", g, rg), "AUC", np.nan) for g, rg, _ in sets] for t, _ in tasks]
    L = [[getattr(_get(R, t, "lr", g, rg), "lo", np.nan) for g, rg, _ in sets] for t, _ in tasks]
    H = [[getattr(_get(R, t, "lr", g, rg), "hi", np.nan) for g, rg, _ in sets] for t, _ in tasks]
    fs.auc_bars(a2, [l for _, _, l in sets], [n for _, n in tasks], V, L, H, [fs.BAR, fs.BAR_LIGHT], ylabel="")
    a2.set_title("(b) Feature sets and rings (logistic regression)", fontsize=8.5, loc="left")
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(pad=0.2, w_pad=0.6)
    fs.save(fig, path)


def fig_logo(C, path):
    from . import figstyle as fs
    import matplotlib.pyplot as plt
    B = C[C.comparison.str.startswith("B:")].copy()
    if B.empty:
        return
    fs.apply()
    B["group"] = B.comparison.str.extract(r"importance of (\w+)")[0]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharex=True, sharey=True)
    for ax, t in zip(axes, TASKS):
        for j, mod in enumerate(["lr", "rf"]):
            x = B[(B.task == t) & (B.model == mod)].set_index("group").reindex(BASE)
            y = -np.arange(len(BASE)) + (j - 0.5) * 0.3
            ax.errorbar(x.dAUC, y, xerr=[x.dAUC - x.lo, x.hi - x.dAUC], fmt="o", ms=3, capsize=1.5, lw=0.7,
                        color=fs.MODEL_COLOR[mod], label=MODEL_LABEL[mod])
        ax.axvline(0, color="gray", ls="--", lw=0.8); ax.set_title(TASK_LABEL[t], fontsize=7.5)
        ax.set_yticks(-np.arange(len(BASE))); ax.set_yticklabels(BASE); ax.set_xlabel("dAUC (all - all minus group)")
    axes[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout(pad=0.2, w_pad=0.4)
    fs.save(fig, path)


def fig_roc(rcv, task, path):
    from sklearn.metrics import roc_curve, roc_auc_score
    from . import figstyle as fs
    folder = Path(rcv) / f"eye_{task}_all_allrings"; grid = np.linspace(0, 1, 201); curves = []
    for mod in ["rf", "lr", "hybrid_zone", "hybrid_orig"]:
        f = folder / mod / "oof.csv"
        if not f.exists():
            continue
        o = pd.read_csv(f); T, A = [], []
        for _, d in o.groupby("cv_rep"):
            fpr, tpr, _ = roc_curve(d.y_true, d.y_prob); T.append(np.interp(grid, fpr, tpr)); A.append(roc_auc_score(d.y_true, d.y_prob))
        T = np.array(T); T[:, 0] = 0
        curves.append((MODEL_LABEL[mod], grid, T.mean(0), np.percentile(T, 2.5, 0), np.percentile(T, 97.5, 0), np.mean(A), fs.MODEL_COLOR[mod]))
    if not curves:
        return False
    fs.roc(curves, path, title=TASK_LABEL[task])
    return True


def shap_figures(exp, F):
    """Fig 1 + Fig 4 (RF, intermediate/advanced) and appendix SHAP figures, all in the shared style."""
    from . import figstyle as fs
    made = []
    for t in ["advanced", "any"]:
        for mdl in ["rf", "hybrid_zone"]:
            d = exp / f"{t}_{mdl}"
            if not (d / "importance_features.csv").exists():
                continue
            imp = pd.read_csv(d / "importance_features.csv")
            gr = pd.read_csv(d / "importance_group_ring.csv")
            main = (t == "advanced" and mdl == "rf")
            pre = "Fig4" if main else f"FigA_SHAP_{t}_{mdl}_"
            fs.group_bar(imp.groupby("group").mean_abs_shap.sum(), F / f"{pre}a_group_importance.png")
            fs.ring_bars(gr[gr.ring != "all"], F / f"{pre}b_ring_averaged.png")
            fs.zone_grid(imp, F / f"{pre}c_zone_maps_linear.png", log=False)
            fs.zone_grid(imp, F / f"{pre}d_zone_maps_log.png", log=True)
            if main:
                tot = imp.groupby("region").mean_abs_shap.sum().to_dict()
                fs.zone_pair(tot, F / "Fig1_zone_importance_circular.png")
                fs.zone_pair(tot, F / "FigA3_zone_importance_square.png", square=True)
                fs.zone_grid(imp, F / "FigA_SHAP_square_zone_maps_log.png", log=True, square=True)
                if (d / "local_examples.csv").exists():
                    fs.local_bars(pd.read_csv(d / "local_examples.csv"), F / "FigA_local_explanations.png")
            made.append(f"{t}_{mdl}")
    return made


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="reruns_eyegrouped")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root); out = Path(a.out or root / "results_pack")
    (out / "tables").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(parents=True, exist_ok=True)
    rcv = root / "repeated_cv"
    R = load_runs(rcv)
    C = pd.read_csv(root / "ofa_value" / "COMPARISONS.csv") if (root / "ofa_value" / "COMPARISONS.csv").exists() else None

    T1 = table_performance(R); save(T1, out / "tables" / "T1_performance", "Discrimination by task and model (repeated nested CV)")
    T2 = table_feature_sets(R); save(T2, out / "tables" / "T2_feature_sets", "Feature-set and ring ablation (eye-grouped)")
    T3 = table_paired(C); save(T3, out / "tables" / "T3_ofa_value_paired", "Pre-specified paired comparisons (dAUC = A - B)")
    T4 = table_models(); save(T4, out / "tables" / "T4_models", "Models and configurations")

    F = out / "figures"
    for old in F.glob("*.png"):          # never mix old- and new-style files
        old.unlink()
    fig3_performance(R, F / "Fig3_performance_ablation.png")
    if C is not None:
        fig_logo(C, F / "FigA_leave_one_group_out.png")
    roc_done = [t for t in TASKS if fig_roc(rcv, t, F / f"FigA2_roc_{t}.png")]
    exp = root / "explain"
    shap_figures(exp, F)
    for t in ["advanced", "any"]:
        for mdl in ["rf", "hybrid_zone"]:
            sj = exp / f"{t}_{mdl}" / "stability.json"
            if sj.exists():
                shutil.copy2(sj, out / "tables" / f"SHAP_stability_{t}_{mdl}.json")
    cal = root / "calibration"
    for f in cal.glob("calibration_dca_*.png"):
        shutil.copy2(f, F / f"FigA_{f.name}")
    if (cal / "CALIBRATION.csv").exists():
        shutil.copy2(cal / "CALIBRATION.csv", out / "tables" / "TA7_calibration.csv")
    ag = root / "age_check" / "COMPARISONS_AGE.csv"
    if ag.exists():
        shutil.copy2(ag, out / "tables" / "TA6_age_comparisons.csv")
        shutil.copy2(root / "age_check" / "AUC_BY_SET.csv", out / "tables" / "TA6_age_auc_by_set.csv")
    diag = root / "diagnostics" / "figures" / "A_stage_direction_heatmap.png"
    if diag.exists():
        shutil.copy2(diag, F / "FigA_univariate_stage_ring_heatmap.png")

    # key numbers as sentences
    def get(t, mod, gb="eye", g="all", rg="all"):
        x = R[(R.task == t) & (R.model == mod) & (R.group_by == gb) & (R.groups == g) & (R.rings == rg)]
        return x.iloc[0] if len(x) else None
    lines = ["# Results pack: Paper 1 (generated by src/results_pack.py)", "", "## Key numbers (copy-ready)"]
    for t in TASKS:
        for mod in ["rf", "lr", "hybrid_zone", "hybrid_orig"]:
            e, s = get(t, mod), get(t, mod, "subject")
            if e is not None:
                lines.append(f"- {TASK_LABEL[t]}, {MODEL_LABEL[mod]}: AUC {ci(e.AUC, e.lo, e.hi)} (permutation p {pfmt(e.p)}), "
                             f"sensitivity {e.sens:.2f}, specificity {e.spec:.2f}"
                             + (f"; subject-grouped {ci(s.AUC, s.lo, s.hi)} (p {pfmt(s.p)})" if s is not None else ""))
    for t in ["advanced", "any"]:
        rr = {rg: get(t, "lr", rg=rg) for rg in ["centre", "middle", "outer"]}
        if all(v is not None for v in rr.values()):
            lines.append(f"- Ring-only models ({TASK_LABEL[t]}, LR): centre {ci(rr['centre'].AUC, rr['centre'].lo, rr['centre'].hi)}, "
                         f"middle {ci(rr['middle'].AUC, rr['middle'].lo, rr['middle'].hi)}, outer {ci(rr['outer'].AUC, rr['outer'].lo, rr['outer'].hi)}")
        sj = exp / f"{t}_rf" / "stability.json"
        if sj.exists():
            s = json.load(open(sj)); z = s["mean_abs_shap_per_zone_by_ring"]
            lines.append(f"- SHAP ({TASK_LABEL[t]}, RF): mean |SHAP| per zone centre {z['centre']:.4f}, middle {z['middle']:.4f}, "
                         f"outer {z['outer']:.4f}; group-ranking stability across repeats rho = {s['group_rank_stability_mean_spearman']:.2f}; "
                         f"top group per repeat {s['top_group_frequency']}")
    if C is not None:
        for t in ["any", "advanced"]:
            x = C[(C.task == t) & (C.model == "rf") & (C.comparison == "A: MPOD+Del+Amp vs MPOD")]
            if len(x):
                x = x.iloc[0]
                lines.append(f"- Pre-specified primary (RF, {TASK_LABEL[t]}): adding OFA to MPOD dAUC {x.dAUC:+.3f} [{x.lo:+.3f} to {x.hi:+.3f}] ({x.conclusion})")
    lines += ["", "## Manuscript mapping (all figures share src/figstyle.py)",
              "| Manuscript item | File |", "|---|---|",
              "| Fig 1 middle/right | figures/Fig1_zone_importance_circular.png |",
              "| Fig 3 (replaces 3a + 3b) | figures/Fig3_performance_ablation.png |",
              "| Fig 4a / 4b / 4c / 4d | figures/Fig4a_group_importance / Fig4b_ring_averaged / Fig4c_zone_maps_linear / Fig4d_zone_maps_log |",
              "| Table 3 | tables/T1_performance |",
              "| Appendix: ROC (A2) | " + (", ".join(f"figures/FigA2_roc_{t}.png" for t in roc_done) if roc_done else "run on the HPC") + " |",
              "| Appendix: focus-group final panel (A3) | figures/FigA3_zone_importance_square.png |",
              "| Appendix: local explanations (old Fig 6) | figures/FigA_local_explanations.png |",
              "| Appendix: leave-one-group-out (old Fig 5) | figures/FigA_leave_one_group_out.png + tables/T3 |",
              "| Appendix: calibration + decision curves | figures/FigA_calibration_dca_*.png + tables/TA7_calibration.csv |",
              "| Appendix: age comparisons | tables/TA6_age_comparisons.csv |",
              "| Appendix: hybrid / any-AMD SHAP, square maps, univariate heatmap | figures/FigA_SHAP_*, FigA_univariate_stage_ring_heatmap.png |",
              "| Tables A2 / A3 / A5 | tables/T2_feature_sets / T4_models / T3_ofa_value_paired |"]
    (out / "RESULTS_PACK.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}  (tables: {len(list((out / 'tables').glob('*.csv')))}, figures: {len(list((out / 'figures').glob('*.png')))})")


if __name__ == "__main__":
    main()
