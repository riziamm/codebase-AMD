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


# ------------------------------------------------------------------ figures
def _plt():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_model_comparison(R, path):
    plt = _plt()
    m = R[(R.groups == "all") & (R.rings == "all")]
    models = ["lr", "rf", "hybrid_zone", "hybrid_orig"]; cols = ["#7f8c8d", "#2e86c1", "#d35400", "#8e44ad"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, t in zip(axes, TASKS):
        for k, (mod, c) in enumerate(zip(models, cols)):
            for j, gb in enumerate(["eye", "subject"]):
                x = m[(m.task == t) & (m.model == mod) & (m.group_by == gb)]
                if len(x):
                    x = x.iloc[0]; xpos = k + (j - 0.5) * 0.35
                    ax.errorbar(xpos, x.AUC, yerr=[[x.AUC - x.lo], [x.hi - x.AUC]], fmt="o" if gb == "eye" else "s",
                                color=c, mfc=c if gb == "eye" else "white", capsize=3)
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        ax.set_xticks(range(4)); ax.set_xticklabels(["LR", "RF", "Zone\nhybrid", "Orig.\nhybrid"], fontsize=8)
        ax.set_title(TASK_LABEL[t], fontsize=9); ax.set_ylim(0.2, 1.0)
    axes[0].set_ylabel("AUC (95% CI)")
    h = [plt.Line2D([], [], marker="o", ls="", color="k", label="eye-grouped CV"),
         plt.Line2D([], [], marker="s", ls="", color="k", mfc="white", label="subject-grouped CV")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Repeated nested cross-validation (10 x 5-fold); dashed line = chance", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=250, bbox_inches="tight"); plt.close(fig)


def fig_feature_sets(R, path):
    plt = _plt()
    e = R[R.group_by == "eye"]
    sets = [("all", "all", "MPOD + OFA"), (MPOD, "all", "MPOD only"), ("Del,Amp", "all", "OFA only"),
            ("all", "centre", "Centre ring"), ("all", "middle", "Middle ring"), ("all", "outer", "Outer ring")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharex=True)
    for ax, t in zip(axes, TASKS):
        ylab = []
        for i, (g, rg, lab) in enumerate(sets):
            for j, (mod, c) in enumerate([("lr", "#7f8c8d"), ("rf", "#2e86c1")]):
                x = e[(e.task == t) & (e.groups == g) & (e.rings == rg) & (e.model == mod)]
                if len(x):
                    x = x.iloc[0]; y = -i + (j - 0.5) * 0.3
                    ax.errorbar(x.AUC, y, xerr=[[x.AUC - x.lo], [x.hi - x.AUC]], fmt="o", color=c, capsize=2, ms=4)
            ylab.append(lab)
        ax.axvline(0.5, color="k", ls="--", lw=0.8); ax.set_yticks(-np.arange(len(sets))); ax.set_yticklabels(ylab, fontsize=8)
        ax.set_title(TASK_LABEL[t], fontsize=9); ax.set_xlim(0.2, 1.0); ax.set_xlabel("AUC (95% CI)")
    h = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in [("#7f8c8d", "Logistic regression"), ("#2e86c1", "Random forest")]]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Feature-set and ring ablation (eye-grouped repeated nested CV; ring-only models: LR)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=250, bbox_inches="tight"); plt.close(fig)


def fig_logo(C, path):
    plt = _plt()
    B = C[C.comparison.str.startswith("B:")].copy()
    if B.empty:
        return
    B["group"] = B.comparison.str.extract(r"importance of (\w+)")[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    for ax, t in zip(axes, TASKS):
        for j, (mod, c) in enumerate([("lr", "#7f8c8d"), ("rf", "#2e86c1")]):
            x = B[(B.task == t) & (B.model == mod)].set_index("group").reindex(BASE)
            y = -np.arange(len(BASE)) + (j - 0.5) * 0.3
            ax.errorbar(x.dAUC, y, xerr=[x.dAUC - x.lo, x.hi - x.dAUC], fmt="o", color=c, capsize=2, ms=4, label=mod.upper())
        ax.axvline(0, color="k", ls="--", lw=0.8); ax.set_yticks(-np.arange(len(BASE))); ax.set_yticklabels(BASE, fontsize=8)
        ax.set_title(TASK_LABEL[t], fontsize=9); ax.set_xlabel("dAUC: all - (all minus group)")
    fig.legend(*axes[-1].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Leave-one-feature-group-out (paired, eye-clustered bootstrap 95% CI); > 0 = group adds information", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=250, bbox_inches="tight"); plt.close(fig)


def fig_roc(rcv, task, path):
    from sklearn.metrics import roc_curve, roc_auc_score
    plt = _plt()
    folder = Path(rcv) / f"eye_{task}_all_allrings"
    grid = np.linspace(0, 1, 201); any_ = False
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    for mod, c in [("lr", "#7f8c8d"), ("rf", "#2e86c1"), ("hybrid_zone", "#d35400"), ("hybrid_orig", "#8e44ad")]:
        f = folder / mod / "oof.csv"
        if not f.exists():
            continue
        o = pd.read_csv(f); tprs, aucs = [], []; nrep = o.cv_rep.nunique()
        for r, d in o.groupby("cv_rep"):
            fpr, tpr, _ = roc_curve(d.y_true, d.y_prob); tprs.append(np.interp(grid, fpr, tpr)); aucs.append(roc_auc_score(d.y_true, d.y_prob))
        T = np.array(tprs); T[:, 0] = 0
        ax.plot(grid, T.mean(0), color=c, label=f"{MODEL_LABEL[mod]} ({np.mean(aucs):.2f})")
        ax.fill_between(grid, np.percentile(T, 2.5, 0), np.percentile(T, 97.5, 0), color=c, alpha=0.12)
        any_ = True
    if not any_:
        plt.close(fig); return False
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"{TASK_LABEL[task]}\nout-of-fold ROC, mean of {nrep} repeats (band = 95% repeat range)", fontsize=8); ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=250, bbox_inches="tight"); plt.close(fig)
    return True


def fig_zone_overall(imp_csv, title, path, square=False, log=False):
    """All-feature zone importance: MPOD (7 stats) | OFA (Del + Amp) | combined, side by side, shared colour scale."""
    plt = _plt()
    from matplotlib.patches import Polygon
    from matplotlib.colors import LogNorm, Normalize
    imp = pd.read_csv(imp_csv)
    panels = {"Structure (MPOD, 7 statistics)": imp[imp.group.isin(BASE[:7])],
              "Function (OFA delay + sensitivity)": imp[imp.group.isin(["Del", "Amp"])], "All features": imp}
    vals = {k: v.groupby("region").mean_abs_shap.sum() for k, v in panels.items()}
    allv = pd.concat(vals.values()); pos = allv[allv > 0]
    norm = (LogNorm(vmin=max(pos.min(), pos.max() * 1e-2), vmax=pos.max()) if log and len(pos) else Normalize(0, allv.max()))
    cmap = plt.get_cmap("viridis"); geo = zone_geometry(square)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, (name, v) in zip(axes, vals.items()):
        for z, poly in geo.items():
            val = v.get(z, 0.0)
            ax.add_patch(Polygon(poly, closed=True, facecolor=cmap(norm(max(val, norm.vmin))), edgecolor="white", lw=0.8))
            cpos = poly.mean(0) if z > 4 else poly[1:].mean(0) * 0.6
            ax.text(*cpos, str(z), ha="center", va="center", fontsize=7, color="white")
        ax.set_xlim(-3.15, 3.15); ax.set_ylim(-3.15, 3.15); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(name, fontsize=9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), shrink=0.75, label="sum of mean |SHAP| per zone" + (" (log)" if log else ""))
    fig.suptitle(title + ("   [square layout, same sectors]" if square else ""), fontsize=10)
    fig.savefig(path, dpi=250, bbox_inches="tight"); plt.close(fig)


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

    fig_model_comparison(R, out / "figures" / "F_model_comparison.png")
    fig_feature_sets(R, out / "figures" / "F_feature_sets.png")
    if C is not None:
        fig_logo(C, out / "figures" / "F_leave_one_group_out.png")
    roc_done = [t for t in TASKS if fig_roc(rcv, t, out / "figures" / f"F_roc_{t}.png")]

    exp = root / "explain"; copied = []
    for t in ["advanced", "any"]:
        for mdl in ["rf", "hybrid_zone"]:
            d = exp / f"{t}_{mdl}"
            if not (d / "importance_features.csv").exists():
                continue
            if mdl == "rf":
                for sq in (False, True):
                    for lg in (False, True):
                        fig_zone_overall(d / "importance_features.csv", f"Random forest | {TASK_LABEL[t]} | zone importance (out-of-fold SHAP)",
                                         out / "figures" / f"F_zone_overall_{t}_rf_{'square' if sq else 'circular'}_{'log' if lg else 'linear'}.png", sq, lg)
            for f in (d / "figures").glob("*.png"):
                dst = out / "figures" / f"SHAP_{t}_{mdl}_{f.name}"; shutil.copy2(f, dst); copied.append(dst.name)
            shutil.copy2(d / "stability.json", out / "tables" / f"SHAP_stability_{t}_{mdl}.json")
    diag = root / "diagnostics" / "figures" / "A_stage_direction_heatmap.png"
    if diag.exists():
        shutil.copy2(diag, out / "figures" / "S_univariate_stage_ring_heatmap.png")

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
    lines += ["", "## Manuscript mapping",
              "| Manuscript item | Replace with |", "|---|---|",
              "| Abstract / Results numbers | Key numbers above, T1 |",
              "| Table 3 (holdout performance) | tables/T1_performance |",
              "| Table A2 (feature indexing) + Fig 3b | tables/T2_feature_sets + figures/F_feature_sets.png |",
              "| Table A3 (DL models) | tables/T4_models + T1 rows for the hybrids |",
              "| Table A4 (multiclass) | stage-specific rows of T1 (early / advanced) |",
              "| Fig 1 middle/right, Fig A3 'final' SHAP panel | figures/F_zone_overall_advanced_rf_circular_linear/log.png |",
              "| Fig 3a (normalisation x sampling) | removed (experiment on the earlier data export) |",
              "| Fig 4a / 4b / 4c / 4d | figures/SHAP_advanced_rf_group_importance / ring_averaged / zone_maps_circular_linear / _log |",
              "| Fig 5 (iterative exclusion) + Figs A4-A6 | figures/F_leave_one_group_out.png + tables/T3 (supplement) |",
              "| Fig 6 (local explanations) | figures/SHAP_advanced_rf_local_examples.png |",
              "| Fig A1 (architecture) | redraw for the zone-token hybrid (T4 row) |",
              "| Fig A2 (ROC) | " + (", ".join(f"figures/F_roc_{t}.png" for t in roc_done) if roc_done else "run on the HPC (needs oof.csv)") + " |",
              "| Supplementary (optional) | figures/S_univariate_stage_ring_heatmap.png; hybrid SHAP: figures/SHAP_*_hybrid_zone_* |"]
    (out / "RESULTS_PACK.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}  (tables: {len(list((out / 'tables').glob('*.csv')))}, figures: {len(list((out / 'figures').glob('*.png')))})")


if __name__ == "__main__":
    main()
