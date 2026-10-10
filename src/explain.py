#!/usr/bin/env python3
"""
explain.py: SHAP explanations of the SAME out-of-fold models that produced the repeated-CV AUCs.

For every repeat r and outer fold f: fit the model on training eyes (nested tuning for lr/rf, fixed config for
hybrid_zone), explain the held-out eyes. Every eye is explained once per repeat, never by a model that saw it.
  rf          -> shap.TreeExplainer (exact)        lr -> shap.LinearExplainer
  hybrid_zone -> shap.KernelExplainer (k-means background of the training eyes)

Outputs (aggregates; no feature values written)
  importance_features.csv    180 features: mean|SHAP|, SD over repeats, direction (Spearman rho value vs SHAP)
  importance_group_ring.csv  9 feature groups x ring (sum of mean|SHAP| over zones) + group totals
  stability.json             repeat-to-repeat Spearman of the group ranking; top-group frequency
  local_examples.csv         top contributions for 2 eyes (most confident correct AMD / normal), SHAP only
  figures/                   group bar, group x ring heatmap, ring bars, zone maps (circular + square; linear + log),
                             local explanation bars
Usage
  python -m src.explain --model rf --task advanced --out reruns_eyegrouped/explain/advanced_rf
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .repeated_cv import (BASE, RING_OF, apply_task, load_data, outer_splits, fit_sklearn, train_dl, dl_predict, _torch)

warnings.filterwarnings("ignore")
RINGS = ["centre", "middle", "outer"]


# ------------------------------------------------------------------ SHAP per fold
def shap_fold(model_name, X, y, groups, tr, te, seed, a, device):
    import shap
    if model_name in ("rf", "lr"):
        pipe = fit_sklearn(model_name, X, y, groups, tr, seed, a.rf_trees)
        pre, clf = pipe[:-1], pipe[-1]
        Xtr, Xte = pre.transform(X[tr]), pre.transform(X[te])
        if model_name == "rf":
            sv = shap.TreeExplainer(clf).shap_values(Xte)
        else:
            sv = shap.LinearExplainer(clf, Xtr).shap_values(Xte)
        prob = pipe.predict_proba(X[te])[:, 1]
    else:
        model, imp, sc = train_dl(model_name, X[tr], y[tr], seed, device, a.dl_epochs, a.dl_lr, a.dl_wd, a.dl_bs)
        Ztr, Zte = sc.transform(imp.transform(X[tr])), sc.transform(imp.transform(X[te]))
        torch, _ = _torch()

        def f(Z):
            with torch.no_grad():
                t = torch.tensor(Z, dtype=torch.float32, device=device)
                return torch.sigmoid(model(t).view(-1)).cpu().numpy()
        bg = shap.kmeans(Ztr, min(10, len(Ztr)))
        sv = shap.KernelExplainer(f, bg).shap_values(Zte, nsamples=a.nsamples, silent=True)
        prob = dl_predict(model, imp, sc, X[te], device)
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
    if sv.ndim == 3:                      # (n, features, classes) in newer shap
        sv = sv[..., 1]
    return sv, prob


# ------------------------------------------------------------------ zone geometry
def _sector_poly(r0, r1, th0, th1, square, n=24):
    th = np.linspace(np.radians(th0), np.radians(th1), n)
    def pts(r, ang):
        if r == 0:
            return np.zeros((1, 2))
        if square:
            t = r / np.maximum(np.abs(np.cos(ang)), np.abs(np.sin(ang)))
            return np.c_[t * np.cos(ang), t * np.sin(ang)]
        return np.c_[r * np.cos(ang), r * np.sin(ang)]
    return np.vstack([pts(r1, th), pts(r0, th[::-1])])


def zone_geometry(square):
    """Zone -> polygon. Centre ring: zones 1-4 = quadrants from 0 deg anticlockwise; middle 5-12 and outer 13-20:
    45-deg sectors from 0 deg anticlockwise (paper Fig. 1 layout)."""
    g = {}
    for k, z in enumerate(range(1, 5)):
        g[z] = _sector_poly(0, 1, 90 * k, 90 * (k + 1), square)
    for k, z in enumerate(range(5, 13)):
        g[z] = _sector_poly(1, 2, 45 * k, 45 * (k + 1), square)
    for k, z in enumerate(range(13, 21)):
        g[z] = _sector_poly(2, 3, 45 * k, 45 * (k + 1), square)
    return g


def zone_maps(imp, path, title, square=False, log=False):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.colors import LogNorm, Normalize
    geo = zone_geometry(square)
    vals = imp.set_index("feature")["mean_abs_shap"]
    pos = vals[vals > 0]
    norm = (LogNorm(vmin=max(pos.min(), pos.max() * 1e-3), vmax=pos.max()) if log and len(pos)
            else Normalize(vmin=0, vmax=vals.max() if vals.max() > 0 else 1))
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(3, 3, figsize=(10, 10.6))
    for ax, b in zip(axes.ravel(), BASE):
        for z, poly in geo.items():
            v = vals.get(f"{b}_region{z}", np.nan)
            col = cmap(norm(max(v, norm.vmin))) if np.isfinite(v) else (0.9, 0.9, 0.9, 1)
            ax.add_patch(Polygon(poly, closed=True, facecolor=col, edgecolor="white", lw=0.8))
            c = poly.mean(0) if z > 4 else poly[1:].mean(0) * 0.6
            ax.text(*c, str(z), ha="center", va="center", fontsize=6.5, color="white")
        ax.set_xlim(-3.15, 3.15); ax.set_ylim(-3.15, 3.15); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(b, fontsize=10)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, label="mean |SHAP| (out-of-fold)" + (" - log scale" if log else ""))
    fig.suptitle(title + ("   [square layout, same sectors]" if square else ""), fontsize=11)
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def bar_and_heatmap(G, R, rep_group, out, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # group bar with SD over repeats
    order = G.sort_values("sum_mean_abs_shap", ascending=False)
    sd = rep_group.std(0).reindex(order.group)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(order.group, order.sum_mean_abs_shap, yerr=sd.values, capsize=3, color="#3b6ea5")
    ax.set_ylabel("Sum of mean |SHAP|"); ax.set_title(title + ": feature-group importance", fontsize=10)
    plt.xticks(rotation=45); fig.savefig(out / "group_importance.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    # group x ring heatmap
    M = R.pivot(index="group", columns="ring", values="sum_mean_abs_shap").reindex(index=BASE, columns=RINGS)
    fig, ax = plt.subplots(figsize=(4.6, 4.8))
    im = ax.imshow(M.values, cmap="viridis", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:.3f}", ha="center", va="center", fontsize=7, color="white")
    ax.set_xticks(range(3)); ax.set_xticklabels(RINGS); ax.set_yticks(range(len(BASE))); ax.set_yticklabels(BASE)
    fig.colorbar(im, ax=ax, shrink=0.8, label="sum of mean |SHAP|"); ax.set_title(title + ": group x ring", fontsize=10)
    fig.savefig(out / "group_ring_heatmap.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    # ring-averaged bars (mean per zone, so rings of different size are comparable)
    fig, ax = plt.subplots(figsize=(7, 3.6)); w = 0.09
    for k, b in enumerate(BASE):
        vals = [R[(R.group == b) & (R.ring == rg)].mean_per_zone.values[0] for rg in RINGS]
        ax.bar(np.arange(3) + (k - 4) * w, vals, w, label=b)
    ax.set_xticks(range(3)); ax.set_xticklabels(RINGS); ax.set_ylabel("mean |SHAP| per zone")
    ax.legend(fontsize=7, ncol=3); ax.set_title(title + ": ring-averaged importance", fontsize=10)
    fig.savefig(out / "ring_averaged.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def local_plot(L, out, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (case, d) in zip(axes, L.groupby("case", sort=False)):
        d = d.sort_values("abs_shap")
        ax.barh(d.feature, d.shap, color=["#c0392b" if v > 0 else "#2e86c1" for v in d.shap])
        ax.set_title(f"{case}\nmean OOF P(disease) = {d.prob.iloc[0]:.2f}", fontsize=9); ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("SHAP (red pushes towards disease)")
    fig.suptitle(title + ": local explanations (top 12 features)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "local_examples.png", dpi=220, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["rf", "lr", "hybrid_zone"])
    ap.add_argument("--task", default="advanced", choices=["any", "early", "advanced"])
    ap.add_argument("--data", nargs="+", default=["reruns_eyegrouped/split/train_mpod.csv", "reruns_eyegrouped/split/test_mpod.csv"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rf_trees", type=int, default=300)
    ap.add_argument("--nsamples", type=int, default=200, help="KernelExplainer samples (hybrid)")
    ap.add_argument("--dl_epochs", type=int, default=80)
    ap.add_argument("--dl_lr", type=float, default=3e-3)
    ap.add_argument("--dl_wd", type=float, default=1e-2)
    ap.add_argument("--dl_bs", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)

    df = apply_task(load_data(a.data), a.task)
    feats = [f"{b}_region{z}" for b in BASE for z in range(1, 21)]
    X = df[feats].to_numpy(float); y = df["yb"].to_numpy(); groups = df["eye_id"].to_numpy()
    splits = outer_splits(df, a.k, a.repeats, a.seed, "eye")        # identical to repeated_cv
    device = "cpu"
    if a.model == "hybrid_zone":
        torch, _ = _torch(); device = "cuda" if torch.cuda.is_available() else "cpu"
    SV = np.full((a.repeats, len(y), len(feats)), np.nan); P = np.full((a.repeats, len(y)), np.nan)
    for r in range(a.repeats):
        for f, (tr, te) in enumerate(splits[r]):
            sv, prob = shap_fold(a.model, X, y, groups, tr, te, a.seed * 1000 + r * 10 + f, a, device)
            SV[r, te] = sv; P[r, te] = prob
        print(f"   repeat {r + 1}/{a.repeats} explained", flush=True)
    from sklearn.metrics import roc_auc_score
    auc = float(np.mean([roc_auc_score(y, P[r]) for r in range(a.repeats)]))

    # feature importance
    absmean_rep = np.abs(SV).mean(1)                      # (R, features)
    imp = pd.DataFrame({"feature": feats, "group": [c.split("_region")[0] for c in feats],
                        "region": [int(c.split("_region")[1]) for c in feats]})
    imp["ring"] = imp.region.map(RING_OF)
    imp["mean_abs_shap"] = absmean_rep.mean(0); imp["sd_over_repeats"] = absmean_rep.std(0)
    flatX = np.repeat(X[None], a.repeats, 0).reshape(-1, len(feats)); flatS = SV.reshape(-1, len(feats))
    imp["direction_rho"] = [spearmanr(flatX[:, j], flatS[:, j], nan_policy="omit")[0] if np.nanstd(flatX[:, j]) > 0 else np.nan
                            for j in range(len(feats))]
    imp["direction"] = np.where(imp.direction_rho > 0, "higher value -> towards disease", "lower value -> towards disease")
    imp.sort_values("mean_abs_shap", ascending=False).to_csv(out / "importance_features.csv", index=False)

    R_ = imp.groupby(["group", "ring"]).agg(sum_mean_abs_shap=("mean_abs_shap", "sum"),
                                            mean_per_zone=("mean_abs_shap", "mean")).reset_index()
    G = imp.groupby("group").mean_abs_shap.sum().rename("sum_mean_abs_shap").reset_index()
    pd.concat([R_, G.assign(ring="all", mean_per_zone=np.nan)]).to_csv(out / "importance_group_ring.csv", index=False)

    rep_group = pd.DataFrame(absmean_rep, columns=feats).T.groupby(lambda c: c.split("_region")[0]).sum().T
    ranks = rep_group.rank(axis=1, ascending=False)
    rhos = [spearmanr(ranks.iloc[i], ranks.iloc[j])[0] for i in range(len(ranks)) for j in range(i + 1, len(ranks))]
    top = rep_group.idxmax(axis=1).value_counts().to_dict()
    ring_tot = R_.groupby("ring").mean_per_zone.mean().reindex(RINGS).to_dict()
    json.dump(dict(model=a.model, task=a.task, repeats=a.repeats, oof_auc=auc,
                   group_rank_stability_mean_spearman=float(np.nanmean(rhos)) if rhos else None,
                   top_group_frequency=top, mean_abs_shap_per_zone_by_ring=ring_tot), open(out / "stability.json", "w"), indent=1)

    # local examples (aggregated over repeats; SHAP values only)
    pm = P.mean(0); sm = SV.mean(0)
    cases = {}
    if (y == 1).any():
        cases["Most confident correct DISEASE eye"] = np.where(y == 1)[0][np.argmax(pm[y == 1])]
    if (y == 0).any():
        cases["Most confident correct NORMAL eye"] = np.where(y == 0)[0][np.argmin(pm[y == 0])]
    L = []
    for name, i in cases.items():
        top12 = np.argsort(-np.abs(sm[i]))[:12]
        for j in top12:
            L.append(dict(case=name, feature=feats[j].replace("_region", "_Z"), shap=sm[i, j], abs_shap=abs(sm[i, j]), prob=pm[i]))
    L = pd.DataFrame(L); L.to_csv(out / "local_examples.csv", index=False)

    title = f"{a.model} | task={a.task} | OOF AUC={auc:.2f}"
    bar_and_heatmap(G, R_, rep_group, out / "figures", title)
    for sq in (False, True):
        for lg in (False, True):
            zone_maps(imp, out / "figures" / f"zone_maps_{'square' if sq else 'circular'}_{'log' if lg else 'linear'}.png",
                      title, square=sq, log=lg)
    if len(L):
        local_plot(L, out / "figures", title)
    print(f"DONE explain {a.model} task={a.task}: OOF AUC {auc:.3f}; top groups {list(G.sort_values('sum_mean_abs_shap', ascending=False).group[:3])}; "
          f"rank stability {np.nanmean(rhos) if rhos else float('nan'):.2f}")


if __name__ == "__main__":
    main()
