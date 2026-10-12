"""
figstyle.py: ONE visual style for every Paper 1 figure, matching the original manuscript figures
(matplotlib defaults, DejaVu Sans; viridis zone maps on the original geometry, radii 0.3/0.6/1.0 with thin black
edges; tab-blue bar charts with value labels; ROC curves with grey dashed diagonal and dashed grid; SHAP red/blue
for local explanations). All figures are saved tight (pad 0.02 in) to minimise whitespace.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LogNorm, Normalize

BASE = ["mean", "median", "std", "iqr", "idr", "skew", "kurt", "Del", "Amp"]
BAR = "#1f77b4"            # matplotlib C0, as in the original Fig 4a
BAR_LIGHT = "#aec7e8"
MODEL_COLOR = {"rf": "blue", "lr": "green", "hybrid_zone": "red", "hybrid_orig": "orange"}   # original ROC palette
MODEL_SHORT = {"lr": "LR", "rf": "RF", "hybrid_zone": "Zone\nhybrid", "hybrid_orig": "Original\nhybrid"}
SHAP_POS, SHAP_NEG = "#ff0051", "#008bfb"   # shap package colours (original waterfall plots)
R_IN, R_MID, R_OUT = 0.3, 0.6, 1.0          # original circular-map radii


def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "legend.title_fontsize": 7.5,
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.8})


def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ------------------------------------------------------------------ zone maps
def _zones(square):
    """zone -> (polygon, label_xy). Same numbering as the original figures:
    centre 1-4 = quadrants anticlockwise from upper-right; middle 5-12 and outer 13-20 = 45-degree sectors
    anticlockwise from 3 o'clock."""
    def ring_pts(r, th):
        if r == 0:
            return np.zeros((1, 2))
        if square:
            t = r / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th)))
            return np.c_[t * np.cos(th), t * np.sin(th)]
        return np.c_[r * np.cos(th), r * np.sin(th)]
    out = {}
    specs = [(z, 0.0, R_IN, 90 * k, 90 * (k + 1)) for k, z in enumerate(range(1, 5))]
    specs += [(z, R_IN, R_MID, 45 * k, 45 * (k + 1)) for k, z in enumerate(range(5, 13))]
    specs += [(z, R_MID, R_OUT, 45 * k, 45 * (k + 1)) for k, z in enumerate(range(13, 21))]
    for z, r0, r1, a0, a1 in specs:
        th = np.radians(np.linspace(a0, a1, 32))
        poly = np.vstack([ring_pts(r1, th), ring_pts(r0, th[::-1])])
        am = np.radians((a0 + a1) / 2); rl = r1 / 1.6 if r0 == 0 else (r0 + r1) / 2
        if square and r0 > 0:
            rl = rl / max(abs(np.cos(am)), abs(np.sin(am))) ** 0.5
        out[z] = (poly, (rl * np.cos(am), rl * np.sin(am)))
    return out


def zone_panel(ax, vals, norm, cmap, square=False, label_size=5.5, title=None):
    for z, (poly, (x, y)) in _zones(square).items():
        v = vals.get(z, np.nan)
        col = cmap(norm(max(v, norm.vmin))) if np.isfinite(v) else (0.85, 0.85, 0.85, 1)
        ax.add_patch(patches.Polygon(poly, closed=True, facecolor=col, edgecolor="black", linewidth=0.3))
        ax.text(x, y, str(z), ha="center", va="center", fontsize=label_size,
                color="white" if (np.isfinite(v) and norm(max(v, norm.vmin)) < 0.3) else "black")
    pad = 0.03
    ax.set_xlim(-R_OUT - pad, R_OUT + pad); ax.set_ylim(-R_OUT - pad, R_OUT + pad)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, pad=2)


def _norm(values, log):
    v = np.asarray([x for x in values if np.isfinite(x)])
    if log:
        pos = v[v > 0]
        return LogNorm(vmin=max(pos.min(), pos.max() * 1e-3), vmax=pos.max())
    return Normalize(vmin=v.min(), vmax=v.max())


def zone_grid(imp, path, log=False, square=False, cbar_label="Mean Abs SHAP"):
    """3x3 maps (one per feature group), shared colour bar; compact (original Fig 4c/4d style)."""
    apply()
    vals = {b: imp[imp.group == b].set_index("region").mean_abs_shap.to_dict() for b in BASE}
    norm = _norm([x for d in vals.values() for x in d.values()], log); cmap = plt.cm.viridis
    fig, axes = plt.subplots(3, 3, figsize=(5.6, 5.4))
    fig.subplots_adjust(left=0.0, right=0.86, top=0.965, bottom=0.0, wspace=0.03, hspace=0.12)
    for ax, b in zip(axes.ravel(), BASE):
        zone_panel(ax, vals[b], norm, cmap, square, title=b)
    cax = fig.add_axes([0.885, 0.12, 0.022, 0.74])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 label=cbar_label + (" (Log Scale)" if log else " (Linear Scale)"))
    save(fig, path)


def zone_pair(vals, path, square=False, cbar_label="Mean Abs SHAP Value"):
    """Fig 1 middle/right: one map, linear and log colour scales side by side, each with its colour bar."""
    apply()
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.45))
    fig.subplots_adjust(left=0.0, right=0.93, top=1.0, bottom=0.0, wspace=0.42)
    for ax, log in zip(axes, (False, True)):
        norm = _norm(list(vals.values()), log)
        zone_panel(ax, vals, norm, plt.cm.viridis, square, label_size=7)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis), ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(cbar_label + (" (Log Scale)" if log else " (Linear Scale)"), fontsize=7)
    save(fig, path)


# ------------------------------------------------------------------ bar charts
def group_bar(series, path, ylabel="Sum of Mean(|SHAP Value|)"):
    """Original Fig 4a style: sorted bars, tab-blue, value labels."""
    apply()
    s = series.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    bars = ax.bar(s.index, s.values, color=BAR)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=6)
    ax.set_ylabel(ylabel); ax.set_ylim(0, s.max() * 1.15)
    ax.tick_params(axis="x", rotation=90)
    save(fig, path)


def ring_bars(ring_df, path):
    """Original Fig 4b style: per ring (Outer, Middle, Centre) grouped bars per feature, viridis colours."""
    apply()
    rings = [("outer", "Outer"), ("middle", "Middle"), ("centre", "Centre")]
    cols = plt.cm.viridis(np.linspace(0, 1, len(BASE)))
    fig, ax = plt.subplots(figsize=(4.4, 2.6)); w = 0.09
    for k, b in enumerate(BASE):
        v = [ring_df[(ring_df.group == b) & (ring_df.ring == r)].mean_per_zone.values[0] for r, _ in rings]
        ax.bar(np.arange(3) + (k - 4) * w, v, w, color=cols[k], edgecolor="black", linewidth=0.3, label=b)
    ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in rings]); ax.set_xlabel("Rings")
    ax.set_ylabel("Ring Averaged SHAP Values")
    ax.legend(title="Features", fontsize=6, title_fontsize=6.5, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.02), frameon=False)
    save(fig, path)


def auc_bars(ax, labels, groups, vals, los, his, colors, ylabel="AUC (95% CI)", vlabel=True):
    """Grouped AUC bars with CI whiskers and value labels; dashed chance line."""
    n = len(groups); w = 0.8 / n
    for j, (g, c) in enumerate(zip(groups, colors)):
        x = np.arange(len(labels)) + (j - (n - 1) / 2) * w
        v, lo, hi = np.array(vals[j]), np.array(los[j]), np.array(his[j])
        ax.bar(x, v, w, color=c, label=g, edgecolor="black", linewidth=0.3)
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor="black", capsize=1.5, lw=0.6)
        if vlabel:
            for xi, vi, hi_ in zip(x, v, hi):
                if np.isfinite(vi):
                    ax.text(xi, hi_ + 0.01, f"{vi:.2f}", ha="center", va="bottom", fontsize=5.5)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylim(0.25, 1.08); ax.set_ylabel(ylabel)


# ------------------------------------------------------------------ ROC, local explanations, calibration
def roc(curves, path, title=None):
    """curves: list of (label, fpr_grid, mean_tpr, lo, hi, auc, color). Original Fig A2 style."""
    apply()
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    for lab, g, t, lo, hi, auc, c in curves:
        ax.plot(g, t, color=c, lw=1.6, label=f"{lab} (AUC = {auc:.3f})")
        ax.fill_between(g, lo, hi, color=c, alpha=0.10, lw=0)
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.grid(True, ls="--", alpha=0.7)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    if title:
        ax.set_title(title, fontsize=8)
    ax.legend(loc="lower right", fontsize=6)
    save(fig, path)


def local_bars(L, path):
    """Local explanations in the shap waterfall colours (red = towards disease)."""
    apply()
    cases = list(dict.fromkeys(L.case))
    fig, axes = plt.subplots(1, len(cases), figsize=(3.3 * len(cases), 2.7))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cases):
        d = L[L.case == c].sort_values("abs_shap").tail(10)
        ax.barh(d.feature, d.shap, color=[SHAP_POS if v > 0 else SHAP_NEG for v in d.shap], height=0.7)
        for y, v in enumerate(d.shap):
            ax.text(v, y, f" {v:+.3f} " if v > 0 else f" {v:+.3f} ", va="center", ha="left" if v > 0 else "right", fontsize=5.5)
        ax.axvline(0, color="black", lw=0.6); ax.set_xlabel("SHAP value")
        lim = d.shap.abs().max() * 1.45; ax.set_xlim(-lim, lim)
        ax.set_title(f"{c.replace('Most confident correct ', '').capitalize()} eye, P(AMD) = {d.prob.iloc[0]:.2f}", fontsize=8)
    fig.tight_layout(pad=0.3)
    save(fig, path)
