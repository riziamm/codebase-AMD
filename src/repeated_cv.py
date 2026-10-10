#!/usr/bin/env python3
"""
repeated_cv.py: repeated NESTED eye-grouped cross-validation + permutation test.

WHY  A 9-eye holdout has AUC SD ~0.20 under no signal; it cannot answer "is there signal?".
     This uses ALL 58 eyes: every eye is scored out-of-fold once per repeat, R repeats.

PROTOCOL (nothing is tuned on outer-test data)
  outer  : repeated stratified K-fold; unit = eye (primary, matches the paper) or subject (conservative)
  inner  : sklearn models tune on training eyes only (3-fold eye-grouped, scoring=roc_auc)
           DL models are NOT tuned (fixed pre-specified config, fixed epochs)
  scaler / imputer fitted on training rows only, inside every fold
  metric : AUC (primary), pooled out-of-fold per repeat, averaged over repeats
  CI     : eye-clustered bootstrap (both repeat sessions of an eye move together)
  test   : label-permutation null; the WHOLE nested pipeline is re-run on permuted labels.
           block scheme (default) permutes each subject's (eye1, eye2) label pair between subjects,
           so fellow-eye label structure is preserved. One-sided p_above and p_below are reported.

MODELS   lr | rf | hybrid_zone (zone-token transformer + zone CNN, ~4k params) | hybrid_orig (paper's parallel CNN-Transformer)

USAGE
  python -m src.repeated_cv run --model lr --data reruns_eyegrouped/split/train_mpod.csv reruns_eyegrouped/split/test_mpod.csv \
         --out reruns_eyegrouped/repeated_cv/eye --group_by eye --k 5 --repeats 10 --n_perm 200 --perm_repeats 2 --n_jobs 8
  python -m src.repeated_cv aggregate --out reruns_eyegrouped/repeated_cv/eye
Resumable: finished models are skipped; permutations resume from null.csv.
Outputs per model: oof.csv (row-level, keep on HPC), null.csv, summary.json
Outputs aggregate: summary.csv, paired.csv, REPORT.md  (aggregates only, shareable)
"""
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from .eye_split import EyeGroupedKFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
BASE = ["mean", "median", "std", "iqr", "idr", "skew", "kurt", "Del", "Amp"]
SKLEARN_MODELS = ("lr", "rf")
DL_MODELS = ("hybrid_zone", "hybrid_orig")
RING = [0] * 4 + [1] * 8 + [2] * 8  # zones 1-4 centre, 5-12 middle, 13-20 outer


# ------------------------------------------------------------------ data
def load_data(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for c in ("Subject", "Eye", "Y"):
        if c not in df.columns:
            sys.exit(f"missing column {c}")
    if "Repeat" not in df.columns:
        df["Repeat"] = 0
    df = df.sort_values(["Subject", "Eye", "Repeat"]).reset_index(drop=True)  # canonical row order
    expected = [f"{b}_region{z}" for b in BASE for z in range(1, 21)]
    if list(df.columns[:180]) != expected:
        sys.exit("first 180 columns are not [mean..Amp] x region1..20 in the expected order")
    df = df.copy()
    df["eye_id"] = df["Subject"].astype(str) + "_" + df["Eye"].astype(str)
    if df.groupby("eye_id")["Y"].nunique().max() > 1:
        sys.exit("Y not constant within an eye")
    df["yb"] = (df["Y"] >= 2).astype(int)
    return df


TASKS = {"any": ([1, 2, 3, 4], [2, 3, 4]), "early": ([1, 2], [2]), "advanced": ([1, 3, 4], [3, 4])}
RING_OF = {z: ("centre" if z <= 4 else "middle" if z <= 12 else "outer") for z in range(1, 21)}


def apply_task(df, task):
    """Keep the AREDS grades of the task and set the binary label yb (1 = disease)."""
    keep, pos = TASKS[task]
    df = df[df["Y"].isin(keep)].reset_index(drop=True).copy()
    df["yb"] = df["Y"].isin(pos).astype(int)
    return df


MPOD_GROUPS = BASE[:7]
FUNC_GROUPS = ["Del", "Amp"]


def _parse_groups(groups):
    """'all' | comma list of BASE groups, plus optional 'coupling' and '-<group>' (leave-one-out)."""
    toks = groups.split(",") if groups != "all" else ["all"]
    G, coupling, drop = [], False, []
    for t in toks:
        if t == "all":
            G += BASE
        elif t == "mpod":
            G += MPOD_GROUPS
        elif t == "coupling":
            coupling = True
        elif t.startswith("-"):
            drop.append(t[1:])
        else:
            G.append(t)
    G = [g for g in dict.fromkeys(G) if g not in drop]
    bad = [g for g in G + drop if g not in BASE]
    if bad:
        sys.exit(f"unknown groups: {bad}")
    return G, coupling


def feature_columns(groups="all", rings="all"):
    G, _ = _parse_groups(groups)
    Rg = ("centre", "middle", "outer") if rings == "all" else tuple(rings.split(","))
    bad = [r for r in Rg if r not in ("centre", "middle", "outer")]
    if bad:
        sys.exit(f"unknown rings: {bad}")
    return [f"{b}_region{z}" for b in G for z in range(1, 21) if RING_OF[z] in Rg]


def coupling_features(df, rings="all"):
    """Structure-function coupling per row (one eye, one session): Spearman rho across the zones between
    each MPOD statistic and each OFA measure (Del, Amp). Computed within the row only -> no leakage."""
    from scipy.stats import rankdata
    Rg = ("centre", "middle", "outer") if rings == "all" else tuple(rings.split(","))
    zones = [z for z in range(1, 21) if RING_OF[z] in Rg]
    out = {}
    for s in MPOD_GROUPS:
        S = df[[f"{s}_region{z}" for z in zones]].to_numpy(float)
        for f in FUNC_GROUPS:
            F = df[[f"{f}_region{z}" for z in zones]].to_numpy(float)
            rho = np.full(len(df), np.nan)
            for i in range(len(df)):
                m = np.isfinite(S[i]) & np.isfinite(F[i])
                if m.sum() >= 4:
                    a, b = rankdata(S[i][m]), rankdata(F[i][m])
                    if a.std() > 0 and b.std() > 0:
                        rho[i] = np.corrcoef(a, b)[0, 1]
            out[f"coupling_{s}_{f}"] = rho
    return pd.DataFrame(out, index=df.index)


def build_X(df, groups="all", rings="all"):
    cols = feature_columns(groups, rings)
    _, coupling = _parse_groups(groups)
    parts = [df[cols]] if cols else []
    if coupling:
        parts.append(coupling_features(df, rings))
    if not parts:
        sys.exit("empty feature set")
    Xdf = pd.concat(parts, axis=1)
    return Xdf.to_numpy(float), list(Xdf.columns)


def outer_splits(df, K, R, seed, group_by):
    if group_by == "eye":
        units = df.groupby("eye_id")["Y"].first(); col = "eye_id"
    else:  # subject-grouped, stratified on worst-eye AREDS collapsed to {1, 2, 3+4}
        units = df.groupby("Subject")["Y"].max().clip(upper=3); col = "Subject"
    out = []
    for r in range(R):
        skf = StratifiedKFold(K, shuffle=True, random_state=seed + r)
        folds = []
        for _, te in skf.split(units.index, units.values):
            m = df[col].isin(set(units.index[te])).values
            folds.append((np.where(~m)[0], np.where(m)[0]))
        out.append(folds)
    return out


def permute_labels(df, rng, scheme):
    """Return a row-level binary label vector with the feature-label association destroyed."""
    eye_y = df.groupby("eye_id")["yb"].first()
    if scheme == "eye":
        new = pd.Series(rng.permutation(eye_y.values), index=eye_y.index)
    else:  # block: shuffle each subject's eye-label tuple between subjects with the same number of eyes
        subj = df.groupby("eye_id")["Subject"].first()
        eyeno = df.groupby("eye_id")["Eye"].first()
        eyes_of = {s: sorted(subj.index[subj == s], key=lambda e: eyeno[e]) for s in subj.unique()}
        new = {}
        for n_eyes in sorted({len(v) for v in eyes_of.values()}):
            S = [s for s, v in eyes_of.items() if len(v) == n_eyes]
            perm = rng.permutation(len(S))
            for i, s in enumerate(S):
                src = [eye_y[e] for e in eyes_of[S[perm[i]]]]
                for e, lab in zip(eyes_of[s], src):
                    new[e] = lab
        new = pd.Series(new)
    return df["eye_id"].map(new).values.astype(int)


# ------------------------------------------------------------------ sklearn models
def make_sklearn(model, seed, rf_trees):
    steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    if model == "lr":
        clf = LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced", max_iter=2000)
        grid = {"clf__C": [1e-3, 1e-2, 1e-1, 1.0]}
    else:
        clf = RandomForestClassifier(n_estimators=rf_trees, class_weight="balanced_subsample",
                                     random_state=seed, n_jobs=1)
        grid = {"clf__max_depth": [3, None], "clf__min_samples_leaf": [1, 3]}
    return Pipeline(steps + [("clf", clf)]), grid


def fit_sklearn(model, X, y, groups, tr, seed, rf_trees):
    """Nested tuning on training rows only; returns the refitted best pipeline."""
    pipe, grid = make_sklearn(model, seed, rf_trees)
    cv = EyeGroupedKFold(groups[tr], n_splits=3, random_state=seed, stratify_on=y[tr])
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, refit=True, n_jobs=1, error_score=0.5)
    gs.fit(X[tr], y[tr])
    return gs.best_estimator_


def _skl_task(model, X, y, groups, tr, te, seed, rf_trees):
    pipe, grid = make_sklearn(model, seed, rf_trees)
    cv = EyeGroupedKFold(groups[tr], n_splits=3, random_state=seed, stratify_on=y[tr])
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, refit=True, n_jobs=1, error_score=0.5)
    gs.fit(X[tr], y[tr])
    return gs.predict_proba(X[te])[:, 1]


# ------------------------------------------------------------------ deep models
def _torch():
    import torch, torch.nn as nn
    return torch, nn


def build_dl(kind):
    torch, nn = _torch()
    if kind == "hybrid_orig":
        try:
            from dl_pipeline_gen import CNNTransformerModel
        except Exception as e:
            sys.exit(f"cannot import CNNTransformerModel from dl_pipeline_gen.py ({e}); run from the repo root")
        return CNNTransformerModel(feature_dim=180, num_classes=1, cnn_units=16, transformer_dim=16,
                                   transformer_heads=2, transformer_layers=1,
                                   architecture_mode="parallel", dropout_rate=0.3)

    class ZoneHybrid(nn.Module):
        """20 zone tokens x 9 features. Attention is over ZONES (real attention, 20 tokens);
        the CNN branch convolves along zones with the 9 features as channels."""
        def __init__(self, d=16, heads=2, layers=1, ch=16, drop=0.3):
            super().__init__()
            self.embed = nn.Linear(9, d)
            self.zone_pos = nn.Parameter(torch.randn(20, d) * 0.02)
            self.ring_emb = nn.Embedding(3, d)
            self.register_buffer("ring_idx", torch.tensor(RING))
            enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=drop,
                                             activation="gelu", batch_first=True)
            self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
            self.conv = nn.Sequential(nn.Conv1d(9, ch, 3, padding=1), nn.BatchNorm1d(ch),
                                      nn.ReLU(), nn.Dropout(drop))
            self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d + ch, 1))

        def forward(self, x):                      # x: (N, 180) feature-major
            z = x.view(-1, 9, 20)                  # (N, feature, zone)
            h = self.embed(z.transpose(1, 2)) + self.zone_pos + self.ring_emb(self.ring_idx)
            h = self.enc(h).mean(1)
            c = self.conv(z).mean(2)
            return self.head(torch.cat([h, c], 1)).squeeze(1)

    return ZoneHybrid()


def train_dl(kind, Xtr, ytr, seed, device, epochs, lr, wd, bs):
    """Train a DL model on training rows only; returns (model, imputer, scaler)."""
    torch, nn = _torch()
    torch.manual_seed(seed); np.random.seed(seed)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    A = torch.tensor(sc.transform(imp.transform(Xtr)), dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    model = build_dl(kind).to(device)
    pw = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(A)
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(A[b]).view(-1), yt[b])
            loss.backward(); opt.step()
    model.eval()
    return model, imp, sc


def dl_predict(model, imp, sc, X, device):
    torch, _ = _torch()
    with torch.no_grad():
        B = torch.tensor(sc.transform(imp.transform(X)), dtype=torch.float32, device=device)
        return torch.sigmoid(model(B).view(-1)).cpu().numpy()


def fit_predict_dl(kind, Xtr, ytr, Xte, seed, device, epochs, lr, wd, bs):
    model, imp, sc = train_dl(kind, Xtr, ytr, seed, device, epochs, lr, wd, bs)
    return dl_predict(model, imp, sc, Xte, device)


# ------------------------------------------------------------------ CV driver
def run_cv(model, X, y, groups, splits, reps, seed_base, args, device):
    """Return OOF scores (len(reps), n_rows) for the given repeat indices using labels y."""
    n = len(y); K = len(splits[0])
    oof = np.full((len(reps), n), np.nan)
    tasks = [(ri, r, f) for ri, r in enumerate(reps) for f in range(K)]
    if model in SKLEARN_MODELS:
        res = Parallel(n_jobs=args.n_jobs)(
            delayed(_skl_task)(model, X, y, groups, splits[r][f][0], splits[r][f][1],
                               seed_base + r * 10 + f, args.rf_trees) for _, r, f in tasks)
    else:
        res = [fit_predict_dl(model, X[splits[r][f][0]], y[splits[r][f][0]], X[splits[r][f][1]],
                              seed_base + r * 10 + f, device, args.dl_epochs, args.dl_lr, args.dl_wd, args.dl_bs)
               for _, r, f in tasks]
    for (ri, r, f), p in zip(tasks, res):
        oof[ri, splits[r][f][1]] = p
    assert not np.isnan(oof).any(), "some rows never scored out-of-fold"
    return oof


def auc_rep(y, oof):
    return np.array([roc_auc_score(y, o) for o in oof])


def eye_boot(df, B, seed):
    """Yield row-index arrays from eye-clustered bootstrap resamples (same for every model)."""
    rng = np.random.default_rng(seed)
    eyes = [np.where(df["eye_id"].values == e)[0] for e in df["eye_id"].unique()]
    for _ in range(B):
        pick = rng.integers(0, len(eyes), len(eyes))
        yield np.concatenate([eyes[i] for i in pick])


def boot_auc(df, y, oof, B, seed):
    out = []
    for rows in eye_boot(df, B, seed):
        yb = y[rows]
        if yb.min() == yb.max():
            continue
        out.append(np.mean([roc_auc_score(yb, o[rows]) for o in oof]))
    return np.array(out)


def threshold_metrics(y, oof):
    sens, spec, bal, ppr, f1a, f1w = [], [], [], [], [], []
    for o in oof:
        yh = (o >= 0.5).astype(int)
        sens.append(((y == 1) & (yh == 1)).sum() / max((y == 1).sum(), 1))
        spec.append(((y == 0) & (yh == 0)).sum() / max((y == 0).sum(), 1))
        bal.append(balanced_accuracy_score(y, yh)); ppr.append(yh.mean())
        f1a.append(f1_score(y, yh, zero_division=0)); f1w.append(f1_score(y, yh, average="weighted", zero_division=0))
    m = lambda v: float(np.mean(v))
    return dict(sensitivity=m(sens), specificity=m(spec), balanced_acc=m(bal), pred_pos_rate=m(ppr),
                f1_amd=m(f1a), f1_weighted=m(f1w))


# ------------------------------------------------------------------ commands
def cmd_run(a):
    out = Path(a.out) / a.model; out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists() and not a.force:
        print(f"[skip] {a.model}: summary.json exists (use --force)"); return
    df = apply_task(load_data(a.data), a.task)
    X, cols = build_X(df, a.groups, a.rings)
    if a.model in DL_MODELS and (a.groups != "all" or a.rings != "all"):
        sys.exit("hybrid models need all 180 features (--groups all --rings all)")
    y = df["yb"].to_numpy(); groups = df["eye_id"].to_numpy()
    device = "cpu"
    if a.model in DL_MODELS:
        torch, _ = _torch(); device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   task={a.task} features={len(cols)} (groups={a.groups}, rings={a.rings})")
    print(f"== {a.model} | group_by={a.group_by} K={a.k} R={a.repeats} perms={a.n_perm}x{a.perm_repeats} device={device}")
    print(f"   rows={len(df)} eyes={df.eye_id.nunique()} subjects={df.Subject.nunique()} AMD rows={int(y.sum())}")
    splits = outer_splits(df, a.k, a.repeats, a.seed, a.group_by)
    t0 = time.time()

    # observed
    if (out / "oof.csv").exists() and not a.force:
        o = pd.read_csv(out / "oof.csv")
        oof = np.vstack([o[o.cv_rep == r].sort_values("row_index").y_prob.values for r in range(a.repeats)])
    else:
        oof = run_cv(a.model, X, y, groups, splits, list(range(a.repeats)), a.seed * 1000, a, device)
        rng = np.random.default_rng(a.seed); codes = rng.permutation(df.eye_id.nunique())
        code = codes[pd.factorize(df.eye_id)[0]]
        rows = [pd.DataFrame(dict(cv_rep=r, row_index=np.arange(len(y)), eye_code=code, y_true=y, y_prob=oof[r]))
                for r in range(a.repeats)]
        pd.concat(rows).to_csv(out / "oof.csv", index=False)
    aucs = auc_rep(y, oof)
    boot = boot_auc(df, y, oof, a.n_boot, a.seed)
    print(f"   observed AUC {aucs.mean():.3f} (SD over repeats {aucs.std(ddof=1) if len(aucs) > 1 else 0:.3f}) "
          f"CI [{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]  ({time.time() - t0:.0f}s)")

    # permutations (resumable, chunked)
    nf = out / "null.csv"
    null = list(pd.read_csv(nf)["T"]) if nf.exists() and not a.force else []
    pr = list(range(min(a.perm_repeats, a.repeats)))
    t_obs = float(aucs[pr].mean())
    while len(null) < a.n_perm:
        ps = list(range(len(null), min(len(null) + a.chunk, a.n_perm)))
        ys = {p: permute_labels(df, np.random.default_rng(a.seed + 777 + p), a.perm_scheme) for p in ps}
        if a.model in SKLEARN_MODELS:
            tasks = [(p, r, f) for p in ps for r in pr for f in range(a.k)]
            res = Parallel(n_jobs=a.n_jobs)(delayed(_skl_task)(
                a.model, X, ys[p], groups, splits[r][f][0], splits[r][f][1],
                a.seed * 1000 + 100000 + p * 100 + r * 10 + f, a.rf_trees) for p, r, f in tasks)
            oo = {p: np.full((len(pr), len(y)), np.nan) for p in ps}
            for (p, r, f), s in zip(tasks, res):
                oo[p][r, splits[r][f][1]] = s
        else:
            oo = {p: run_cv(a.model, X, ys[p], groups, splits, pr, a.seed * 1000 + 100000 + p * 100, a, device) for p in ps}
        for p in ps:
            null.append(float(np.mean(auc_rep(ys[p], oo[p]))))
        pd.DataFrame({"perm": range(len(null)), "T": null}).to_csv(nf, index=False)
        print(f"   permutations {len(null)}/{a.n_perm}  null mean {np.mean(null):.3f}  ({time.time() - t0:.0f}s)", flush=True)
    null = np.array(null[:a.n_perm])
    p_above = (1 + (null >= t_obs).sum()) / (1 + len(null)) if len(null) else np.nan
    p_below = (1 + (null <= t_obs).sum()) / (1 + len(null)) if len(null) else np.nan
    s = dict(model=a.model, task=a.task, groups=a.groups, rings=a.rings, n_features=len(cols), feature_names=cols, group_by=a.group_by, k=a.k, repeats=a.repeats, n_rows=len(df), n_eyes=int(df.eye_id.nunique()),
             n_subjects=int(df.Subject.nunique()), auc_mean=float(aucs.mean()),
             auc_sd_repeats=float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
             auc_ci_lo=float(np.percentile(boot, 2.5)), auc_ci_hi=float(np.percentile(boot, 97.5)),
             auc_per_repeat=[float(v) for v in aucs], perm_n=int(len(null)), perm_repeats=len(pr), perm_scheme=a.perm_scheme,
             T_obs=t_obs, null_mean=float(null.mean()) if len(null) else None, null_sd=float(null.std()) if len(null) else None,
             p_above=float(p_above), p_below=float(p_below),
             all_positive_f1=float(2 * y.mean() / (1 + y.mean())), runtime_s=time.time() - t0,
             **threshold_metrics(y, oof), args={k: (v if not isinstance(v, list) else [Path(x).name for x in v]) for k, v in vars(a).items() if k != "func"})
    json.dump(s, open(out / "summary.json", "w"), indent=1)
    print(f"   DONE {a.model}: AUC {s['auc_mean']:.3f}  p_above {p_above:.3f}  p_below {p_below:.3f}")


def cmd_aggregate(a):
    root = Path(a.out)
    models = [m for m in (*SKLEARN_MODELS, *DL_MODELS) if (root / m / "summary.json").exists()]
    if not models:
        sys.exit("no finished models found")
    S = {m: json.load(open(root / m / "summary.json")) for m in models}
    t0 = S[models[0]]
    df = apply_task(load_data(a.data), t0.get("task", "any")); y = df["yb"].to_numpy()
    OOF = {}
    for m in models:
        o = pd.read_csv(root / m / "oof.csv")
        OOF[m] = np.vstack([o[o.cv_rep == r].sort_values("row_index").y_prob.values for r in range(S[m]["repeats"])])
    rows = []
    for m in models:
        s = S[m]
        verdict = ("n/a (no permutation test)" if s.get("perm_n", 0) == 0 else
                   "SIGNAL" if (s["p_above"] < 0.05 and s["auc_mean"] >= 0.65)
                   else "BELOW CHANCE" if s["p_below"] < 0.05 else "NO DETECTABLE SIGNAL")
        rows.append(dict(model=m, task=s.get("task", "any"), groups=s.get("groups", "all"), rings=s.get("rings", "all"),
                         group_by=s["group_by"], n_eyes=s["n_eyes"], repeats=s["repeats"],
                         AUC=s["auc_mean"], AUC_lo=s["auc_ci_lo"], AUC_hi=s["auc_ci_hi"], AUC_sd_repeats=s["auc_sd_repeats"],
                         p_above=s["p_above"], p_below=s["p_below"], null_mean=s["null_mean"], null_sd=s["null_sd"],
                         sensitivity=s["sensitivity"], specificity=s["specificity"], balanced_acc=s["balanced_acc"],
                         pred_pos_rate=s["pred_pos_rate"], f1_amd=s["f1_amd"], f1_weighted=s["f1_weighted"],
                         all_positive_f1=s["all_positive_f1"], verdict=verdict))
    summ = pd.DataFrame(rows); summ.to_csv(root / "summary.csv", index=False)
    pairs = []
    for i, A in enumerate(models):
        for B in models[i + 1:]:
            R = min(len(OOF[A]), len(OOF[B]))
            obs = np.mean([roc_auc_score(y, OOF[A][r]) - roc_auc_score(y, OOF[B][r]) for r in range(R)])
            d = []
            for rows_ in eye_boot(df, a.n_boot, 42):
                yb = y[rows_]
                if yb.min() == yb.max():
                    continue
                d.append(np.mean([roc_auc_score(yb, OOF[A][r][rows_]) - roc_auc_score(yb, OOF[B][r][rows_]) for r in range(R)]))
            d = np.array(d)
            pairs.append(dict(comparison=f"{A} - {B}", dAUC=obs, lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                              boot_p_two_sided=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
    pd.DataFrame(pairs, columns=["comparison", "dAUC", "lo", "hi", "boot_p_two_sided"]).to_csv(root / "paired.csv", index=False)
    pd.set_option("display.width", 220)
    show = summ[["model", "AUC", "AUC_lo", "AUC_hi", "p_above", "p_below", "sensitivity", "specificity", "pred_pos_rate", "verdict"]]
    md = ["# Repeated nested eye-grouped CV", "", f"task = {summ.task.iloc[0]}, groups = {summ.groups.iloc[0]}, rings = {summ.rings.iloc[0]}, "
          f"group_by = {summ.group_by.iloc[0]}, eyes = {summ.n_eyes.iloc[0]}, repeats = {summ.repeats.iloc[0]}", "",
          show.round(3).to_markdown(index=False) if hasattr(show, "to_markdown") else show.round(3).to_string(index=False), "",
          "All-positive F1 baseline: %.3f" % summ.all_positive_f1.iloc[0], "", "Paired dAUC (eye-clustered bootstrap):", "",
          pd.DataFrame(pairs).round(3).to_string(index=False) if pairs else "n/a"]
    (root / "REPORT.md").write_text("\n".join(md) + "\n")
    print(show.round(3).to_string(index=False)); print()
    if pairs:
        print(pd.DataFrame(pairs).round(3).to_string(index=False))
    print("\nVERDICT per model:", dict(zip(summ.model, summ.verdict)))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "aggregate"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--data", nargs="+", default=["reruns_eyegrouped/split/train_mpod.csv", "reruns_eyegrouped/split/test_mpod.csv"])
        p.add_argument("--n_boot", type=int, default=2000)
        if name == "run":
            p.add_argument("--model", required=True, choices=[*SKLEARN_MODELS, *DL_MODELS])
            p.add_argument("--group_by", default="eye", choices=["eye", "subject"])
            p.add_argument("--task", default="any", choices=list(TASKS))
            p.add_argument("--groups", default="all", help="comma list of feature groups, e.g. Del,Amp  (default all 9)")
            p.add_argument("--rings", default="all", help="comma list of rings: centre,middle,outer (default all)")
            p.add_argument("--k", type=int, default=5)
            p.add_argument("--repeats", type=int, default=10)
            p.add_argument("--n_perm", type=int, default=200)
            p.add_argument("--perm_repeats", type=int, default=2)
            p.add_argument("--perm_scheme", default="block", choices=["block", "eye"])
            p.add_argument("--chunk", type=int, default=10)
            p.add_argument("--n_jobs", type=int, default=4)
            p.add_argument("--rf_trees", type=int, default=300)
            p.add_argument("--dl_epochs", type=int, default=80)
            p.add_argument("--dl_lr", type=float, default=3e-3)
            p.add_argument("--dl_wd", type=float, default=1e-2)
            p.add_argument("--dl_bs", type=int, default=16)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--force", action="store_true")
            p.set_defaults(func=cmd_run)
        else:
            p.set_defaults(func=cmd_aggregate)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
