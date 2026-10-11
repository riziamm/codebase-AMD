#!/usr/bin/env python3
"""
p2_data.py: shared loading for Paper 2 (docs/ANALYSIS_PLAN_PAPER2.md). Pure functions, nothing exported.

Inputs (from src.mat_to_csv, verified export only, never reruns_SCRAMBLED_archive/)
  mpod.csv          180 features [mean..kurt, Del, Amp] x region1..20 + Subject, Repeat, Eye, Y
  ofa44.csv         88 features [Del2, Amp2] x region1..44 + Subject, Repeat, Eye, Y
  demographics.csv  Subject, Eye, Repeat, Age, Sex, VA, VAlc
  zone_rings.csv    region -> ring (1 centre, 2 middle, 3 outer); default 1-4 / 5-12 / 13-20
  ofa44_bands.csv   OPTIONAL region -> band for the 44-region field (fixed in the plan before running)

Eye level = mean of the two test sessions (pre-specified). Column names of summaries use '|' (e.g. 'Del|centre')
so that no output ever carries '*_region*' feature columns.
"""
import hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

MPOD_STATS = ["mean", "median", "std", "iqr", "idr", "skew", "kurt"]
BASE = MPOD_STATS + ["Del", "Amp"]
RING_NAME = {1: "centre", 2: "middle", 3: "outer"}
DEFAULT_RING = {z: 1 if z <= 4 else 2 if z <= 12 else 3 for z in range(1, 21)}
KEY = ["Subject", "Eye", "Repeat"]


# ------------------------------------------------------------------ loading
def _read(path, n_feat, prefixes, n_regions):
    df = pd.read_csv(path)
    exp = [f"{b}_region{z}" for b in prefixes for z in range(1, n_regions + 1)]
    if list(df.columns[:n_feat]) != exp:
        sys.exit(f"{path}: first {n_feat} columns are not {prefixes[0]}..{prefixes[-1]} x region1..{n_regions}")
    for c in KEY + ["Y"]:
        if c not in df.columns:
            sys.exit(f"{path}: missing column {c}")
    if len(df) != df[KEY].drop_duplicates().shape[0]:
        sys.exit(f"{path}: duplicate Subject/Eye/Repeat rows")
    return df.sort_values(KEY).reset_index(drop=True)


def load_sessions(export_dir):
    """Session-level tables (116 rows each), aligned on Subject/Eye/Repeat."""
    ex = Path(export_dir)
    mp = _read(ex / "mpod.csv", 180, BASE, 20)
    o44 = _read(ex / "ofa44.csv", 88, ["Del2", "Amp2"], 44) if (ex / "ofa44.csv").exists() else None
    if o44 is not None:
        if not (mp[KEY + ["Y"]].values == o44[KEY + ["Y"]].values).all():
            sys.exit("mpod.csv and ofa44.csv rows/labels do not align")
    if mp.groupby(["Subject", "Eye"])["Y"].nunique().max() > 1:
        sys.exit("AREDS grade Y not constant within an eye")
    return mp, o44


def load_rings(export_dir):
    p = Path(export_dir) / "zone_rings.csv"
    if not p.exists():
        return dict(DEFAULT_RING)
    r = pd.read_csv(p)
    ring = dict(zip(r.region.astype(int), r.ring.astype(int)))
    if set(ring) != set(range(1, 21)) or set(ring.values()) - {1, 2, 3}:
        sys.exit("zone_rings.csv must map regions 1..20 to rings 1..3")
    return ring


def load_bands44(path):
    """Optional region -> band map for the 44 regions. Returns None if no file."""
    if not path:
        return None
    b = pd.read_csv(path)
    if set(b.region.astype(int)) != set(range(1, 45)):
        sys.exit(f"{path}: must list regions 1..44 exactly once")
    return dict(zip(b.region.astype(int), b.band.astype(str)))


def encode_sex(s):
    """Map Sex to 0/1 deterministically (sorted unique labels). Returns (codes, mapping)."""
    vals = sorted(pd.Series(s).dropna().astype(str).unique())
    if len(vals) > 2:
        sys.exit(f"Sex has {len(vals)} levels; expected 2")
    mp = {v: i for i, v in enumerate(vals)}
    return pd.Series(s).astype(str).map(mp).where(pd.Series(s).notna()).astype(float).values, mp


def load_demo(export_dir):
    p = Path(export_dir) / "demographics.csv"
    if not p.exists():
        sys.exit(f"{p} not found: re-export with --demographics demographics_raw.csv")
    dm = pd.read_csv(p).sort_values(KEY).reset_index(drop=True)
    need = KEY + ["Age", "Sex", "VA", "VAlc"]
    miss = [c for c in need if c not in dm.columns]
    if miss:
        sys.exit(f"demographics.csv missing {miss}")
    # VA / VAlc / Age must be constant within an eye (one value per eye in d.T)
    for c in ("Age", "VA", "VAlc"):
        g = dm.groupby(["Subject", "Eye"])[c].nunique(dropna=True)
        if g.max() > 1:
            sys.exit(f"demographics: {c} differs between sessions of the same eye")
    return dm


# ------------------------------------------------------------------ summaries
def ring_summaries(df, ring):
    """Per row: mean over the zones of each ring (+ whole field 'all') for every base group."""
    out = {}
    for b in BASE:
        for rg in (1, 2, 3):
            cols = [f"{b}_region{z}" for z in range(1, 21) if ring[z] == rg]
            out[f"{b}|{RING_NAME[rg]}"] = df[cols].mean(1).values
        out[f"{b}|all"] = df[[f"{b}_region{z}" for z in range(1, 21)]].mean(1).values
    return pd.DataFrame(out, index=df.index)


def ofa44_summaries(df, bands=None):
    """Per row: 44-region field summaries. With a band map: band means; always: whole-field mean and worst."""
    out = {}
    for b in ("Del2", "Amp2"):
        cols = [f"{b}_region{z}" for z in range(1, 45)]
        out[f"{b}|all"] = df[cols].mean(1).values
        if bands:
            for bd in sorted(set(bands.values())):
                out[f"{b}|{bd}"] = df[[f"{b}_region{z}" for z in range(1, 45) if bands[z] == bd]].mean(1).values
    out["Del2|worst"] = df[[f"Del2_region{z}" for z in range(1, 45)]].max(1).values   # longest delay
    out["Amp2|worst"] = df[[f"Amp2_region{z}" for z in range(1, 45)]].min(1).values   # lowest sensitivity
    return pd.DataFrame(out, index=df.index)


def eye_table(export_dir, bands44_path=None):
    """One row per eye (58): session means of all features + ring/band summaries + demographics.
    Returns (E, info). E keeps Subject/Eye only for grouping inside the HPC; never write E to share/."""
    mp, o44 = load_sessions(export_dir)
    ring = load_rings(export_dir)
    bands = load_bands44(bands44_path)
    sess = pd.concat([mp, ring_summaries(mp, ring)], axis=1)
    if o44 is not None:
        sess = pd.concat([sess, o44.drop(columns=KEY + ["Y"]), ofa44_summaries(o44, bands)], axis=1)
    num = [c for c in sess.columns if c not in KEY + ["Y"]]
    E = sess.groupby(["Subject", "Eye"], as_index=False).agg({**{c: "mean" for c in num}, "Y": "first"})
    n_sess = sess.groupby(["Subject", "Eye"]).size().values
    dm = load_demo(export_dir).groupby(["Subject", "Eye"], as_index=False)[["Age", "Sex", "VA", "VAlc"]].first()
    E = E.merge(dm, on=["Subject", "Eye"], how="left", validate="1:1")
    E["Sex01"], sexmap = encode_sex(E["Sex"])
    E["eye_id"] = E["Subject"].astype(str) + "_" + E["Eye"].astype(str)
    E = E.sort_values(["Subject", "Eye"]).reset_index(drop=True)
    info = dict(n_eyes=len(E), n_subjects=int(E.Subject.nunique()), sessions_per_eye=sorted(set(n_sess.tolist())),
                areds_per_eye={int(k): int(v) for k, v in E.Y.value_counts().sort_index().items()},
                ring_map_default=(ring == DEFAULT_RING), bands44=sorted(set(bands.values())) if bands else None,
                sex_coding={str(k): v for k, v in sexmap.items()}, has_ofa44=o44 is not None)
    return E, info


# ------------------------------------------------------------------ feature sets
def cols_ring(groups, rings=("centre", "middle", "outer")):
    return [f"{g}|{r}" for g in groups for r in rings]


def cols_regions(groups):
    return [f"{g}_region{z}" for g in groups for z in range(1, 21)]


# ------------------------------------------------------------------ record keeping
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        c = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "--", "src", "scripts"], stderr=subprocess.DEVNULL) != 0
        return c + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def plan_status(plan="docs/ANALYSIS_PLAN_PAPER2.md"):
    p = Path(plan)
    if not p.exists():
        return dict(plan_sha256=None, signed_off=False)
    import re
    m = re.search(r"^\*\*Status:\s*(.+?)\*\*", p.read_text(), flags=re.M)   # the bold status line only
    return dict(plan_sha256=sha256(p), status=m.group(1).strip() if m else None,
                signed_off=bool(m) and m.group(1).strip().upper().startswith("SIGNED OFF"))


def require_signoff(allow_unsigned):
    st = plan_status()
    if not st["signed_off"] and not allow_unsigned:
        sys.exit("STOP: docs/ANALYSIS_PLAN_PAPER2.md is not marked 'Status: SIGNED OFF'. "
                 "Share the plan with co-authors first (pre-specification). Synthetic tests: --allow_unsigned.")
    return st


def write_run_info(out, args, export_dir, extra=None):
    """Provenance: commit, plan hash, input-file hashes (hashes only, no data), arguments, time."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ex = Path(export_dir)
    files = {f.name: sha256(f) for f in sorted(ex.glob("*.csv"))}
    info = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), git_commit=git_commit(), **plan_status(),
                python=sys.version.split()[0], args=vars(args), input_sha256=files)
    try:
        import sklearn, scipy
        info["versions"] = dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                                scipy=scipy.__version__)
    except Exception:
        pass
    if extra:
        info.update(extra)
    json.dump(info, open(out / "run_info.json", "w"), indent=1, default=str)
    return info


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
