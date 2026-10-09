#!/usr/bin/env python3
"""
mat_to_csv.py: unroll the experiment .mat into the flat CSV layout used by the pipeline.

Data in the .mat (struct `d`), all 4-D arrays are [Region x Eye x Subject x Repeat]
  Eye order: 1 = left, 2 = right.          d.sevs: [Subject x Eye] AREDS grade 1-4
  d.MPOD.{mean,median,std,iqr,idr,skew,kurt}  20 x 2 x 29 x 2   (MPOD summary stats per OFA region)
  d.Del, d.Amp                                20 x 2 x 29 x 2   (OFA total deviations, 20 macular regions)
  d.Del2, d.Amp2                              44 x 2 x 29 x 2   (OFA total deviations, 44-region wider field)

Outputs (row order = Subject -> Eye -> Repeat, identical to the original mpod.csv)
  mpod.csv          180 features [mean..kurt, Del, Amp] x region1..20  + Subject, Repeat, Eye, Y
  ofa44.csv         88 features  [Del2, Amp2] x region1..44            + Subject, Repeat, Eye, Y
  zone_rings.csv    region -> ring for the 20 regions (if d.p.ring exists)
  demographics.csv  per row: Subject, Eye, Repeat, Age, Sex, VA, VAlc   (only if --demographics given)
                    (MATLAB tables can't be read from .mat by Python; export once in MATLAB:
                       writetable(d.T, 'demographics_raw.csv') )

Usage
  python -m src.mat_to_csv --mat path/to/experiment.mat --out data/export
  python -m src.mat_to_csv --mat ... --out data/export --compare data/mpod.csv        # verify vs old CSV
  python -m src.mat_to_csv --mat ... --out data/export --demographics demographics_raw.csv
SbjID is never written (privacy); Subject = position 1..29 in the .mat, as in the original CSV.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

MPOD_STATS = ["mean", "median", "std", "iqr", "idr", "skew", "kurt"]
ID_COLS = ["Subject", "Repeat", "Eye", "Y"]


# ------------------------------------------------------------------ loading
def _load_scipy(path):
    from scipy.io import loadmat
    m = loadmat(path, squeeze_me=False, struct_as_record=False)
    d = m["d"][0, 0] if "d" in m else None
    if d is None:  # saved with -struct: fields at top level
        class _D: pass
        d = _D()
        for k, v in m.items():
            if not k.startswith("__"):
                setattr(d, k, v)

    def get(obj, name):
        v = getattr(obj, name, None)
        if v is None:
            return None
        if isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
            v = v.flat[0]
        return v
    return d, get


def _load_h5(path):  # MATLAB v7.3 (HDF5): arrays are stored with reversed axes
    import h5py
    f = h5py.File(path, "r")
    d = f["d"] if "d" in f else f

    def get(obj, name):
        if name not in obj:
            return None
        v = obj[name]
        if isinstance(v, h5py.Group):
            return v
        return np.array(v).T  # undo column-major reversal
    return d, get


def load_mat(path):
    try:
        return _load_scipy(path)
    except (NotImplementedError, ValueError) as e:   # v7.3 (HDF5) files are not readable by scipy
        try:
            return _load_h5(path)
        except Exception:
            raise e


def arr(get, obj, name, shape_regions):
    v = get(obj, name)
    if v is None:
        return None
    v = np.asarray(v, dtype=float)
    exp = (shape_regions, 2, 29, 2)
    if v.shape[:2] != exp[:2] or v.ndim != 4 or v.shape[3] != 2:
        sys.exit(f"{name}: shape {v.shape}, expected [{shape_regions} x 2 x nSbj x 2] (Region x Eye x Subject x Repeat)")
    return v


# ------------------------------------------------------------------ unrolling
def unroll(blocks, names, sevs):
    """blocks: list of [R x 2 x S x 2] arrays; names: column prefixes. Row order Subject -> Eye -> Repeat."""
    S = blocks[0].shape[2]
    rows, ids = [], []
    for s in range(S):
        for e in range(2):
            for r in range(2):
                rows.append(np.concatenate([b[:, e, s, r] for b in blocks]))
                ids.append((s + 1, r + 1, e + 1, sevs[s, e]))
    cols = [f"{n}_region{z}" for n, b in zip(names, blocks) for z in range(1, b.shape[0] + 1)]
    df = pd.DataFrame(np.vstack(rows), columns=cols)
    idf = pd.DataFrame(ids, columns=ID_COLS)
    if idf["Y"].isna().any():
        sys.exit("d.sevs has missing grades")
    idf["Y"] = idf["Y"].astype(int)
    return pd.concat([df, idf], axis=1)


def _fields(obj):
    f = getattr(obj, "_fieldnames", None)          # scipy mat_struct
    if f is None and hasattr(obj, "keys"):           # h5py group
        f = list(obj.keys())
    return list(f) if f else []


def write_rings(d, get, out, ring_field=None):
    """d.p.ring may be numeric (20 values) or a struct. For a struct, use --ring_field, else the first
    numeric field with 20 values (region -> ring number) or a field holding 3 lists of region numbers."""
    p = get(d, "p")
    ring = get(p, "ring") if p is not None else None
    if ring is None:
        print("d.p.ring not found -> zone_rings.csv skipped"); return
    if isinstance(ring, np.ndarray) and ring.dtype == object and ring.size == 3:
        # struct ARRAY (1x3): element k holds the regions of ring k, e.g. d.p.ring(k).r
        f = ring_field or (_fields(ring.flat[0]) or ["r"])[0]
        lists = [np.asarray(getattr(el, f), dtype=float).ravel() for el in ring.ravel()]
        if sorted(np.concatenate(lists).astype(int).tolist()) != list(range(1, 21)):
            raise ValueError(f"d.p.ring(k).{f} does not cover regions 1..20 exactly once")
        vec = np.zeros(20)
        for k, regs in enumerate(lists, start=1):
            vec[regs.astype(int) - 1] = k
        print(f"d.p.ring is a 1x3 struct array; using d.p.ring(k).{f}")
        ring = vec
    fields = _fields(ring) if not isinstance(ring, np.ndarray) else []
    if fields:
        print(f"d.p.ring is a struct with fields: {fields}")
        cand = [ring_field] if ring_field else fields
        vec = None
        for f in cand:
            v = get(ring, f)
            try:
                arr_ = np.asarray(v, dtype=float).ravel()
            except Exception:
                arr_ = None
            if arr_ is not None and arr_.size == 20:
                vec, used = arr_, f; break
            if v is not None and np.asarray(v, dtype=object).size == 3:   # 3 cells of region lists
                try:
                    lists = [np.asarray(x, dtype=float).ravel() for x in np.asarray(v, dtype=object).ravel()]
                    if sorted(np.concatenate(lists).astype(int).tolist()) == list(range(1, 21)):
                        vec = np.zeros(20)
                        for k, regs in enumerate(lists, start=1):
                            vec[regs.astype(int) - 1] = k
                        used = f; break
                except Exception:
                    pass
        if vec is None:
            raise ValueError(f"no 20-value or 3-list field found in d.p.ring (fields {fields}); pass --ring_field")
        print(f"  using d.p.ring.{used}")
        ring = vec
    ring = np.asarray(ring, dtype=float).ravel()
    if ring.size != 20:
        raise ValueError(f"ring map has {ring.size} values, expected 20")
    pd.DataFrame({"region": np.arange(1, 21), "ring": ring.astype(int)}).to_csv(out / "zone_rings.csv", index=False)
    print(f"wrote {out/'zone_rings.csv'}  rings: {pd.Series(ring.astype(int)).value_counts().sort_index().to_dict()}")


def layout_search(blocks4d, old):
    """Was the old CSV a re-shaped (scrambled) version of the same 4-D arrays?  Tries every axis order
    x C/F memory order x both reshape directions, and checks value multisets."""
    from itertools import permutations
    print("\nLAYOUT SEARCH (is the old CSV the same numbers, unrolled differently?)")
    axes = ["Region", "Eye", "Subject", "Repeat"]
    any_hit = False
    for blk, A in blocks4d.items():
        Y = old[[f"{blk}_region{z}" for z in range(1, 21)]].to_numpy(float)   # old file, ORIGINAL row order
        same_values = np.allclose(np.sort(A.ravel()), np.sort(Y.ravel()), equal_nan=True)
        hits = []
        for p in permutations(range(4)):
            B = np.transpose(A, p)
            for order in ("C", "F"):
                for how in ("rows", "cols"):
                    M = B.reshape(116, 20, order=order) if how == "rows" else B.reshape(20, 116, order=order).T
                    if np.allclose(M, Y, equal_nan=True):
                        hits.append(f"np.transpose(A,{p}).reshape({'116,20' if how=='rows' else '20,116'},order='{order}')"
                                    + (".T" if how == "cols" else "") + f"   [axes order {[axes[i] for i in p]}]")
        any_hit |= bool(hits)
        print(f"  {blk:6s} same multiset of values: {str(same_values):5s}  exact layout found: "
              f"{hits[0] if hits else 'none'}")
    if not any_hit:
        print("  No reshape of the .mat arrays reproduces the old CSV.")
        print("  If 'same multiset' is False -> the old CSV came from DIFFERENT data (another .mat / version).")
        print("  If 'same multiset' is True  -> same numbers, scrambled in a way not covered here; send me the old unwrapping code.")
    else:
        print("  => the old CSV used the layout above. Compare it with the CORRECT unrolling (Subject -> Eye -> Repeat rows,")
        print("     20 regions per feature): if they differ, the old features were attached to the wrong eyes/subjects.")


def compare(new, old_path, blocks4d=None):
    old = pd.read_csv(old_path)
    key = ["Subject", "Eye", "Repeat"]
    a = new.sort_values(key).reset_index(drop=True); b = old.sort_values(key).reset_index(drop=True)
    ok = True
    if list(a.columns) != list(b.columns):
        print(f"  COLUMNS differ: new {len(a.columns)} vs old {len(b.columns)}"); ok = False
    if len(a) != len(b) or not (a[ID_COLS].values == b[ID_COLS].values).all():
        print("  ID/LABEL columns differ (Subject/Repeat/Eye/Y)"); ok = False
    else:
        print("  ID/LABEL columns identical (Subject, Repeat, Eye, Y)")
    common = [c for c in a.columns if c in b.columns and c not in ID_COLS]
    if b[common].isna().all().all():
        print("  old file has no feature values (blanked copy) -> only IDs/labels compared")
    else:
        for blk in MPOD_STATS + ["Del", "Amp"]:
            cs = [c for c in common if c.split("_region")[0] == blk]
            x, y = a[cs].to_numpy(float), b[cs].to_numpy(float)
            same_nan = np.array_equal(np.isnan(x), np.isnan(y))
            diff = np.nanmax(np.abs(x - y)) if np.isfinite(x - y).any() else 0.0
            flag = "OK " if same_nan and diff < 1e-6 else "DIFF"
            ok &= flag == "OK "
            print(f"  {flag} {blk:6s} max|new-old| = {diff:.2e}  NaN pattern {'same' if same_nan else 'DIFFERENT'}")
    print("COMPARE:", "IDENTICAL to old CSV" if ok else "MISMATCH (see above)")
    if not ok and not b[common].isna().all().all():
        diagnose(a, b)
        if blocks4d is not None:
            layout_search(blocks4d, old)
    return ok


def diagnose(a, b):
    """Find HOW the old CSV relates to the .mat export (row mapping, region order, sorting, scaling)."""
    from scipy.spatial.distance import cdist
    print("\nDIAGNOSIS (new = from .mat, old = your CSV)")
    print(f"  {'block':6s} {'corr':>6s} {'slope':>6s} {'rows matched':>13s} {'same subj':>9s} {'same eye':>8s} "
          f"{'same rep':>8s} {'sorted-row match':>16s} {'region perm':>11s}")
    for blk in MPOD_STATS + ["Del", "Amp"]:
        cs = [f"{blk}_region{z}" for z in range(1, 21)]
        X, Y = a[cs].to_numpy(float), b[cs].to_numpy(float)
        x, y = X.ravel(), Y.ravel(); m = np.isfinite(x) & np.isfinite(y)
        corr = np.corrcoef(x[m], y[m])[0, 1]
        slope = np.polyfit(x[m], y[m], 1)[0]
        scale = np.nanstd(Y) + 1e-12
        D = cdist(np.nan_to_num(Y), np.nan_to_num(X))           # old rows vs new rows
        j = D.argmin(1); dmin = D[np.arange(len(Y)), j] / scale
        hit = dmin < 1e-6
        same = lambda col: (b[col].values[hit] == a[col].values[j[hit]]).mean() if hit.any() else np.nan
        srt = np.nanmax(np.abs(np.sort(X, 1) - np.sort(Y, 1))) / scale < 1e-6
        C = np.corrcoef(np.nan_to_num(X).T, np.nan_to_num(Y).T)[:20, 20:]   # new region i vs old region k
        best = C.argmax(0)
        perm = "identity" if (best == np.arange(20)).all() else ("permuted" if len(set(best)) == 20 and C.max(0).min() > .99 else "no 1:1")
        print(f"  {blk:6s} {corr:6.3f} {slope:6.2f} {hit.mean():12.0%} {same('Subject'):9.0%} {same('Eye'):8.0%} "
              f"{same('Repeat'):8.0%} {str(srt):>16s} {perm:>11s}")
    print("  Read: rows matched 100% + same subj 100% + same eye 0%  => eyes swapped in the old CSV;")
    print("        sorted-row match True => old CSV had zone values sorted within each feature;")
    print("        corr ~1 but slope != 1 => old values were rescaled/normalised; corr ~0 => different source data.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None, help="old mpod.csv to verify against")
    ap.add_argument("--demographics", default=None, help="CSV from MATLAB: writetable(d.T,'demographics_raw.csv')")
    ap.add_argument("--ring_field", default=None, help="field of the d.p.ring struct holding the region list/map (default r)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d, get = load_mat(a.mat)

    sevs = np.asarray(get(d, "sevs"), dtype=float)
    mp = get(d, "MPOD")
    if mp is None:
        sys.exit("d.MPOD not found")
    core = [arr(get, mp, s, 20) for s in MPOD_STATS] + [arr(get, d, "Del", 20), arr(get, d, "Amp", 20)]
    if any(b is None for b in core):
        sys.exit("missing one of d.MPOD.<stat>, d.Del, d.Amp")
    S = core[0].shape[2]
    if sevs.shape != (S, 2):
        sys.exit(f"d.sevs shape {sevs.shape}, expected ({S}, 2) [Subject x Eye]")

    mpod = unroll(core, MPOD_STATS + ["Del", "Amp"], sevs)
    mpod.to_csv(out / "mpod.csv", index=False)
    print(f"wrote {out/'mpod.csv'}  shape {mpod.shape}  subjects={S}  NaN cells={int(mpod.iloc[:, :180].isna().sum().sum())}")
    print("  AREDS per eye:", mpod.drop_duplicates(['Subject', 'Eye']).Y.value_counts().sort_index().to_dict())

    d2, a2 = arr(get, d, "Del2", 44), arr(get, d, "Amp2", 44)
    if d2 is not None and a2 is not None:
        ofa = unroll([d2, a2], ["Del2", "Amp2"], sevs)
        ofa.to_csv(out / "ofa44.csv", index=False)
        print(f"wrote {out/'ofa44.csv'}  shape {ofa.shape}  NaN cells={int(ofa.iloc[:, :88].isna().sum().sum())}")
    else:
        print("Del2/Amp2 not found -> ofa44.csv skipped")

    try:
        write_rings(d, get, out, a.ring_field)
    except Exception as e:  # never block the main export on the optional ring map
        print(f"WARNING: zone_rings.csv skipped ({e}). Default rings 1-4/5-12/13-20 will be used.")

    if a.demographics:
        T = pd.read_csv(a.demographics)
        if len(T) != S:
            sys.exit(f"demographics has {len(T)} rows, expected {S}")
        rows = []
        for s in range(S):
            for e in range(2):
                side = "L" if e == 0 else "R"
                for r in range(2):
                    rows.append(dict(Subject=s + 1, Eye=e + 1, Repeat=r + 1, Age=T.loc[s, "Age"], Sex=T.loc[s, "Sex"],
                                     VA=T.get(f"Va{side}E", pd.Series([np.nan] * S))[s],
                                     VAlc=T.get(f"VaLc{side}E", pd.Series([np.nan] * S))[s]))
                # sanity: AREDS in T must match d.sevs
                col = f"AREDSsev{side}"
                if col in T and not np.isnan(T.loc[s, col]) and T.loc[s, col] != sevs[s, e]:
                    print(f"  WARNING subject {s+1} eye {side}: T.{col}={T.loc[s, col]} but d.sevs={sevs[s, e]}")
        pd.DataFrame(rows).to_csv(out / "demographics.csv", index=False)
        print(f"wrote {out/'demographics.csv'} (SbjID not written)")

    if a.compare:
        print(f"\nComparing with {a.compare}:")
        b4 = dict(zip(MPOD_STATS + ["Del", "Amp"], core))
        sys.exit(0 if compare(mpod, a.compare, b4) else 1)


if __name__ == "__main__":
    main()
