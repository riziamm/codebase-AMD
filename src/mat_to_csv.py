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


def compare(new, old_path):
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
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None, help="old mpod.csv to verify against")
    ap.add_argument("--demographics", default=None, help="CSV from MATLAB: writetable(d.T,'demographics_raw.csv')")
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

    p = get(d, "p")
    ring = get(p, "ring") if p is not None else None
    if ring is not None:
        ring = np.asarray(ring, dtype=float).ravel()
        pd.DataFrame({"region": np.arange(1, len(ring) + 1), "ring": ring.astype(int)}).to_csv(out / "zone_rings.csv", index=False)
        print(f"wrote {out/'zone_rings.csv'}  rings: {pd.Series(ring.astype(int)).value_counts().sort_index().to_dict()}")

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
        sys.exit(0 if compare(mpod, a.compare) else 1)


if __name__ == "__main__":
    main()
