#!/usr/bin/env python3
"""
make_synthetic_mat.py: SYNTHETIC data with the exact schema of the real experiment .mat (no patient data).

Writes
  <out>/synthetic.mat            struct d: MPOD.{mean..kurt}, Del, Amp (20x2x29x2), Del2, Amp2 (44x2x29x2),
                                 sevs (29x2), p.ring (1x3 struct array, field r)
  <out>/demographics_raw.csv     same columns as writetable(d.T): SbjID, Age, Sex, VaLE, VaRE, VaLcLE, VaLcRE,
                                 AREDSsevL, AREDSsevR, VisitN, VisitLast   (SbjID is fake)
  <out>/ofa44_bands.csv          example region -> band map for the 44-region field (FAKE layout, test only)

AREDS per eye = 26 / 15 / 10 / 7 (as in the real data). --mode signal plants an OFA -> low-contrast VA effect;
--mode null has none (used to check that the pipeline does not invent one).

  python tests/make_synthetic_mat.py --out /tmp/p2syn --mode signal --seed 1
  python -m src.mat_to_csv --mat /tmp/p2syn/synthetic.mat --out /tmp/p2syn/export --demographics /tmp/p2syn/demographics_raw.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import savemat

PAIRS = [(1, 1)] * 10 + [(1, 2)] * 6 + [(2, 2)] * 4 + [(2, 3)] + [(3, 3)] * 3 + [(3, 4)] * 3 + [(4, 4)] * 2  # 29 subjects
STATS = ["mean", "median", "std", "iqr", "idr", "skew", "kurt"]
RING = np.array([0] * 4 + [1] * 8 + [2] * 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["signal", "null"], default="signal")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S = len(PAIRS)
    order = rng.permutation(S)
    sevs = np.array([PAIRS[i] for i in order], dtype=float)            # [Subject x Eye]
    if rng.random() < 0.5:
        sevs = sevs[:, ::-1].copy()
    assert sorted(np.bincount(sevs.astype(int).ravel())[1:].tolist()) == sorted([26, 15, 10, 7])

    age = 68 + 2.0 * sevs.max(1) + rng.normal(0, 6, S)                  # older with worse grade (confounding)
    sex = rng.choice(["F", "M"], S)

    # eye-level latent states
    adv = (sevs >= 3).astype(float)
    del_lat = rng.normal(0, 1, (S, 2)) + 0.4 * (sevs - 1) + 0.6 * rng.normal(0, 1, (S, 1))   # fellow-eye correlation
    amp_lat = rng.normal(0, 1, (S, 2)) - 0.3 * (sevs - 1)
    mp_lat = rng.normal(0, 1, (S, 2)) - 0.8 * adv

    def sess(x_eye, noise, nreg):
        """[nreg x 2 x S x 2] from an eye-level [S x 2 x nreg] true value + session noise."""
        A = np.empty((nreg, 2, S, 2))
        for r in range(2):
            A[:, :, :, r] = np.transpose(x_eye + rng.normal(0, noise, x_eye.shape), (2, 1, 0))
        return A

    zone_eff = rng.normal(0, 0.3, 20)
    d = {"MPOD": {}}
    for k, st in enumerate(STATS):
        base = {"mean": 0.45, "median": 0.44, "std": 0.08, "iqr": 0.1, "idr": 0.2, "skew": 0.0, "kurt": 3.0}[st]
        ecc = np.array([1.0, 0.7, 0.45])[RING] if st in ("mean", "median") else np.ones(20)
        true = base * ecc[None, None, :] + 0.05 * mp_lat[:, :, None] * (RING[None, None, :] + 1) / 3 \
               + 0.03 * rng.normal(0, 1, (S, 2, 20))
        noise = 0.02 if st in ("mean", "median") else (0.02 + 0.03 * (RING >= 1))[None, None, :] * np.ones((S, 2, 20))
        d["MPOD"][st] = sess(true, noise, 20)
    d["Del"] = sess(del_lat[:, :, None] + zone_eff[None, None, :] + 0.5 * rng.normal(0, 1, (S, 2, 20)), 0.35, 20)
    d["Amp"] = sess(amp_lat[:, :, None] + 0.5 * rng.normal(0, 1, (S, 2, 20)), 0.5, 20)
    d["Del2"] = sess(del_lat[:, :, None] + 0.6 * rng.normal(0, 1, (S, 2, 44)), 0.45, 44)
    d["Amp2"] = sess(amp_lat[:, :, None] + 0.6 * rng.normal(0, 1, (S, 2, 44)), 0.6, 44)
    for k in ("Del", "Amp", "Del2", "Amp2"):                           # a few missing cells, as in real exports
        m = rng.random(d[k].shape) < 0.01
        d[k][m] = np.nan
    d["sevs"] = sevs
    ring = np.zeros((1, 3), dtype=[("r", "O")])
    for k, rr in enumerate([range(1, 5), range(5, 13), range(13, 21)]):
        ring[0, k]["r"] = np.array(list(rr), dtype=float).reshape(1, -1)
    d["p"] = {"ring": ring}
    savemat(out / "synthetic.mat", {"d": d}, do_compression=True)

    # visual acuity (logMAR; higher = worse). Planted effect of Del only in --mode signal.
    beta = 0.10 if a.mode == "signal" else 0.0
    eye_age = np.repeat(age[:, None], 2, 1)
    va_lc = 0.15 + 0.012 * (eye_age - 70) + 0.25 * adv + beta * del_lat + rng.normal(0, 0.10, (S, 2)) \
            + rng.normal(0, 0.05, (S, 1))
    va = 0.4 * va_lc + 0.15 * adv + rng.normal(0, 0.06, (S, 2)) - 0.05
    va[rng.random((S, 2)) < 0.03] = np.nan
    T = pd.DataFrame(dict(SbjID=[f"FAKE{i:03d}" for i in range(S)], Age=np.round(age, 1), Sex=sex,
                          VaLE=np.round(va[:, 0], 2), VaRE=np.round(va[:, 1], 2),
                          VaLcLE=np.round(va_lc[:, 0], 2), VaLcRE=np.round(va_lc[:, 1], 2),
                          AREDSsevL=sevs[:, 0], AREDSsevR=sevs[:, 1],
                          VisitN=rng.choice([1, 1, 1, 2, 3], S), VisitLast=rng.choice([0, 1], S)))
    T.to_csv(out / "demographics_raw.csv", index=False)
    pd.DataFrame(dict(region=np.arange(1, 45), band=np.where(np.arange(1, 45) <= 16, "central", "peripheral"))) \
        .to_csv(out / "ofa44_bands.csv", index=False)
    print(f"wrote {out}/synthetic.mat, demographics_raw.csv, ofa44_bands.csv  (mode={a.mode}, seed={a.seed})")


if __name__ == "__main__":
    main()
