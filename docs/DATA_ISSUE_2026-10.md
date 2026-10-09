# Data issue found October 2026: the original `mpod.csv` was scrambled

## Finding
`src/mat_to_csv.py --compare` found that the `mpod.csv` behind the submitted manuscript (and all reruns
up to Oct 2026) contains the **same numbers** as the `.mat` files (2024 and current versions; identical),
but **unrolled with the wrong axis order**. The layout search reproduced it exactly for all 9 feature blocks:

    old_block = np.transpose(A, (2, 1, 0, 3)).reshape(116, 20, order='F')
    A = d.<block>  with axes [Region x Eye x Subject x Repeat] = 20 x 2 x 29 x 2

## Consequence
- Old row *i* holds data of subject `i % 29 + 1`, eye `(i // 29) % 2 + 1`, but was labelled with the
  Subject/Eye/AREDS of row *i* in Subject -> Eye -> Repeat order.
  Only **3.4 %** of rows carry features from the eye whose label they have.
- Each old "row" held **10 regions x 2 repeat sessions** (odd regions for rows 0-57, even for 58-115),
  not 20 regions of one session; so the "region" columns were not regions.
- Binary-label agreement between a row's label and its feature-source eye: 60.3 % (chance 50.5 %).

So all models, metrics, SHAP region maps and ring-level interpretations derived from the old CSV are
not interpretable. Old outputs are kept for the record in `share/ARCHIVE_scrambled_input/`.

## Fix
`python -m src.mat_to_csv --mat <file>.mat --out data/export` writes the correct layout:
rows Subject -> Eye -> Repeat, columns [mean, median, std, iqr, idr, skew, kurt, Del, Amp] x region 1..20.
IDs/labels and row order are identical to the old file, so `make_split.py` reproduces the same frozen
holdout eyes (manifest SHA-256 unchanged); only the feature values are corrected.
