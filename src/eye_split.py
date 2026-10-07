"""
eye_split.py — drop-in helpers so the existing pipeline splits by EYE.

Put this file next to main.py. All functions are pure (no data written).

    from eye_split import eye_ids_from_df, grouped_train_test_split, grouped_cv

    groups = eye_ids_from_df(data)              # BEFORE any column filtering
    X_tr, X_te, y_tr, y_te, g_tr, g_te = grouped_train_test_split(
        X, y, groups, test_size=0.15, random_state=42, stratify_on=y4)
    cv = grouped_cv(y_tr, g_tr, n_splits=5, random_state=42)  # list of (train, test)
    GridSearchCV(..., cv=cv)                    # sklearn accepts an iterable of splits

Logic: AREDS grade is constant within an eye, so stratifying the table of
unique eyes and mapping back to rows is an exact stratified *group* split.
Compatible with scikit-learn 0.24+.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def eye_ids_from_df(df, subject_col="Subject", eye_col="Eye"):
    return (df[subject_col].astype(str) + "_" + df[eye_col].astype(str)).values


def _eye_table(y, groups):
    t = pd.DataFrame({"g": groups, "y": y})
    if (t.groupby("g")["y"].nunique() > 1).any():
        raise ValueError("Label not constant within an eye; cannot stratify by eye.")
    return t.drop_duplicates("g").sort_values("g").reset_index(drop=True)  # same order as make_split.py


def grouped_train_test_split(X, y, groups, test_size=0.15, random_state=42, stratify_on=None):
    """Drop-in for train_test_split(X, y, test_size, random_state, stratify=y).
    Returns X_train, X_test, y_train, y_test, groups_train, groups_test."""
    X = np.asarray(X); y = np.asarray(y); groups = np.asarray(groups)
    strat = np.asarray(stratify_on) if stratify_on is not None else y
    eyes = _eye_table(strat, groups)
    n_splits = max(2, int(round(1.0 / test_size)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    _, te = next(iter(skf.split(eyes["g"], eyes["y"])))
    test_eyes = set(eyes.loc[te, "g"])
    m = np.isin(groups, list(test_eyes))
    return X[~m], X[m], y[~m], y[m], groups[~m], groups[m]


def grouped_cv(y, groups, n_splits=5, random_state=42, stratify_on=None):
    """Return a list of (train_idx, test_idx) with no eye shared across folds.
    Pass directly as cv= to GridSearchCV / cross_validate / your DL fold loop."""
    y = np.asarray(y); groups = np.asarray(groups)
    strat = np.asarray(stratify_on) if stratify_on is not None else y
    eyes = _eye_table(strat, groups)
    k = min(n_splits, eyes["y"].value_counts().min())  # every class in every fold
    k = max(k, 2)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    splits = []
    for _, te in skf.split(eyes["g"], eyes["y"]):
        te_eyes = set(eyes.loc[te, "g"])
        m = np.isin(groups, list(te_eyes))
        splits.append((np.where(~m)[0], np.where(m)[0]))
    return splits


def assert_no_eye_overlap(groups_a, groups_b, label="split"):
    shared = set(np.asarray(groups_a)) & set(np.asarray(groups_b))
    if shared:
        raise AssertionError(f"{label}: {len(shared)} eye(s) in both partitions, e.g. {list(shared)[:3]}")
    print(f"[eye_split] {label}: OK, no shared eyes "
          f"({len(set(groups_a))} vs {len(set(groups_b))} eyes)")


class EyeGroupedKFold:
    """Drop-in replacement for StratifiedKFold where the caller cannot pass groups.

        skf = EyeGroupedKFold(groups_train, n_splits=5, random_state=42)
        for tr, va in skf.split(X_train, y_train): ...
        GridSearchCV(..., cv=skf)

    `groups` must be aligned row-by-row with the X passed to .split().
    """
    def __init__(self, groups, n_splits=5, random_state=42, stratify_on=None):
        self.groups = np.asarray(groups)
        self.n_splits = n_splits
        self.random_state = random_state
        self.stratify_on = stratify_on

    def split(self, X, y=None, groups=None):
        if len(X) != len(self.groups):
            raise ValueError(f"EyeGroupedKFold: X has {len(X)} rows but groups has {len(self.groups)}")
        y = np.asarray(y) if y is not None else np.zeros(len(X))
        for tr, te in grouped_cv(y, self.groups, self.n_splits, self.random_state, self.stratify_on):
            yield tr, te

    def get_n_splits(self, X=None, y=None, groups=None):
        if y is None:
            return self.n_splits
        return len(grouped_cv(np.asarray(y), self.groups, self.n_splits,
                              self.random_state, self.stratify_on))
