"""Apply-only reimplementation of puf_models/data.py's four preprocessing views.

WHY THIS EXISTS
---------------
`data.py`'s `prepare_*(X_fit, X_other)` functions FIT on their first argument
every time they are called. Nothing about that fitted state (numeric medians /
means / stds, category vocabularies, one-hot layout, frequency-encoding scales)
was ever serialised alongside the models, so scoring a new site today means
dragging the 2M-row PUF fit split along and re-running the fit -- and silently
produces a different model input the moment someone passes the new site as
`X_fit`.

`fit_state()` captures that state once from the frozen bundle's own fit rows;
`apply_state()` reproduces the transform with no fitting at all. The two must
stay bit-compatible with data.py, so the ordering conventions it relies on are
mirrored here exactly and are called out in comments where they are load-bearing.
"""

import numpy as np
import pandas as pd

VIEWS = ("linear", "matrix", "tree", "embed")


def split_feature_types(X):
    """Numeric dtype -> numeric, everything else -> categorical.

    Order follows X.columns, and every view lays its output out as
    num_cols-then-cat_cols, so the caller must feed columns in the contract's
    declared order or the matrix silently transposes meaning.
    """
    cat_cols, num_cols = [], []
    for c in X.columns:
        (num_cols if pd.api.types.is_numeric_dtype(X[c]) else cat_cols).append(c)
    return num_cols, cat_cols


def _fit_numeric(X, num_cols):
    med = X[num_cols].median(numeric_only=True).fillna(0.0)
    filled = X[num_cols].fillna(med)
    mean = filled.mean()
    std = filled.std().replace(0.0, 1.0).fillna(1.0)
    return {"median": med, "mean": mean, "std": std}


def _num_impute(X, num_cols, st):
    return X[num_cols].fillna(st["median"]).to_numpy(dtype=np.float32)


def _num_impute_scale(X, num_cols, st):
    arr = X[num_cols].fillna(st["median"])
    arr = (arr - st["mean"]) / st["std"]
    return arr.to_numpy(dtype=np.float32)


# ── fit ──────────────────────────────────────────────────────────────────────

def fit_state(view, X_fit, onehot_max_card=30, max_vocab=1000):
    num_cols, cat_cols = split_feature_types(X_fit)
    st = {"view": view, "num_cols": list(num_cols), "cat_cols": list(cat_cols)}

    if view in ("linear", "matrix", "embed"):
        st["numeric"] = _fit_numeric(X_fit, num_cols)

    if view == "linear":
        cats = {}
        names = list(num_cols)
        for c in cat_cols:
            vc = X_fit[c].value_counts(dropna=True)
            if len(vc) <= onehot_max_card:
                # value_counts order (descending count) -- NOT sorted, NOT
                # first-appearance. get_dummies emits columns in this order.
                lv = [str(v) for v in vc.index]
                cats[c] = {"mode": "onehot", "levels": lv}
                names += [f"{c}={v}" for v in lv]
            else:
                freq = {str(k): float(v) for k, v in (vc / vc.sum()).items()}
                f = X_fit[c].map(freq).fillna(0.0).to_numpy(dtype=np.float32)
                sd = float(f.std())  # numpy ddof=0, matching data.py
                cats[c] = {"mode": "freq", "freq": freq,
                           "mu": float(f.mean()), "sd": sd or 1.0}
                names.append(f"{c}__freq")
        st["cats"] = cats
        st["feature_names"] = names

    elif view in ("matrix", "tree"):
        cats = {}
        for c in cat_cols:
            lv = [str(v) for v in pd.Index(X_fit[c].dropna().unique())]
            if view == "tree" and not lv:
                # Empty category array makes XGBoost's pandas transform call
                # np.vectorize on a size-0 input and raise. Same guard as data.py.
                lv = ["__all_missing__"]
            # first-appearance order from .unique(), which is what data.py uses
            # for these two views -- deliberately different from `linear`.
            cats[c] = {"mode": "codes", "levels": lv}
        st["cats"] = cats
        st["feature_names"] = list(num_cols) + list(cat_cols)

    elif view == "embed":
        cats, cards = {}, []
        for c in cat_cols:
            vc = X_fit[c].value_counts(dropna=True)
            keep = [str(v) for v in vc.index[:max_vocab]]
            cats[c] = {"mode": "vocab",
                       "vocab": {v: i + 1 for i, v in enumerate(keep)}}  # 0 = <unk>
            cards.append(len(keep) + 1)
        st["cats"] = cats
        st["cardinalities"] = cards
        st["feature_names"] = list(num_cols) + list(cat_cols)

    else:
        raise ValueError(f"unknown view: {view}")

    return st


# ── apply ────────────────────────────────────────────────────────────────────

def apply_state(st, X):
    """Transform X with an already-fitted state. Fits nothing."""
    view = st["view"]
    num_cols, cat_cols = st["num_cols"], st["cat_cols"]

    missing = [c for c in num_cols + cat_cols if c not in X.columns]
    if missing:
        raise KeyError(f"columns absent from the aligned frame: {missing}")

    if view == "tree":
        out = pd.DataFrame(index=X.index)
        for c in num_cols:
            out[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32)
        for c in cat_cols:
            dt = pd.CategoricalDtype(categories=st["cats"][c]["levels"])
            out[c] = X[c].astype("object").astype(dt)  # unseen/NaN -> NaN category
        return out

    stn = st["numeric"]
    if view == "linear":
        blocks = [_num_impute_scale(X, num_cols, stn)]
        for c in cat_cols:
            spec = st["cats"][c]
            if spec["mode"] == "onehot":
                cat = pd.Categorical(X[c].astype("object"),
                                     categories=spec["levels"])
                blocks.append(pd.get_dummies(cat).to_numpy(dtype=np.float32))
            else:
                f = X[c].astype("object").map(spec["freq"]).fillna(0.0)
                f = f.to_numpy(dtype=np.float32)
                blocks.append(((f - spec["mu"]) / spec["sd"]).reshape(-1, 1))
        return np.hstack(blocks)

    if view == "matrix":
        num = _num_impute(X, num_cols, stn)
        cat = np.empty((len(X), len(cat_cols)), dtype=np.float32)
        for j, c in enumerate(cat_cols):
            lv = st["cats"][c]["levels"]
            cat[:, j] = pd.Categorical(X[c].astype("object"), categories=lv).codes
        return np.hstack([num, cat])

    if view == "embed":
        xnum = _num_impute_scale(X, num_cols, stn)
        xcat = np.zeros((len(X), len(cat_cols)), dtype=np.int64)
        for j, c in enumerate(cat_cols):
            vocab = st["cats"][c]["vocab"]
            xcat[:, j] = X[c].astype("object").map(vocab).fillna(0).to_numpy(dtype=np.int64)
        return xnum, xcat

    raise ValueError(f"unknown view: {view}")
