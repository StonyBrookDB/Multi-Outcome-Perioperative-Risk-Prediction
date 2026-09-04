"""
Loading + preprocessing for the PUF model zoo.

X_train/X_test are shared across all outcomes, so a preprocessing view is fit
once on X_train and reused for every label. Three views are provided, each
fit on train and applied to test with no leakage:

  linear  -> dense float32 matrix: median-imputed + standardized numerics,
             one-hot for low-cardinality categoricals, frequency-encoding for
             high-cardinality ones.  (logistic regression)
  matrix  -> dense float32 matrix: median-imputed numerics (no scaling) +
             ordinal-encoded categoricals (missing/unseen -> -1).
             (random forest -- sklearn RF can't take NaN or category dtype)
  tree    -> a pandas frame with NaN-carrying float numerics + pandas
             'category' dtype categoricals, categories fixed from train.
             (xgboost enable_categorical / lightgbm native categorical)
  embed   -> (X_num float32 [scaled+imputed], X_cat int64 [top-K codes, 0=unk],
             cardinalities) for entity-embedding nets. (mlp, ft_transformer)
"""

import os

import numpy as np
import pandas as pd

import config


# ── Loading ──────────────────────────────────────────────────────────────────

def load_X():
    X_train = pd.read_parquet(os.path.join(config.X_DIR, "X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(config.X_DIR, "X_test.parquet"))
    # keep identical column order
    X_test = X_test[X_train.columns]
    return X_train, X_test


def load_y(label):
    ytr = pd.read_parquet(
        os.path.join(config.Y_DIR, f"Y_train_{label}.parquet")
    )[label].to_numpy().astype(np.float32)
    yte = pd.read_parquet(
        os.path.join(config.Y_DIR, f"Y_test_{label}.parquet")
    )[label].to_numpy().astype(np.float32)
    return ytr, yte


def load_benchmark():
    btr = pd.read_parquet(os.path.join(config.BENCH_DIR, "benchmark_train.parquet"))
    bte = pd.read_parquet(os.path.join(config.BENCH_DIR, "benchmark_test.parquet"))
    return btr, bte


def split_feature_types(X):
    """Numeric dtype -> numeric, everything else -> categorical.

    build_dataset guarantees every column is either int64/float64 or a string
    category, so testing for numeric dtype is equivalent to testing for
    pd.StringDtype -- but stable across pandas versions: the same parquet
    string column round-trips to pd.StringDtype under pandas 3.x and plain
    `object` under pandas 2.x, and the StringDtype check would then silently
    treat every categorical as numeric and crash in _fit_numeric. (Same fix as
    ../sbuh_models/data.py.)
    """
    cat_cols, num_cols = [], []
    for c in X.columns:
        (num_cols if pd.api.types.is_numeric_dtype(X[c]) else cat_cols).append(c)
    return num_cols, cat_cols


# ── Numeric helpers ──────────────────────────────────────────────────────────

def _fit_numeric(X_train, num_cols):
    # A column that is entirely NaN in train has a NaN median; fall back to 0
    # so downstream imputation cannot leak NaN into sklearn LR/RF (which reject
    # it). build_dataset already drops fully-missing columns, so this only bites
    # on subsamples, but keep it robust.
    med = X_train[num_cols].median(numeric_only=True).fillna(0.0)
    filled = X_train[num_cols].fillna(med)
    mean = filled.mean()
    std = filled.std().replace(0.0, 1.0).fillna(1.0)
    return {"median": med, "mean": mean, "std": std}


def _num_impute(X, num_cols, stats):
    return X[num_cols].fillna(stats["median"]).to_numpy(dtype=np.float32)


def _num_impute_scale(X, num_cols, stats):
    arr = X[num_cols].fillna(stats["median"])
    arr = (arr - stats["mean"]) / stats["std"]
    return arr.to_numpy(dtype=np.float32)


# ── linear view (logistic regression) ────────────────────────────────────────

def prepare_linear(X_train, X_test):
    num_cols, cat_cols = split_feature_types(X_train)
    stats = _fit_numeric(X_train, num_cols)

    blocks_tr = [_num_impute_scale(X_train, num_cols, stats)]
    blocks_te = [_num_impute_scale(X_test, num_cols, stats)]
    feature_names = list(num_cols)

    for c in cat_cols:
        vc = X_train[c].value_counts(dropna=True)
        card = len(vc)
        if card <= config.ONEHOT_MAX_CARD:
            cats = list(vc.index)
            tr = pd.Categorical(X_train[c], categories=cats)
            te = pd.Categorical(X_test[c], categories=cats)
            oh_tr = pd.get_dummies(tr).to_numpy(dtype=np.float32)
            oh_te = pd.get_dummies(te).to_numpy(dtype=np.float32)
            blocks_tr.append(oh_tr)
            blocks_te.append(oh_te)
            feature_names += [f"{c}={v}" for v in cats]
        else:
            # frequency encoding -> single standardized numeric column
            freq = (vc / vc.sum()).to_dict()
            f_tr = X_train[c].map(freq).fillna(0.0).to_numpy(dtype=np.float32)
            f_te = X_test[c].map(freq).fillna(0.0).to_numpy(dtype=np.float32)
            mu, sd = f_tr.mean(), f_tr.std() or 1.0
            blocks_tr.append(((f_tr - mu) / sd).reshape(-1, 1))
            blocks_te.append(((f_te - mu) / sd).reshape(-1, 1))
            feature_names.append(f"{c}__freq")

    Xtr = np.hstack(blocks_tr)
    Xte = np.hstack(blocks_te)
    return Xtr, Xte, feature_names


# ── matrix view (random forest) ──────────────────────────────────────────────

def prepare_matrix(X_train, X_test):
    num_cols, cat_cols = split_feature_types(X_train)
    stats = _fit_numeric(X_train, num_cols)

    num_tr = _num_impute(X_train, num_cols, stats)
    num_te = _num_impute(X_test, num_cols, stats)

    cat_tr = np.empty((len(X_train), len(cat_cols)), dtype=np.float32)
    cat_te = np.empty((len(X_test), len(cat_cols)), dtype=np.float32)
    for j, c in enumerate(cat_cols):
        cats = pd.Index(X_train[c].dropna().unique())
        cat_tr[:, j] = pd.Categorical(X_train[c], categories=cats).codes  # -1 = NaN
        cat_te[:, j] = pd.Categorical(X_test[c], categories=cats).codes   # -1 = unseen/NaN

    Xtr = np.hstack([num_tr, cat_tr])
    Xte = np.hstack([num_te, cat_te])
    return Xtr, Xte, list(num_cols) + list(cat_cols)


# ── tree view (xgboost / lightgbm native categorical) ────────────────────────

def prepare_tree(X_train, X_test):
    num_cols, cat_cols = split_feature_types(X_train)

    Xtr = pd.DataFrame(index=X_train.index)
    Xte = pd.DataFrame(index=X_test.index)

    for c in num_cols:
        Xtr[c] = pd.to_numeric(X_train[c], errors="coerce").astype(np.float32)
        Xte[c] = pd.to_numeric(X_test[c], errors="coerce").astype(np.float32)

    for c in cat_cols:
        cats = pd.Index(X_train[c].dropna().unique())
        if len(cats) == 0:
            # Column entirely missing in train: an empty category array makes
            # XGBoost's pandas transform call np.vectorize on a size-0 input
            # and raise "cannot call `vectorize` on size 0 inputs" (LightGBM
            # tolerates it). One placeholder level keeps the matrix shape and
            # column order identical to the other views; every row stays NaN,
            # so the column carries no signal either way. Same guard as
            # ../sbuh_models/data.py, where an earlier build has 25 such columns.
            cats = pd.Index(["__all_missing__"])
        dtype = pd.CategoricalDtype(categories=cats)
        Xtr[c] = X_train[c].astype(dtype)   # unseen/NaN -> NaN category
        Xte[c] = X_test[c].astype(dtype)

    return Xtr, Xte, num_cols, cat_cols


# ── embed view (mlp / ft-transformer) ────────────────────────────────────────

def prepare_embed(X_train, X_test):
    num_cols, cat_cols = split_feature_types(X_train)
    stats = _fit_numeric(X_train, num_cols)

    Xnum_tr = _num_impute_scale(X_train, num_cols, stats)
    Xnum_te = _num_impute_scale(X_test, num_cols, stats)

    Xcat_tr = np.zeros((len(X_train), len(cat_cols)), dtype=np.int64)
    Xcat_te = np.zeros((len(X_test), len(cat_cols)), dtype=np.int64)
    cardinalities = []  # includes the reserved 0 = <unk>/missing index
    for j, c in enumerate(cat_cols):
        vc = X_train[c].value_counts(dropna=True)
        keep = list(vc.index[: config.MAX_VOCAB])
        vocab = {v: i + 1 for i, v in enumerate(keep)}  # 0 reserved for <unk>
        Xcat_tr[:, j] = X_train[c].map(vocab).fillna(0).to_numpy(dtype=np.int64)
        Xcat_te[:, j] = X_test[c].map(vocab).fillna(0).to_numpy(dtype=np.int64)
        cardinalities.append(len(keep) + 1)

    return (Xnum_tr, Xcat_tr), (Xnum_te, Xcat_te), cardinalities, (num_cols, cat_cols)
