"""
Data loading, splitting and preprocessing for the tabular arm.

The preprocessing contract is fit-once/apply-everywhere. ../data.py's
`prepare_*(X_fit, X_other)` REFITS on its first argument every call and never
writes that fitted state anywhere, so this module uses the fit/apply pair from
feature_alignment/lib/prep_apply.py instead:

    state = fit_state(view, X_fit)      # once, on the fit split, and serialised
    rep   = apply_state(state, X_any)   # everywhere else -- fits nothing

The two produce bit-identical output to ../data.py for all four views (verified
against it on real 2018-2022 rows), so the tabular arm stays comparable to the earlier runs
on the preprocessing itself; what changes is that the state now survives the run
and SBUH can be scored without the PUF training pool.
"""

import os
import sys
import zlib

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import config_tabular  # noqa: E402

sys.path.insert(0, config_tabular.ALIGN_DIR)
from lib import prep_apply  # noqa: E402


# ── Loading ──────────────────────────────────────────────────────────────────

def model_features():
    """The 69 columns the models actually consume.

    Read from the file the data build wrote; never recomputed here. The 4
    columns it omits are 100% empty in the PUF training pool -- a training-set
    judgement, made once, upstream, with the audit table alongside it.
    """
    return pd.read_csv(config_tabular.FEATURE_LIST)["name"].tolist()


def load_X(cohort, split, columns=None):
    p = os.path.join(config_tabular.DATA_DIR, f"{cohort}_X_{split}.parquet")
    X = pd.read_parquet(p, columns=columns)
    return X if columns is None else X[columns]     # enforce contract order


def load_y(cohort, split, label):
    p = os.path.join(config_tabular.DATA_DIR, f"{cohort}_Y_{split}_{label}.parquet")
    return pd.read_parquet(p).iloc[:, 0].to_numpy().astype(np.float32)


def load_meta(cohort, split):
    return pd.read_parquet(
        os.path.join(config_tabular.DATA_DIR, f"{cohort}_meta_{split}.parquet"))


# ── fit / val split ──────────────────────────────────────────────────────────

def split_indices(y, label):
    """Stratified fit (95%) / val (5%) indices over the 2018-2022 training pool.

    Seeded per label so the split matches that outcome's prevalence, and is
    IDENTICAL for all six model families on that outcome. Each (model, label) is
    its own Slurm array task, so the seed has to be reproducible ACROSS
    PROCESSES: the per-label offset uses zlib.crc32, never the builtin hash().
    Python randomises string hashing per process, so hash() would hand the six
    tasks six different splits for the same outcome and silently destroy
    comparability.

    LogReg and Random Forest do not early-stop and never read the val rows, but
    those rows are excluded from their fit anyway, so all six models are trained
    on exactly the same patients.
    """
    idx = np.arange(len(y))
    seed = config_tabular.SEED + (zlib.crc32(label.encode()) % 10_000)
    fit_idx, val_idx = train_test_split(
        idx, test_size=config_tabular.VAL_FRAC, random_state=seed, stratify=y)
    return np.sort(fit_idx), np.sort(val_idx)


# ── Preprocessing state ──────────────────────────────────────────────────────

def fit_prep(view, X_fit):
    return prep_apply.fit_state(view, X_fit,
                                onehot_max_card=config_tabular.ONEHOT_MAX_CARD,
                                max_vocab=config_tabular.MAX_VOCAB)


def apply_prep(state, X):
    return prep_apply.apply_state(state, X)


def save_prep(state, path):
    """Serialise the fitted state, atomically.

    xgboost and lightgbm share the tree view for a label and run as separate
    array tasks, as do mlp and ft_transformer for embed; both can therefore be
    writing the same shared prep/<label>__<view>.joblib at the same moment. A
    plain joblib.dump would let one task read a half-written file. Write to a
    task-private temp name and rename -- rename is atomic within a filesystem,
    so a reader sees either the old file or the complete new one.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    joblib.dump(state, tmp, compress=3)
    os.replace(tmp, path)
    return path


def load_prep(path):
    return joblib.load(path)
