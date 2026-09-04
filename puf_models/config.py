"""
Shared configuration for the PUF model zoo.

Six model families are trained on the four binary outcomes prepared in
dataset/processed/puf/ (X_train/X_test + Y_train_*/Y_test_* + benchmark_*):

    logistic_regression, random_forest, xgboost, lightgbm, mlp, ft_transformer
        x  { any_compl, mortality, readmission, unplanned_or }   =  24 runs

The X_train/X_test parquets are already curated by
zihan/all_puf_split/build_dataset.py: identifiers, benchmark columns
(MORTPROB/MORBPROB) and every outcome-source column have already been removed,
so nothing here re-derives labels or worries about that class of leakage. We
just consume X as the predictor matrix.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
#
# Defaults reproduce experiment 1 exactly (X/Y/benchmark all side by side in
# dataset/processed/puf/, results in ./outputs). Later experiments point at a
# different feature build and a different output folder via env vars, so the
# runs never share a directory -- see submit2.sh.

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = "/vast/projects/akumar-group/NSQIP/dataset/processed/puf"
X_DIR = os.environ.get("PUF_X_DIR", DATA_DIR)          # X_{train,test}.parquet
Y_DIR = os.environ.get("PUF_Y_DIR", DATA_DIR)          # Y_{train,test}_<label>.parquet
BENCH_DIR = os.environ.get("PUF_BENCH_DIR", DATA_DIR)  # benchmark_{train,test}.parquet
OUT_DIR = os.environ.get("PUF_OUT_DIR", os.path.join(REPO_DIR, "outputs"))

# ── Task definition ──────────────────────────────────────────────────────────

LABELS = ["any_compl", "mortality", "readmission", "unplanned_or"]

# Which ACS-computed benchmark probability maps to which outcome (used only for
# a reference baseline in benchmark_baseline.py, never as a model input).
BENCHMARK_FOR = {
    "any_compl": "MORBPROB",
    "mortality": "MORTPROB",
}

SEED = 42

# ── Preprocessing knobs ──────────────────────────────────────────────────────

# A column of X is treated as categorical iff pandas typed it as string,
# otherwise numeric. (see data.split_feature_types)

# Logistic regression: categoricals with <= this many levels are one-hot
# encoded; higher-cardinality columns (free-text procedure fields, CPT text,
# etc.) are frequency-encoded into a single numeric column so the design matrix
# stays tractable at ~5M rows.
ONEHOT_MAX_CARD = 30

# Entity-embedding models (MLP, FT-Transformer): keep the top-K most frequent
# levels per categorical column; everything rarer + missing collapses to a
# single <unk> index (0). Caps embedding-table size for columns like
# OTHERPROC1 (~5.7k levels).
MAX_VOCAB = 1000

# ── Model hyperparameters ────────────────────────────────────────────────────

LR_PARAMS = dict(
    solver="saga",
    penalty="l2",
    C=1.0,
    max_iter=200,
    tol=1e-3,
    class_weight="balanced",
    n_jobs=-1,
)

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=50,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=SEED,
)

XGB_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    device="cuda",              # set to "cpu" if no GPU
    eval_metric="aucpr",
    early_stopping_rounds=50,
    random_state=SEED,
)

LGBM_PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=-1,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=100,
    reg_lambda=1.0,
    device_type="cpu",          # set to "gpu" if the lightgbm build supports it
    random_state=SEED,
    n_jobs=-1,
)

# Torch models (MLP, FT-Transformer) share this training config.
TORCH_TRAIN = dict(
    val_frac=0.05,
    batch_size=8192,
    max_epochs=30,
    patience=5,               # early-stop on val AUROC
    lr=1e-3,
    weight_decay=1e-5,
    num_workers=4,
)

MLP_PARAMS = dict(
    embed_dropout=0.1,
    hidden_dims=(512, 256, 128),
    hidden_dropout=0.3,
)

FT_PARAMS = dict(
    d_token=64,
    n_blocks=3,
    n_heads=8,
    attn_dropout=0.2,
    ffn_dropout=0.1,
    residual_dropout=0.0,
    ffn_factor=4 / 3,
    batch_size=2048,          # attention is heavier than the MLP -> smaller batch
)


def out_dir(model, label):
    d = os.path.join(OUT_DIR, model, label)
    os.makedirs(d, exist_ok=True)
    return d
