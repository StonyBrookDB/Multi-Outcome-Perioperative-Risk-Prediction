"""
Experiment 5 configuration.

the tabular arm trains the six model families on the 73-column clinician-signed preop
feature set (zihan/feature_73/) and reports two tables: the PUF 2024 test year,
and the SAME frozen models scored on SBUH 2024-2025. Nothing else is compared.

Two requirements shape the layout:

  1. every model object is serialised, so a frozen model can be scored on a new
     cohort without refitting;
  2. every fitted PREPROCESSING state is serialised alongside it, so scoring a
     new site does not mean dragging the multi-million-row fit split along and
     refitting the medians, vocabularies and one-hot layout.

So: every model object is serialised, every preprocessing state is serialised,
and the feature set is exactly what the clinicians signed. Without the first two
the downstream zihan/feature_alignment/ scoring path cannot consume this bundle
and the experiment is worthless again.

Feature policy: the data files carry all 73 signed columns. Model input is the
69 in features/features_model_69.csv -- the 4 omitted (DPRHEMOGLOBIN,
DPRHEMO_A1C, PRPT, DPRPT) are 100% empty in the PUF training pool, so there is
nothing to learn from them and XGBoost raises on the empty category array. That
judgement used the training set only. Nothing here drops a feature for any other
reason: removals go through the Decision column of puf_feature_list_AKedit.xlsx,
never through this file.

NO PROBABILITY CALIBRATION. All six families train with class re-weighting
(class_weight / scale_pos_weight / BCE pos_weight) to cope with rare outcomes,
which deliberately distorts the output scale. The saved scores rank patients;
they are NOT risks. Brier, log-loss, ECE and calibration slope are therefore not
computed and not reported -- on a re-weighted raw scale they would mean nothing.
For the same reason there are no fixed risk thresholds ("flag if risk >= 5%"):
every operating point here is rank-based.

Model hyperparameters are inherited verbatim from ../config.py so the tabular arm stays
comparable to the earlier runs on everything except the data and the plumbing.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                       # zihan/puf_models
NSQIP = os.path.dirname(REPO)                      # zihan
sys.path.insert(0, REPO)

import config as _parent  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = os.environ.get(
    "TABULAR_DATA_DIR", "/vast/projects/akumar-group/NSQIP/zihan/feature_73")
OUT_DIR = os.environ.get("TABULAR_OUT_DIR", os.path.join(REPO, "results", "tabular"))

# prep_apply.py lives in the alignment repo and is the single implementation of
# the apply-only transforms. the tabular arm imports it rather than vendoring a copy, so
# the state written here and the state read by feature_alignment/score.py can
# never drift apart.
ALIGN_DIR = os.environ.get(
    "TABULAR_ALIGN_DIR", os.path.join(NSQIP, "feature_alignment"))

FEATURE_LIST = os.path.join(DATA_DIR, "features", "features_model_69.csv")

# ── Cohorts ──────────────────────────────────────────────────────────────────
#
# Split rule, fixed upstream and NOT re-derived here: operative year <=2022 is
# train, >=2024 is test, all of 2023 is left out of both sides.
#
#   puf_X_train  4,995,670 x 73   2018-2022     model fitting
#   puf_X_test     963,565 x 73   2024          internal test
#   sbuh_X_test      3,490 x 73   2024-2025     external validation, never fit
#
# sbuh_X_train exists but is deliberately unused: SBUH is external validation.
LABELS = ["mortality", "readmission", "unplanned_or", "any_compl"]
EXT_COHORT = "sbuh"
SEED = _parent.SEED

# ── Two-way split of the 2018-2022 training pool ────────────────────────────
#
# fit 95% / val 5%, stratified on the outcome.
#
# val exists for early stopping (GBDTs, neural nets). LogReg and Random Forest
# never look at it -- but those rows are still excluded from their fit, so all
# six families train on the IDENTICAL rows and the model-to-model comparison is
# about the models and nothing else.
FIT_FRAC = 0.95
VAL_FRAC = 0.05
assert abs(FIT_FRAC + VAL_FRAC - 1.0) < 1e-9

# ── Preprocessing knobs (inherited) ─────────────────────────────────────────

ONEHOT_MAX_CARD = _parent.ONEHOT_MAX_CARD          # 30
MAX_VOCAB = _parent.MAX_VOCAB                      # 1000

VIEW_FOR = {
    "logistic_regression": "linear",
    "random_forest": "matrix",
    "xgboost": "tree",
    "lightgbm": "tree",
    "mlp": "embed",
    "ft_transformer": "embed",
}
MODELS = list(VIEW_FOR)

ARTIFACT_FOR = {
    "logistic_regression": "model/model.joblib",
    "random_forest": "model/model.joblib",
    "xgboost": "model/xgboost.json",
    "lightgbm": "model/lightgbm.txt",
    "mlp": "model/state_dict.pt",
    "ft_transformer": "model/state_dict.pt",
}

# ── Model hyperparameters (inherited verbatim, threading made cgroup-aware) ──
#
# n_jobs=-1 asks sklearn/LightGBM for os.cpu_count() threads, which is the
# NODE's core count, not the cores Slurm actually granted this task. On a 64-core
# node running four array tasks that is a 4x oversubscription and every task
# slows down. Take the allocation instead.

def _n_threads():
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(n) if n else (os.cpu_count() or 1)


N_THREADS = _n_threads()

# LogisticRegression takes neither of these any more under sklearn 1.8: n_jobs
# is a documented no-op for a binary saga fit, and penalty="l2" is deprecated in
# favour of the (identical) default. Both only produced FutureWarnings; dropping
# them changes no number.
LR_PARAMS = {k: v for k, v in _parent.LR_PARAMS.items()
             if k not in ("n_jobs", "penalty")}
RF_PARAMS = dict(_parent.RF_PARAMS, n_jobs=N_THREADS)
# metric= is NOT cosmetic. Without it LightGBM also tracks the binary objective's
# default metric, binary_logloss, alongside the average_precision passed to fit().
# The early_stopping callback then fires on whichever tracked metric stalls
# first -- and because every model here is trained with scale_pos_weight (7.9 for
# any_compl), validation logloss is at its best on iteration 1 and only degrades
# from there. The observed result was best_iteration = 1, i.e. a single tree
# standing in for a 1500-tree GBM. Naming the metric explicitly keeps logloss out
# of the early-stopping decision entirely; train.py additionally passes
# first_metric_only=True as a second line of defence.
LGBM_PARAMS = dict(_parent.LGBM_PARAMS, n_jobs=N_THREADS,
                   metric="average_precision")

XGB_PARAMS = dict(_parent.XGB_PARAMS)
XGB_PARAMS["device"] = os.environ.get("TABULAR_XGB_DEVICE", "cpu")
XGB_PARAMS["n_jobs"] = N_THREADS

TORCH_TRAIN = dict(_parent.TORCH_TRAIN)
MLP_PARAMS = _parent.MLP_PARAMS
FT_PARAMS = _parent.FT_PARAMS

# ── Operating points (rank-based only) ──────────────────────────────────────
#
# TOP_K_PERCENT   "flag the k% highest-scoring patients". A capacity rule: it
#                 needs no calibration, and it is the same rule in both cohorts,
#                 which is what makes PUF and SBUH comparable.
# F1 threshold    two variants, because they answer different questions:
#                   f1_crossfit  threshold learned on the half of the cohort
#                                that excludes each patient -- never selects and
#                                reports on the same rows. PUF only; running it
#                                on SBUH would be fitting a threshold on SBUH.
#                   f1_val       one threshold, fitted once on the PUF
#                                VALIDATION split, then frozen and applied
#                                unchanged to PUF test AND to SBUH. This is the
#                                transferable operating point, and the only
#                                honest way to put a fixed cut on 3,490 external
#                                cases.
TOP_K_PERCENT = [1.0, 2.0, 5.0, 10.0]

# 1000 is the default; the brief allows up to 2000 (N_BOOT=2000 in run5_post).
N_BOOT = 1000

PRED_FILE = {"puf": "test_pred.parquet", "sbuh": "sbuh_pred.parquet"}


def out_dir(model, label, root=None):
    d = os.path.join(root or OUT_DIR, model, label)
    os.makedirs(d, exist_ok=True)
    return d


def prep_path(label, view, root=None):
    """Where the fitted preprocessing state for (label, view) is archived.

    Layout matches feature_alignment/contracts/<name>/prep/, so a bundle
    pointing at the tabular arm consumes these directly instead of refitting them.
    """
    d = os.path.join(root or OUT_DIR, "prep")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{label}__{view}.joblib")
