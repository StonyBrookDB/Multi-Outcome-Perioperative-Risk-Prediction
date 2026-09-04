"""
Experiment 6 configuration -- frozen-encoder embedding classifiers.

the text-encoder arm adds an LLM-embedding arm to the the tabular arm benchmark, under the tabular arm's protocol, so
its rows can share a table with the six tabular families without an asterisk.
See PLAN.md for the full rationale; the short version is that the collaborator's
earlier arm used a different validation fraction (0.15 vs 0.05), a different
split, focal loss instead of class-weighted BCE, and early stopping on val F1
instead of val AUROC. Nothing there is wrong on its own, but it means those rows
were not comparable with the tabular ones.

WHAT IS FIXED HERE AND WHY IT MUST NOT MOVE

  Everything in this file is prespecified. The manuscript states that
  hyperparameters were fixed before fitting rather than searched; the text-encoder arm inherits
  that claim, so no value here may be changed in response to an observed
  number. If a gate in PLAN.md sec 9 fails, fix the bug and re-freeze -- do not
  tune. The frozen state is recorded by config_sha256() and re-checked at the
  end of the run.

WHAT IS READ FROM the tabular arm, NOT REDERIVED

  the 73-column feature build      feature_73/
  the four outcome parquets        feature_73/{puf,sbuh}_Y_*
  the fit/val split                results/tabular/<model>/<label>/split_indices.parquet

  The split is per OUTCOME, not global. data_tabular.py:split_indices() stratifies on
  the outcome with seed SEED + crc32(label) % 10000, so the four outcomes get
  four different splits (val-set overlap ~12,700 of 249,784); for a given
  outcome all six families share one split. the text-encoder arm reads the tabular arm's saved file rather
  than calling train_test_split again, so a scikit-learn RNG change can never
  silently move the boundary.

  NOTE for anyone verifying this: compare the `row` column of that parquet, not
  the `split` column. `split` is just 'fit' x 4,745,886 followed by 'val' x
  249,784 and is byte-identical for every model and every outcome; comparing it
  will tell you the splits match when they do not.

EMBEDDINGS ARE LABEL-INDEPENDENT

  Each case is embedded ONCE per encoder and the matrix is reused for all four
  outcomes. Only the downstream head is per-outcome.
"""

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                       # zihan/puf_models
NSQIP = os.path.dirname(REPO)                      # zihan
sys.path.insert(0, REPO)

import config as _parent  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = os.environ.get(
    "ENCODER_DATA_DIR", "/vast/projects/akumar-group/NSQIP/zihan/feature_73")
OUT_DIR = os.environ.get("ENCODER_OUT_DIR", os.path.join(REPO, "results", "text_encoder"))
EMB_DIR = os.path.join(OUT_DIR, "embeddings")

# the tabular arm's outputs, read-only. Used for the per-outcome split and, at reporting
# time, for the tabular comparison rows.
TABULAR_OUT_DIR = os.environ.get(
    "ENCODER_TABULAR_OUT_DIR", os.path.join(REPO, "results", "tabular"))
# Any of the six families carries the same split for a given label; mlp is used
# by convention. Asserted against a second family in verify.py.
SPLIT_SOURCE_MODEL = "mlp"

FEATURE_LIST = os.path.join(DATA_DIR, "features", "features_model_69.csv")

# ── Cohorts (identical to the tabular arm) ──────────────────────────────────────────────

N_ROWS = {
    ("puf", "train"): 4_995_670,       # 2018-2022, development
    ("puf", "test"): 963_565,          # 2024, temporal test
    ("sbuh", "test"): 3_490,           # 2024-2025, single institution
}
COHORTS = [("puf", "train"), ("puf", "test"), ("sbuh", "test")]
LABELS = ["mortality", "readmission", "unplanned_or", "any_compl"]
SEED = _parent.SEED

# ── Serialization ────────────────────────────────────────────────────────────
#
# 73 fields, always all of them, in the fixed column order of feature_73.
# Missing values fill the slot with MISSING_TOKEN rather than being omitted:
# omitting them would make text LENGTH encode missingness, and the two cohorts
# differ systematically in lab completeness (SBUH 53.2 fields present per case
# vs PUF 51.2; PRPTT 45.6% vs 27.0%), so an omit-missing template would hand
# the two cohorts different length distributions for a reason that has nothing
# to do with the patient.

MISSING_TOKEN = "undefined"

# DPRHEMOGLOBIN, DPRHEMO_A1C, PRPT and DPRPT are 100% empty across 2018-2022,
# which is why the tabular arm instantiated only 69 of the 73 as model inputs. Two of them
# are NOT empty later: DPRHEMOGLOBIN is 83.4% populated in PUF 2024 and 93.4%
# at the single site, DPRHEMO_A1C 30.3% and 35.0%. Serialising them as recorded
# would hand the LLM arm information at test time that the tabular arm
# structurally cannot see, and would put real values in slots the head only
# ever saw as `undefined` while training. Masking them in EVERY cohort keeps
# both arms on the same information at zero cost.
#
# Set False to reproduce the collaborator's per-cohort behaviour instead; doing
# so breaks arm-to-arm comparability and must be disclosed if reported.
MASK_TRAINING_EMPTY = True
TRAINING_EMPTY_COLS = ["DPRHEMOGLOBIN", "DPRHEMO_A1C", "PRPT", "DPRPT"]

# ── Encoders ─────────────────────────────────────────────────────────────────
#
# Frozen feature extractors. No encoder weight is updated at any point.
#
# Pooling follows each model card and is prespecified per encoder, never chosen
# by performance. MedEmbed-large-v0.1 is a BGE-large derivative trained AS an
# embedding model and pools on CLS. The two BERTs are masked-LM checkpoints
# whose CLS token was never trained as a sentence representation, so mean
# pooling over the attention-masked last hidden state is the correct reading of
# them -- using CLS there would measure the wrong thing and understate them.
#
# revision is pinned per encoder before the run: HuggingFace authors update
# weights in place, so a bare model name is not a reproducible reference.
# fetch_models.py resolves and writes these; a null revision means "not yet
# pinned" and embed.py refuses to run.

ENCODERS = {
    "medembed_large": dict(
        hf_id="abhinand/MedEmbed-large-v0.1",
        revision="963121bfb9c625475f65b08fb54990ce9c4e7a1a",
        dim=1024,
        pooling="cls",
    ),
    "clinicalbert": dict(
        hf_id="medicalai/ClinicalBERT",
        revision="f7c7f65227cb311f33a79c24858d875876d478ac",
        dim=768,
        pooling="mean",
    ),
    "bio_clinicalbert": dict(
        hf_id="emilyalsentzer/Bio_ClinicalBERT",
        revision="d5892b39a4adaed74b92212a44081509db72f87b",
        dim=768,
        pooling="mean",
    ),
}

MAX_LEN = 512
EMB_DTYPE = "float16"
EMBED_BATCH = 256           # chunks per forward pass; length-sorted within shard

# ── Chunking ─────────────────────────────────────────────────────────────────
#
# All 73 fields do not fit. Measured 2026-09-01 on 2,000 PUF 2024 cases, the
# serialised sentence is 1,029 tokens for MedEmbed-large, 1,038 for
# ClinicalBERT and 1,073 for Bio_ClinicalBERT, against a 512-position limit in
# all three: 100% of cases overflow.
#
# This is not a new problem, it is an unnoticed one. The collaborator's
# embedder resolved max_length to min(model_max, 512) and called the tokenizer
# with truncation=True, so every case in that run was cut at 512 -- the first
# 40 of the 73 fields went in and the remaining 33 (sex, race, smoking status,
# surgical specialty, weight, work RVU, case acuity, emergency flag, dyspnea,
# preoperative renal failure, open wound, weight loss, ...) never reached the
# encoder at all. That is the most likely explanation for that arm's lower
# discrimination, and it is a third respect in which those rows were not what
# the manuscript said they were.
#
# The fix keeps every field: partition the 73 into CONTIGUOUS groups, render
# each group as its own standalone sentence, embed each, and mean-pool. No
# field is ever split across a boundary, and no field is ever dropped.
#
# The partition is FIXED -- the same field grouping for every patient, every
# cohort and every encoder -- so chunk composition is never patient-dependent.
# It is computed once by chunking.build_plan() from worst-case per-field token
# cost measured across all three tokenizers, then frozen to chunk_plan.json and
# folded into config_sha256(). It is a function of the template and the
# tokenizers only; no outcome is involved at any point.
CHUNK_BUDGET = 480          # token ceiling per chunk, leaving MAX_LEN - 480 = 32
                            # for [CLS]/[SEP] plus a margin for rows longer than
                            # the calibration sample
CHUNK_PLAN_FILE = os.path.join(HERE, "chunk_plan.json")

# Each chunk's vector is L2-normalized, the chunks are averaged, and the mean is
# L2-normalized again. Normalizing before the mean stops a single long chunk
# (the one holding PRNCPTX free text) from dominating by magnitude; normalizing
# after keeps every stored row on the unit sphere, which is what the downstream
# head and the linear probe both expect.
L2_NORMALIZE = True
CHUNK_POOL = "mean"

# ── Downstream head: the tabular arm's MLP, verbatim ────────────────────────────────────
#
# Only the input representation differs from the tabular arm's `mlp` cell. That is the
# whole point of the arm: MLP-on-embedding vs MLP-on-tabular is a
# single-variable contrast, which is what lets the rows share a table.
#
# Deliberately NOT focal loss and NOT a balanced sampler. Class re-weighting is
# already applied through pos_weight, exactly as in the tabular arm; focal loss would add
# two more free parameters and distort the score scale a second time, on top of
# a re-weighting the manuscript already has to disclose.
#
# Early stopping on val AUROC, not val F1: F1 is threshold-dependent and is not
# a reported metric, so stopping on it would select the model by a number the
# paper never shows.

HEAD_PARAMS = dict(
    hidden_dims=_parent.MLP_PARAMS["hidden_dims"],        # (512, 256, 128)
    hidden_dropout=_parent.MLP_PARAMS["hidden_dropout"],  # 0.3
)
HEAD_TRAIN = dict(
    batch_size=_parent.TORCH_TRAIN["batch_size"],         # 8192
    max_epochs=_parent.TORCH_TRAIN["max_epochs"],         # 30
    patience=_parent.TORCH_TRAIN["patience"],             # 5, on val AUROC
    lr=_parent.TORCH_TRAIN["lr"],                         # 1e-3
    weight_decay=_parent.TORCH_TRAIN["weight_decay"],     # 1e-5
)

# Secondary, reported in text only, never in the main table. A linear probe
# separates "the representation carries less information" from "the head was
# not strong enough", which is the first question a reviewer asks.
LINEAR_PROBE = dict(max_iter=200, tol=1e-3, C=1.0)

# ── Evaluation (identical to the tabular arm) ───────────────────────────────────────────

N_BOOT = 1000
PRED_FILE = {"puf": "test_pred.parquet", "sbuh": "sbuh_pred.parquet"}


# ── Layout helpers ───────────────────────────────────────────────────────────

def emb_path(encoder, cohort, split):
    d = os.path.join(EMB_DIR, encoder)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{cohort}_{split}.npy")


def manifest_path(encoder):
    d = os.path.join(EMB_DIR, encoder)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "manifest.json")


def out_dir(encoder, label, root=None):
    d = os.path.join(root or OUT_DIR, encoder, label)
    os.makedirs(d, exist_ok=True)
    return d


def split_path(label):
    """the tabular arm's saved fit/val split for one outcome. Read, never rederived."""
    return os.path.join(TABULAR_OUT_DIR, SPLIT_SOURCE_MODEL, label,
                        "split_indices.parquet")


def config_sha256():
    """Hash of this file plus the label dictionary.

    Recorded in each manifest at generation time and re-checked when the run
    finishes, so "the configuration was frozen before the run" is a verifiable
    statement rather than an assurance.
    """
    h = hashlib.sha256()
    for p in (os.path.join(HERE, "config_encoder.py"),
              os.path.join(HERE, "labels73.py"),
              CHUNK_PLAN_FILE):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} does not exist yet; run gate_check.py --calibrate first")
        with open(p, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()
