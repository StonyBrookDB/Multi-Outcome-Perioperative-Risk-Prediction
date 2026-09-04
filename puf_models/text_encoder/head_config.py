"""
Head-stage configuration, frozen separately from the embedding stage.

WHY THIS IS NOT IN config_encoder.py

  config_encoder.config_sha256() hashes config_encoder.py + labels73.py + chunk_plan.json, and
  every embedding shard records that hash when it completes. Editing config_encoder.py
  while the embedding array is in flight would leave shards finished before the
  edit carrying one hash and shards finished after it carrying another, and
  verify.py would then report every early shard as stale -- a false alarm that
  looks exactly like real corruption.

  The two stages also freeze at different times: the embedding config had to be
  final before the array was submitted, whereas the head config only has to be
  final before the heads run. Splitting them makes that explicit instead of
  pretending one freeze point covers both.

  head_sha256() is recorded in each train_meta.json the same way.
"""

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config_encoder  # noqa: E402

# ── Input standardization ────────────────────────────────────────────────────
#
# The stored embeddings are L2-normalized, so each of the 1024 (or 768)
# components sits around 1/sqrt(d) ~ 0.03. PyTorch initialises a Linear layer
# uniformly on +/- 1/sqrt(d_in), also ~0.03, so the first pre-activation has a
# standard deviation near 0.03 -- roughly thirty times smaller than what the tabular arm's
# MLP sees, because the tabular arm standardizes its numeric predictors to mean 0 and unit
# variance before they reach the network. Left alone, the head would start with
# vanishing gradients and could easily fail to train inside 30 epochs, and the
# LLM arm would then look weak for a reason that has nothing to do with the
# representation.
#
# So: standardize per dimension, fitted on the FIT SPLIT ONLY and applied
# unchanged to val, the PUF test year and the single-institution cohort. This is
# not a new modelling choice -- it is the tabular arm's own preprocessing policy for
# numeric model inputs, applied to the numeric inputs the text-encoder arm happens to have. It
# is decided on that basis, not by comparing the AUROC of the two options.
#
# (The collaborator's head used batchnorm on the input, which addresses the same
# problem a different way.)
STANDARDIZE_INPUT = True

# Where the fitted (mean, scale) is archived, one per encoder, so scoring a new
# cohort later never needs the training pool. Mirrors the tabular arm's prep/ layout.
def scaler_path(encoder, label, root=None):
    d = os.path.join(root or config_encoder.OUT_DIR, "prep")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{encoder}__{label}.npz")


# ── Memory policy ────────────────────────────────────────────────────────────
#
# Load the embedding matrices into RAM rather than indexing the memmap per
# batch. Measured on this filesystem: a random 8,192-row gather out of the
# 10 GB memmap takes ~2.4 s, which is 23 min per epoch and 11.6 h for one
# (encoder, outcome) cell -- every head task would hit its wall clock. The same
# rows read sequentially into RAM cost seconds, and the fit split is only
# 9.7 GB in fp16 against the 96 GB the head job requests.
LOAD_TO_RAM = True

# fp16 in RAM, cast to fp32 per batch. Halves the resident set at no cost:
# the matrices were written in fp16, so nothing is lost by keeping them so.
RAM_DTYPE = "float16"


def head_sha256():
    with open(os.path.join(HERE, "head_config.py"), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()
