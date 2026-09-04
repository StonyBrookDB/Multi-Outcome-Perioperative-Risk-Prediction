#!/usr/bin/env bash
#
# Experiment 6 environment.
#
# the text-encoder arm only READS feature_73/ and results/tabular/. Nothing here rebuilds,
# re-splits or rewrites anything belonging to an earlier experiment.

REPO=/vast/projects/akumar-group/NSQIP/zihan/puf_models

export ENCODER_DATA_DIR="${ENCODER_DATA_DIR:-/vast/projects/akumar-group/NSQIP/zihan/feature_73}"
export ENCODER_OUT_DIR="${ENCODER_OUT_DIR:-$REPO/results/text_encoder}"
export ENCODER_TABULAR_OUT_DIR="${ENCODER_TABULAR_OUT_DIR:-$REPO/results/tabular}"

# Shared model cache. Populated on the LOGIN node by fetch_models.py; compute
# nodes have no route to huggingface.co, and OFFLINE=1 turns a would-be silent
# download-and-hang into an immediate error.
export HF_HOME="${HF_HOME:-/vast/projects/akumar-group/NSQIP/zihan/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

for f in puf_X_train.parquet puf_X_test.parquet sbuh_X_test.parquet; do
  if [ ! -s "$ENCODER_DATA_DIR/$f" ]; then
    echo "the text-encoder arm: missing $ENCODER_DATA_DIR/$f" >&2
    exit 1
  fi
done
if [ ! -s "$ENCODER_TABULAR_OUT_DIR/mlp/mortality/split_indices.parquet" ]; then
  echo "the text-encoder arm: missing the tabular arm split at $ENCODER_TABULAR_OUT_DIR" >&2
  exit 1
fi

mkdir -p "$ENCODER_OUT_DIR" "$REPO/logs"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
