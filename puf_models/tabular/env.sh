#!/usr/bin/env bash
#
# Experiment 5 environment.
#
# Data is the clinician-signed 73-column build in zihan/feature_73/, already
# split (train <=2022, test >=2024, all of 2023 left out). Nothing here rebuilds
# or re-splits it -- the tabular arm only reads.
#
# This does not repoint the shared config.py: the tabular arm has its own
# config_tabular.py and its own entry points, so nothing outside this directory
# can be touched by anything run from it.

REPO=/vast/projects/akumar-group/NSQIP/zihan/puf_models

export TABULAR_DATA_DIR="${TABULAR_DATA_DIR:-/vast/projects/akumar-group/NSQIP/zihan/feature_73}"
export TABULAR_OUT_DIR="${TABULAR_OUT_DIR:-$REPO/results/tabular}"
# (no separate external output root: sbuh_pred.parquet lands in each cell dir)
export TABULAR_ALIGN_DIR="${TABULAR_ALIGN_DIR:-/vast/projects/akumar-group/NSQIP/zihan/feature_alignment}"

for f in puf_X_train.parquet puf_X_test.parquet sbuh_X_test.parquet \
         features/features_model_69.csv; do
  if [ ! -s "$TABULAR_DATA_DIR/$f" ]; then
    echo "the tabular arm: missing $TABULAR_DATA_DIR/$f" >&2
    exit 1
  fi
done
if [ ! -s "$TABULAR_ALIGN_DIR/lib/prep_apply.py" ]; then
  echo "the tabular arm: missing $TABULAR_ALIGN_DIR/lib/prep_apply.py (apply-only preprocessing)" >&2
  exit 1
fi

mkdir -p "$TABULAR_OUT_DIR" "$REPO/logs"

# Keep BLAS from spawning a thread per physical core inside every array task;
# the model libraries are told the allocation separately (config_tabular.N_THREADS).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
