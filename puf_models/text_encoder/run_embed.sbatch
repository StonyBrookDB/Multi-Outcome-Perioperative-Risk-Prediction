#!/bin/bash
#
# TEXT-ENCODER ARM, embedding generation.
#
# One array task = one (encoder, cohort, split, shard). Shards write disjoint
# row ranges into a PREALLOCATED .npy, so tasks are order-independent, safe to
# run in parallel, and safe to re-run individually. A task whose DONE marker
# exists exits in seconds, so re-submitting the whole array is always safe.
#
# Array size = 3 encoders x (TRAIN_SHARDS + TEST_SHARDS + 1 sbuh shard).
# With TRAIN_SHARDS=16 TEST_SHARDS=4 that is 3 x 21 = 63 tasks, 0-62.
#
# Partition/GRES are supplied by submit.sh on the command line, because the
# right choice depends on what is free: --gres=gpu:a100:1 is the only way to
# reach the 8-GPU a100 node, which is several times faster than the RTX 6000
# branch nodes.
#
#SBATCH --job-name=encoder_embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-62
#SBATCH --output=/vast/projects/akumar-group/NSQIP/zihan/puf_models/logs/encoder_embed_%A_%a.log

set -euo pipefail
cd /vast/projects/akumar-group/NSQIP/zihan/puf_models/text_encoder
source ./env.sh

PY="${ENCODER_PYTHON:-/vast/projects/fusheng-group/zding/miniconda3/envs/mistral_env/bin/python}"

TRAIN_SHARDS="${ENCODER_TRAIN_SHARDS:-16}"
TEST_SHARDS="${ENCODER_TEST_SHARDS:-4}"
PER_ENCODER=$((TRAIN_SHARDS + TEST_SHARDS + 1))

ENCODERS=(medembed_large clinicalbert bio_clinicalbert)
E=$((SLURM_ARRAY_TASK_ID / PER_ENCODER))
K=$((SLURM_ARRAY_TASK_ID % PER_ENCODER))
ENCODER="${ENCODERS[$E]}"

if   [ "$K" -lt "$TRAIN_SHARDS" ]; then
  COHORT=puf;  SPLIT=train; SHARD=$K;                          N=$TRAIN_SHARDS
elif [ "$K" -lt $((TRAIN_SHARDS + TEST_SHARDS)) ]; then
  COHORT=puf;  SPLIT=test;  SHARD=$((K - TRAIN_SHARDS));       N=$TEST_SHARDS
else
  COHORT=sbuh; SPLIT=test;  SHARD=0;                           N=1
fi

echo "the text-encoder arm task $SLURM_ARRAY_TASK_ID -> $ENCODER $COHORT/$SPLIT shard $SHARD of $N on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PY" embed.py --encoder "$ENCODER" --cohort "$COHORT" --split "$SPLIT" \
               --shard "$SHARD" --n-shards "$N" ${ENCODER_FORCE:+--force}
