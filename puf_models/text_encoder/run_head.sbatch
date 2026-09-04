#!/bin/bash
#
# TEXT-ENCODER ARM, downstream heads. 3 array tasks = one per encoder, each looping
# over the four outcomes in a single process.
#
# Not 12 tasks. The embedding matrices are the same for all four outcomes, so
# twelve tasks would reload 12 GB twelve times AND ask the scheduler for twelve
# GPU allocations on a cluster where allocations are the scarce thing. Three
# tasks load once each and need three slots.
#
# Each (encoder, outcome) still writes its own DONE marker, so a task killed
# partway resumes at the outcome it stopped on rather than redoing the lot.
#
#SBATCH --job-name=encoder_head
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-2
#SBATCH --output=/vast/projects/akumar-group/NSQIP/zihan/puf_models/logs/encoder_head_%A_%a.log

set -euo pipefail
cd /vast/projects/akumar-group/NSQIP/zihan/puf_models/text_encoder
source ./env.sh

PY="${ENCODER_PYTHON:-/vast/projects/fusheng-group/zding/miniconda3/envs/mistral_env/bin/python}"

ENCODERS=(medembed_large clinicalbert bio_clinicalbert)
ENCODER="${ENCODERS[$SLURM_ARRAY_TASK_ID]}"

echo "the text-encoder arm head task $SLURM_ARRAY_TASK_ID -> $ENCODER (all 4 outcomes) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PY" train_head.py --encoder "$ENCODER" --label all ${ENCODER_FORCE:+--force}
