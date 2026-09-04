#!/bin/bash
#
# TABULAR ARM, GPU half: the entity-embedding MLP and the FT-Transformer.
# 8 array tasks = 2 models x 4 outcomes, one GPU each.
#
# Partition, wall clock and GRES are deliberately NOT pinned here -- submit.sh
# supplies them on the sbatch command line, because the right choice depends on
# what is free at submission time:
#
#   a100   --gres=gpu:a100:1  is the only way to ask for gpu007; it is the sole
#          node with a100 GRES (8 of them) and is several times faster than the
#          gpu011-014 RTX 6000 branch nodes. Slurm has no soft node preference:
#          --nodelist is a hard constraint and --prefer only matches features,
#          which are identical across these nodes.
#   short  gpu_short caps a job at 2 hours but backfills far sooner when every
#          GPU is allocated. Safe here only because training checkpoints after
#          every epoch and resumes from the checkpoint -- see submit.sh's
#          chained submission.
#
# Cells that already have a DONE marker exit in seconds, so re-running this
# array is always safe.
#
#SBATCH --job-name=tabular_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-7
#SBATCH --output=/vast/projects/akumar-group/NSQIP/zihan/puf_models/logs/tabular_gpu_%A_%a.log

set -euo pipefail
cd /vast/projects/akumar-group/NSQIP/zihan/puf_models/tabular

source ./env.sh

PY="${TABULAR_PYTHON:-/vast/projects/fusheng-group/zding/miniconda3/envs/nsqip_env/bin/python}"

MODELS=(mlp ft_transformer)
LABELS=(mortality readmission unplanned_or any_compl)
MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID / 4))]}"
LABEL="${LABELS[$((SLURM_ARRAY_TASK_ID % 4))]}"

echo "the tabular arm GPU task $SLURM_ARRAY_TASK_ID -> $MODEL / $LABEL on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# TABULAR_FORCE=1 retrains a cell that already carries a DONE marker -- needed when
# a cell completed but produced a bad model (see the LightGBM early-stopping note
# in config_tabular.py), since the default is to skip anything already DONE.
"$PY" train.py --model "$MODEL" --label "$LABEL" ${TABULAR_FORCE:+--force}
