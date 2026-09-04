#!/bin/bash
#
# TABULAR ARM, GPU-free fallback for the neural nets.
#
# Same code, same seeds, same checkpointing; torch just runs on CPU (train.py
# picks the device from torch.cuda.is_available() and takes its thread count
# from the allocation).
#
# --array is supplied by submit.sh, and the default it passes is 0-3 -- the MLP
# only. That is deliberate, from measured timings on this data:
#
#   MLP              ~5 s/epoch per 57k rows on 8 threads  ->  roughly 7 min per
#                    epoch at 4.7M rows, a few hours per outcome. Fine on CPU.
#   FT-Transformer   ~7 s per 2048-row step on 8 threads   ->  roughly 4-5 HOURS
#                    per epoch at 4.7M rows, i.e. days per outcome even with
#                    early stopping. Not a fallback, a trap.
#
# So the FT-Transformer waits for a GPU; the chained gpu_short jobs
# (GPU_MODE=short) get one within a couple of hours even when the cluster is
# fully allocated, and an epoch there is minutes rather than hours. Override
# with `sbatch --array=0-7 run_cpu_torch.sbatch` only if you have decided the
# multi-day CPU run is worth it.
#
#SBATCH --job-name=tabular_torch_cpu
#SBATCH --partition=cpu_extra
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --array=0-7
#SBATCH --output=/vast/projects/akumar-group/NSQIP/zihan/puf_models/logs/tabular_torchcpu_%A_%a.log

set -euo pipefail
cd /vast/projects/akumar-group/NSQIP/zihan/puf_models/tabular

source ./env.sh
export CUDA_VISIBLE_DEVICES=""       # cpu_extra includes gpu011-014; do not grab one

PY="${TABULAR_PYTHON:-/vast/projects/fusheng-group/zding/miniconda3/envs/nsqip_env/bin/python}"

MODELS=(mlp ft_transformer)
LABELS=(mortality readmission unplanned_or any_compl)
MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID / 4))]}"
LABEL="${LABELS[$((SLURM_ARRAY_TASK_ID % 4))]}"

echo "the tabular arm CPU-torch task $SLURM_ARRAY_TASK_ID -> $MODEL / $LABEL on $(hostname)"
echo "  CPUS = ${SLURM_CPUS_PER_TASK:-?}  (no GPU by design)"

# TABULAR_FORCE=1 retrains a cell that already carries a DONE marker -- needed when
# a cell completed but produced a bad model (see the LightGBM early-stopping note
# in config_tabular.py), since the default is to skip anything already DONE.
"$PY" train.py --model "$MODEL" --label "$LABEL" ${TABULAR_FORCE:+--force}
