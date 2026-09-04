#!/bin/bash
#
# TABULAR ARM, CPU half: Logistic Regression, Random Forest, LightGBM, XGBoost.
# None of them needs a GPU at this scale, and the CPU partition is idle while
# every GPU on the cluster is allocated, so these start immediately.
#
# 16 array tasks = 4 models x 4 outcomes, one cell each. Splitting by outcome as
# well as by model is not just parallelism: the fit/val/calib split is seeded
# per outcome, so each cell refits its own preprocessing state anyway and there
# is nothing to share by keeping the four outcomes in one task.
#
#SBATCH --job-name=tabular_cpu
#SBATCH --partition=cpu_extra
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --array=0-15
#SBATCH --output=/vast/projects/akumar-group/NSQIP/zihan/puf_models/logs/tabular_cpu_%A_%a.log

set -euo pipefail
cd /vast/projects/akumar-group/NSQIP/zihan/puf_models/tabular

source ./env.sh
export TABULAR_XGB_DEVICE=cpu

PY="${TABULAR_PYTHON:-/vast/projects/fusheng-group/zding/miniconda3/envs/nsqip_env/bin/python}"

MODELS=(logistic_regression random_forest lightgbm xgboost)
LABELS=(mortality readmission unplanned_or any_compl)
MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID / 4))]}"
LABEL="${LABELS[$((SLURM_ARRAY_TASK_ID % 4))]}"

echo "the tabular arm CPU task $SLURM_ARRAY_TASK_ID -> $MODEL / $LABEL on $(hostname)"
echo "  DATA = $TABULAR_DATA_DIR"
echo "  OUT  = $TABULAR_OUT_DIR"
echo "  CPUS = ${SLURM_CPUS_PER_TASK:-?}"

# TABULAR_FORCE=1 retrains a cell that already carries a DONE marker -- needed when
# a cell completed but produced a bad model (see the LightGBM early-stopping note
# in config_tabular.py), since the default is to skip anything already DONE.
"$PY" train.py --model "$MODEL" --label "$LABEL" ${TABULAR_FORCE:+--force}
