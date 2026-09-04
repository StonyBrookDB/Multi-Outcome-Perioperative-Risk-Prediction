#!/bin/bash
#
# TABULAR ARM post-processing. Reads saved models and saved predictions only:
# CPU, no retraining, no GPU.
#
#   verify.py          the hard requirements (model object, preprocessing state,
#                      disjoint fit/val split, shared split across the six
#                      families) before anything downstream trusts the run
#   feature_report.py  DYSPNEA / WNDINF / WTLOSS and the other availability
#                      shifts, written out as evidence rather than patched away
#   external.py        SBUH 2024-2025 scored by the FROZEN model + FROZEN
#                      preprocessing state. Fits nothing.
#   evaluate.py x2     summary_puf.csv and summary_sbuh.csv, both with bootstrap
#                      CIs on every number
#   verify.py --strict nothing silently missing at the end
#
# There is no calibration step: the tabular arm reports rank metrics and rank-based
# operating points only.
#
#SBATCH --job-name=tabular_post
#SBATCH --partition=cpu_extra
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=/vast/projects/akumar-group/NSQIP/zihan/puf_models/logs/tabular_post_%j.log

set -euo pipefail
cd /vast/projects/akumar-group/NSQIP/zihan/puf_models/tabular

source ./env.sh
export CUDA_VISIBLE_DEVICES=""
export BOOT_N_JOBS="${SLURM_CPUS_PER_TASK:-32}"

PY="${TABULAR_PYTHON:-/vast/projects/fusheng-group/zding/miniconda3/envs/nsqip_env/bin/python}"
NB="${N_BOOT:-1000}"

"$PY" verify.py
"$PY" feature_report.py
"$PY" external.py
"$PY" evaluate.py --cohort puf  --n-boot "$NB"
"$PY" evaluate.py --cohort sbuh --n-boot "$NB"
"$PY" verify.py --strict

echo
echo "results:"
echo "  $TABULAR_OUT_DIR/summary_puf.csv"
echo "  $TABULAR_OUT_DIR/summary_sbuh.csv"
echo "  $TABULAR_OUT_DIR/summary_puf_ranking.csv   (wide AUROC/AUPRC pivot)"
echo "  $TABULAR_OUT_DIR/summary_sbuh_ranking.csv"
echo "  $TABULAR_OUT_DIR/feature_availability.md"
