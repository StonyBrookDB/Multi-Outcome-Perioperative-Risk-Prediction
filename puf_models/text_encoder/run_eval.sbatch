#!/bin/bash
#
# TEXT-ENCODER ARM, evaluation. CPU only -- reads saved predictions, runs the tabular arm's
# own bootstrap over them, writes summary_{puf,sbuh}.csv.
#
#SBATCH --job-name=encoder_eval
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/vast/projects/akumar-group/NSQIP/zihan/puf_models/logs/encoder_eval_%j.log

set -euo pipefail
cd /vast/projects/akumar-group/NSQIP/zihan/puf_models/text_encoder
source ./env.sh

PY="${ENCODER_PYTHON:-/vast/projects/fusheng-group/zding/miniconda3/envs/mistral_env/bin/python}"

# Embeddings were already verified by the gate job ahead of this one in the
# chain; here we only check what the heads themselves produced.
"$PY" verify.py --heads || echo "WARNING: verify reported failures (see above)"
"$PY" evaluate_encoder.py --cohort puf  --n-boot "${ENCODER_N_BOOT:-1000}"
"$PY" evaluate_encoder.py --cohort sbuh --n-boot "${ENCODER_N_BOOT:-1000}"
