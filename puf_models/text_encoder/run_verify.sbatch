#!/bin/bash
#
# TEXT-ENCODER ARM, verification gate. CPU only, minutes.
#
# Exists as its own job so it can sit between the embedding array and the head
# array as an --dependency=afterok barrier: verify.py exits non-zero when a
# check fails, and a chained head job then never starts. That is the whole
# point -- heads trained on embeddings with an uncovered shard would produce
# perfectly plausible numbers from partly-zero inputs.
#
#SBATCH --job-name=encoder_verify
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/vast/projects/akumar-group/NSQIP/zihan/puf_models/logs/encoder_verify_%j.log

set -euo pipefail
cd /vast/projects/akumar-group/NSQIP/zihan/puf_models/text_encoder
source ./env.sh

PY="${ENCODER_PYTHON:-/vast/projects/fusheng-group/zding/miniconda3/envs/mistral_env/bin/python}"
"$PY" verify.py --embeddings
