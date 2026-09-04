#!/bin/bash
#
# TEXT-ENCODER ARM, gates 1 and 2 (PLAN.md sec 9). Nothing long is queued until this
# job's log shows GATE 1: PASS and a throughput number.
#
# Small and short on purpose: one GPU, 20k rows. Its whole job is to answer
#   (a) does any chunk still exceed 512 tokens for any encoder, in any cohort;
#   (b) how many rows/s does each encoder actually do on this hardware,
# and (b) is what decides whether all three encoders are affordable or whether
# clinicalbert gets dropped.
#
#SBATCH --job-name=encoder_gate
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/vast/projects/akumar-group/NSQIP/zihan/puf_models/logs/encoder_gate_%j.log

set -euo pipefail
cd /vast/projects/akumar-group/NSQIP/zihan/puf_models/text_encoder
source ./env.sh

PY="${ENCODER_PYTHON:-/vast/projects/fusheng-group/zding/miniconda3/envs/mistral_env/bin/python}"

echo "host $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

echo "===== gate 1: zero truncation ====="
"$PY" gate_check.py --verify --n 20000

echo
echo "===== gate 2: throughput ====="
"$PY" gate_check.py --throughput --n 20000

echo
echo "===== gate 4: end-to-end dry run on SBUH (3,490 rows) ====="
# Exercises the real write path -- memmap allocation, shard row ranges, pooling,
# the truncation assertion -- on the smallest cohort, so a bug in embed.py
# surfaces here in a minute instead of 63 array tasks later.
#
# Writes into a THROWAWAY tree, not the real one. The dry run and array task
# (bio_clinicalbert, sbuh) would otherwise target the same .npy, and if both
# were ever queued at once they would race on memmap allocation.
export ENCODER_OUT_DIR="$ENCODER_OUT_DIR/_gate_dryrun"
mkdir -p "$ENCODER_OUT_DIR"
"$PY" embed.py --encoder bio_clinicalbert --cohort sbuh --split test \
               --shard 0 --n-shards 1 --force

"$PY" - <<'PYEOF'
import sys; sys.path.insert(0, '.')
import numpy as np, config_encoder
p = config_encoder.emb_path('bio_clinicalbert', 'sbuh', 'test')
a = np.load(p, mmap_mode='r')
norms = np.linalg.norm(a[:200].astype(np.float32), axis=1)
print(f"  {p}")
print(f"  shape {a.shape} dtype {a.dtype}")
print(f"  all-zero rows in first 200: {(np.abs(a[:200]).sum(1) == 0).sum()}")
print(f"  L2 norm mean {norms.mean():.4f} (expect ~1.0)")
assert a.shape == (3490, 768), a.shape
assert abs(norms.mean() - 1.0) < 1e-2, norms.mean()
print("  GATE 4: PASS")
PYEOF
