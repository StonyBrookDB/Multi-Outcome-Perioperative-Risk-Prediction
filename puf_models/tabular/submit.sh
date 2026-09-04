#!/usr/bin/env bash
#
# TABULAR ARM one-shot submit. Run this FROM A LOGIN NODE -- sbatch is not on
# the compute nodes' PATH (it lives in /cm/shared/apps/slurm/current/bin).
#
#     bash submit.sh                  # pick the GPU strategy automatically
#     GPU_MODE=a100  bash submit.sh   # queue for gpu007's A100s, however long
#     GPU_MODE=short bash submit.sh   # chained 2h gpu_short jobs, any GPU
#     GPU_MODE=any   bash submit.sh   # gpu_extra, any GPU, 10h
#     GPU_MODE=cpu   bash submit.sh   # MLP on CPU, FT-Transformer on gpu_short
#     GPU_MODE=none  bash submit.sh   # CPU models + post only, no MLP/FT
#     CHAIN=8 GPU_MODE=short bash submit.sh    # longer short-job chain
#
# Three pieces:
#   1. run_cpu.sbatch    LR / RF / LightGBM / XGBoost, 16 tasks (4 models x 4
#                         outcomes). The CPU partition is idle, so these start
#                         now instead of waiting behind a GPU.
#   2. the GPU half       MLP / FT-Transformer, 8 tasks. Strategy below.
#   3. run_post.sbatch   verify -> feature report -> SBUH inference ->
#                         evaluate both cohorts -> verify --strict.
#
# CHOOSING THE GPU STRATEGY
# -------------------------
# Slurm cannot express "prefer node X, else anything": --nodelist is a hard
# constraint and --prefer matches features, which are identical on gpu007 and
# gpu011-014. The only thing that distinguishes gpu007 is its GRES type -- it is
# the sole node with gpu:a100 (8 of them) and is several times faster than the
# RTX 6000 branch nodes. So --gres=gpu:a100:1 IS the way to pin to gpu007, and
# the choice has to be made at submission time.
#
# When every GPU on the cluster is allocated, that pin can mean a long wait. The
# alternative is gpu_short: a 2-hour cap, but it backfills far sooner. That is
# only safe because train.py checkpoints after every epoch and resumes from the
# checkpoint, and skips cells that already carry a DONE marker -- so N chained
# 2-hour jobs behave like one 2N-hour job. `auto` picks the pin when an A100 is
# actually free and the chain when nothing is.
#
# Only ONE GPU strategy is submitted. Do not run two at once: two jobs training
# the same cell would write the same checkpoint file.
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$PATH:/cm/shared/apps/slurm/current/bin"
command -v sbatch >/dev/null || { echo "sbatch not found -- submit from a login node" >&2; exit 1; }

source ./env.sh
mkdir -p ../logs

N_GPU_TASKS=8
CHAIN="${CHAIN:-6}"

free_a100() {
  # CfgTRES is the configured count, AllocTRES the allocated one; the difference
  # is what a new job could actually be given right now.
  local cfg alloc
  cfg=$(scontrol show node gpu007 2>/dev/null \
        | grep -oP 'CfgTRES=\S*gres/gpu:a100=\K[0-9]+' || true)
  alloc=$(scontrol show node gpu007 2>/dev/null \
          | grep -oP 'AllocTRES=\S*gres/gpu:a100=\K[0-9]+' || true)
  echo $(( ${cfg:-0} - ${alloc:-0} ))
}

MODE="${GPU_MODE:-auto}"
if [ "$MODE" = "auto" ]; then
  n=$(free_a100)
  if [ "$n" -ge 1 ]; then
    MODE=a100
    echo "gpu007 has $n A100(s) free -> pinning there (GPU_MODE=a100)"
  else
    MODE=short
    echo "gpu007 has 0 A100s free -> chained gpu_short backfill (GPU_MODE=short)."
    echo "  GPU_MODE=a100 instead if you would rather queue for an A100,"
    echo "  GPU_MODE=cpu  if you would rather not wait for a GPU at all."
  fi
fi

# ── 1. CPU models ────────────────────────────────────────────────────────────
cpu_id=$(sbatch --parsable run_cpu.sbatch)
echo "CPU array   : job $cpu_id   LR / RF / LightGBM / XGBoost, 16 tasks"

# ── 2. neural nets ───────────────────────────────────────────────────────────
DEPS="afterok:$cpu_id"
case "$MODE" in
  a100)
    gpu_id=$(sbatch --parsable --partition=gpu_extra --gres=gpu:a100:1 \
                    --time=10:00:00 run_gpu.sbatch)
    echo "GPU array   : job $gpu_id   MLP / FT-Transformer on gpu007 (A100), 8 tasks"
    DEPS="$DEPS,afterany:$gpu_id"
    ;;
  any)
    gpu_id=$(sbatch --parsable --partition=gpu_extra --gpus=1 \
                    --time=10:00:00 run_gpu.sbatch)
    echo "GPU array   : job $gpu_id   MLP / FT-Transformer on any GPU, 8 tasks"
    DEPS="$DEPS,afterany:$gpu_id"
    ;;
  short)
    prev=""
    for i in $(seq 1 "$CHAIN"); do
      if [ -z "$prev" ]; then
        gpu_id=$(sbatch --parsable --partition=gpu_short --gpus=1 \
                        --time=02:00:00 run_gpu.sbatch)
      else
        # afterany, not afterok: a link that hits the 2h wall ends in TIMEOUT,
        # and the next link is exactly what is supposed to pick up from its
        # checkpoint. Links whose cells are all DONE exit in seconds.
        gpu_id=$(sbatch --parsable --partition=gpu_short --gpus=1 \
                        --time=02:00:00 --dependency=afterany:"$prev" \
                        run_gpu.sbatch)
      fi
      echo "GPU chain $i : job $gpu_id   gpu_short 2h, resumes from checkpoint"
      prev="$gpu_id"
    done
    DEPS="$DEPS,afterany:$prev"
    ;;
  cpu)
    # Measured on this data: the MLP is a few hours per outcome on CPU, which is
    # a real fallback. The FT-Transformer is ~4-5 hours PER EPOCH on CPU, which
    # is not -- so it still goes to the gpu_short chain, where an epoch is
    # minutes. Use GPU_MODE=cpu-all to force both onto CPU anyway.
    mlp_id=$(sbatch --parsable --array=0-3 run_cpu_torch.sbatch)
    echo "MLP (CPU)   : job $mlp_id   cpu_extra, 4 tasks (hours per outcome)"
    prev=""
    for i in $(seq 1 "$CHAIN"); do
      dep=()
      if [ -n "$prev" ]; then dep=(--dependency=afterany:"$prev"); fi
      ft_id=$(sbatch --parsable --partition=gpu_short --gpus=1 --time=02:00:00 \
                     --array=4-7 "${dep[@]}" run_gpu.sbatch)
      echo "FT chain $i : job $ft_id   gpu_short 2h, resumes from checkpoint"
      prev="$ft_id"
    done
    DEPS="$DEPS,afterany:$mlp_id,afterany:$prev"
    ;;
  cpu-all)
    gpu_id=$(sbatch --parsable --array=0-7 run_cpu_torch.sbatch)
    echo "torch (CPU) : job $gpu_id   MLP + FT-Transformer on cpu_extra, 8 tasks"
    echo "              WARNING: the FT-Transformer is ~4-5 h PER EPOCH on CPU."
    echo "              Expect days per outcome. GPU_MODE=short is far faster."
    DEPS="$DEPS,afterany:$gpu_id"
    ;;
  none)
    echo "GPU half    : SKIPPED (GPU_MODE=none). summary_*.csv will cover four"
    echo "              model families, not six; rerun the GPU half later and"
    echo "              re-submit run_post.sbatch."
    ;;
  *)
    echo "GPU_MODE must be auto, a100, short, any, cpu, cpu-all or none " \
         "(got '$MODE')" >&2
    exit 1
    ;;
esac

# ── 3. post-processing ───────────────────────────────────────────────────────
post_id=$(sbatch --parsable --dependency="$DEPS" run_post.sbatch)
echo "post        : job $post_id   verify + feature report + SBUH inference + evaluate"

cat <<TXT

watch    : squeue -u \$USER
logs     : $(cd .. && pwd)/logs/tabular_*.log
results  : $TABULAR_OUT_DIR/summary_puf.csv
           $TABULAR_OUT_DIR/summary_sbuh.csv
           $TABULAR_OUT_DIR/feature_availability.md

Rerunning any array is safe: cells with a DONE marker are skipped, and
unfinished neural nets resume from their per-epoch checkpoint.
TXT
