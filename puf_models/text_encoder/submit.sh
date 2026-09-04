#!/usr/bin/env bash
#
# the text-encoder arm submission helper. Nothing here runs work itself -- it only calls sbatch.
#
#   ./submit.sh gate            gates 1+2, one GPU, ~15 min. Run this FIRST.
#   ./submit.sh embed           embedding array, after the gate log is clean
#   ./submit.sh embed a100      ... pinned to the 8-GPU a100 node
#   ./submit.sh chain           verify -> head -> eval, chained on the embed job
#   ./submit.sh head            12 head tasks, after embeddings verify clean
#   ./submit.sh eval            bootstrap + summary CSVs (CPU)
#   ./submit.sh verify          check what is on disk, runs here, no sbatch
#   ./submit.sh status          what has finished so far
#
# The gate job must show "GATE 1: PASS" before embed is submitted. Gate 1 is
# the check that the earlier arm silently failed: every case truncated at 512,
# 33 of 73 fields never reaching the encoder.

set -euo pipefail
cd "$(dirname "$0")"

CMD="${1:-}"; FLAVOUR="${2:-}"

case "$CMD" in
  gate)
    sbatch run_gate.sbatch
    ;;
  embed)
    if [ ! -s chunk_plan.json ]; then
      echo "chunk_plan.json missing -- run: python gate_check.py --calibrate" >&2
      exit 1
    fi
    # Refuse until a gate log actually records a pass. Submitting `gate` and
    # `embed` back to back does NOT work: the array would queue behind a gate
    # whose result nobody has read, which is the whole point of having a gate.
    # ENCODER_SKIP_GATE=1 overrides, deliberately and visibly.
    LOGS=/vast/projects/akumar-group/NSQIP/zihan/puf_models/logs
    if [ "${ENCODER_SKIP_GATE:-0}" != "1" ] \
       && ! grep -lq "GATE 4: PASS" "$LOGS"/encoder_gate_*.log 2>/dev/null; then
      echo "refusing: no encoder_gate_*.log records GATE 4: PASS." >&2
      echo "  run ./submit.sh gate, wait for it, read the log, then retry." >&2
      echo "  override with ENCODER_SKIP_GATE=1 if you know why." >&2
      exit 1
    fi
    # Gate 2 measured the longest single task -- a medembed_large train shard,
    # 312,229 rows at 246 rows/s -- at ~21 min. Ask for the gpu_short maximum
    # anyway.
    #
    # 1 h looked like a comfortable 3x margin and backfills into smaller gaps,
    # but that benchmark ran on a quiet node with a warm cache. Each shard also
    # does ~940k single-threaded tokenizations (3 chunks x 312k rows,
    # TOKENIZERS_PARALLELISM=false), and that part is CPU-bound: the GPU is
    # exclusive but the node's cores are not, so a busy node slows exactly the
    # stage the benchmark could not see. A TIMEOUT costs the whole shard's ~22
    # min, an over-long limit costs a little backfill eligibility. 2 h is the
    # partition cap, so asking for it carries no penalty beyond that.
    #
    # Default is ANY gpu, deliberately. `a100` pins to --gres=gpu:a100:1, which
    # is the only way to reach the 8-GPU a100 node but also caps concurrency at
    # 8 and queues behind everything else wanting it. Gate 2's numbers came off
    # an RTX PRO 6000 Blackwell on the branch nodes, so the a100 branch is not
    # known to be faster per card -- it is just scarcer.
    if [ "$FLAVOUR" = "a100" ]; then
      sbatch --partition=gpu_long --gres=gpu:a100:1 --time=08:00:00 run_embed.sbatch
    else
      sbatch --partition=gpu_long --gres=gpu:1 --time=08:00:00 run_embed.sbatch
    fi
    ;;
  head)
    # gpu_long, not gpu_short. Same nodes, but gpu_short caps every job at 2 h
    # and a medembed cell has already been measured at 95 s/epoch with spikes to
    # 508 s under contention -- four outcomes do not reliably fit. gpu_long
    # allows 8 h (gpu_extra allows 10 days). Asking for headroom on a job that
    # exits when it is done costs a little backfill priority; running out of
    # wall clock costs the whole cell and, through afterok, the eval that
    # depends on it.
    sbatch --partition=gpu_long --gres=gpu:1 --time=08:00:00 run_head.sbatch
    ;;
  eval)
    sbatch run_eval.sbatch
    ;;
  chain)
    # Everything after the embedding array, submitted at once and gated on each
    # other. afterok means a stage only starts if the previous one exited 0, so
    # a failed verify stops the chain instead of training heads on bad input.
    #
    #   ./submit.sh chain            chain onto the currently running embed job
    #   ./submit.sh chain 35919      chain onto a specific job id
    DEP="${FLAVOUR:-$(squeue -u "$USER" -h -n encoder_embed -o '%A' | head -1)}"
    if [ -z "$DEP" ]; then
      echo "no running encoder_embed job found; pass the job id explicitly" >&2
      exit 1
    fi
    V=$(sbatch --parsable --dependency=afterok:"$DEP" run_verify.sbatch)
    H=$(sbatch --parsable --dependency=afterok:"$V" \
        --partition=gpu_long --gres=gpu:1 --time=08:00:00 run_head.sbatch)
    E=$(sbatch --parsable --dependency=afterok:"$H" run_eval.sbatch)
    echo "chained onto embed $DEP:"
    echo "  verify $V  ->  head $H  ->  eval $E"
    echo "results land in $(cd .. && pwd)/results/text_encoder/summary_{puf,sbuh}.csv"
    ;;
  verify)
    source ./env.sh
    "${ENCODER_PYTHON:-/vast/projects/fusheng-group/zding/miniconda3/envs/mistral_env/bin/python}" verify.py --all
    ;;
  status)
    source ./env.sh
    for enc in medembed_large clinicalbert bio_clinicalbert; do
      d="$ENCODER_OUT_DIR/embeddings/$enc"
      n=$(ls "$d/_shards" 2>/dev/null | wc -l)
      echo "$enc: $n shard(s) done"
      ls -la "$d"/*.npy 2>/dev/null | awk '{printf "    %s  %s\n", $5, $9}' || true
    done
    ;;
  *)
    sed -n '2,15p' "$0"; exit 1
    ;;
esac
