"""
Generate frozen-encoder embeddings for one (encoder, cohort, split) shard.

    python embed.py --encoder bio_clinicalbert --cohort puf --split test \
                    --shard 0 --n-shards 8

ROW ORDER IS THE CONTRACT

  Everything downstream aligns by position: the outcome parquets, the tabular arm's split
  file, the per-case scores. So the output array is PREALLOCATED at full height
  and each shard writes only its own row range into that file. There is no
  concatenation step, which means there is no concatenation bug -- the classic
  way this goes wrong is shards finishing out of order and being stacked in
  completion order rather than row order.

  Two consequences worth knowing:
    - shards can run in any order, in parallel, and can be re-run individually;
    - a crashed shard leaves zeros in its range, so completion is tracked by a
      per-shard DONE marker and verified by verify.py, never by "the file exists".

CHUNKING

  Each row becomes n_chunks passages (see chunking.py). All chunks for a batch
  of rows go through the encoder in one pass, then are pooled back per row:
  L2-normalize each chunk vector, average, L2-normalize again. Normalizing
  before the average stops the chunk holding the free-text procedure
  description from dominating by magnitude alone.

NO ENCODER WEIGHT IS UPDATED. torch.inference_mode throughout, model in eval.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import chunking     # noqa: E402
import config_encoder      # noqa: E402
import serialize    # noqa: E402


def shard_bounds(n_rows, shard, n_shards):
    edges = np.linspace(0, n_rows, n_shards + 1).astype(np.int64)
    return int(edges[shard]), int(edges[shard + 1])


def done_marker(encoder, cohort, split, shard, n_shards):
    d = os.path.join(config_encoder.EMB_DIR, encoder, "_shards")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{cohort}_{split}.{shard}of{n_shards}.done")


def allocate(path, n_rows, dim):
    """Create the full-height array once; every shard then opens it r+.

    np.lib.format.open_memmap writes a real .npy header, so the result is a
    normal .npy that np.load(mmap_mode='r') reads without special handling.
    """
    if os.path.exists(path):
        a = np.lib.format.open_memmap(path, mode="r")
        if a.shape != (n_rows, dim):
            raise AssertionError(
                f"{path} exists with shape {a.shape}, expected {(n_rows, dim)}. "
                f"Delete it deliberately if the plan changed.")
        del a
        return
    tmp = f"{path}.alloc{os.getpid()}"
    a = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.dtype(config_encoder.EMB_DTYPE), shape=(n_rows, dim))
    a.flush()
    del a
    os.replace(tmp, path)


def load_encoder(slug, device):
    from transformers import AutoModel, AutoTokenizer

    spec = config_encoder.ENCODERS[slug]
    if not spec.get("revision"):
        raise SystemExit(
            f"{slug}: no pinned revision in config_encoder.ENCODERS. "
            f"Run fetch_models.py first -- an unpinned model name is not a "
            f"reproducible reference.")
    kw = dict(revision=spec["revision"])
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], **kw)
    model = AutoModel.from_pretrained(spec["hf_id"], **kw)
    model.eval().to(device)
    if device.type == "cuda":
        model.half()
    return tok, model, spec


def pool(hidden, attn_mask, how):
    if how == "cls":
        return hidden[:, 0]
    if how == "mean":
        m = attn_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)
    raise ValueError(f"unknown pooling {how!r}")


def embed_shard(encoder, cohort, split, shard, n_shards, force=False):
    marker = done_marker(encoder, cohort, split, shard, n_shards)
    if os.path.exists(marker) and not force:
        print(f"[skip] {marker} exists")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("no GPU visible; embed.py is not meant for the login node")

    plan = chunking.load_plan()
    tok, model, spec = load_encoder(encoder, device)
    dim = spec["dim"]

    n_rows = serialize.n_rows(cohort, split)
    lo, hi = shard_bounds(n_rows, shard, n_shards)
    path = config_encoder.emb_path(encoder, cohort, split)
    allocate(path, n_rows, dim)
    out = np.lib.format.open_memmap(path, mode="r+")
    if out.shape[1] != dim:
        raise AssertionError(f"{path} has width {out.shape[1]}, encoder is {dim}")

    print(f"[{encoder}] {cohort}_{split} shard {shard}/{n_shards} "
          f"rows [{lo:,},{hi:,}) of {n_rows:,}, {plan['n_chunks']} chunks/row, "
          f"dim {dim} on {torch.cuda.get_device_name(0)}", flush=True)

    cols = serialize.columns(cohort, split)
    mask = (set(config_encoder.TRAINING_EMPTY_COLS)
            if config_encoder.MASK_TRAINING_EMPTY else set())
    n_chunks = plan["n_chunks"]
    rows_per_pass = max(1, config_encoder.EMBED_BATCH // n_chunks)

    truncated = 0
    done_rows = 0
    t0 = time.time()
    pending_rows, pending_texts = [], []

    def flush():
        nonlocal truncated, done_rows, pending_rows, pending_texts
        if not pending_rows:
            return
        enc = tok(pending_texts, padding=True, truncation=True,
                  max_length=config_encoder.MAX_LEN, return_tensors="pt")
        # Truncation must never fire: the plan is sized so it cannot. Counted
        # rather than assumed, and reported at the end -- a silent truncation
        # here is exactly the failure that made the earlier arm unusable.
        full = tok(pending_texts, padding=False, truncation=False)["input_ids"]
        truncated += sum(1 for ids in full if len(ids) > config_encoder.MAX_LEN)
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.inference_mode():
            hidden = model(**enc).last_hidden_state
            vec = pool(hidden, enc["attention_mask"], spec["pooling"]).float()
        vec = torch.nn.functional.normalize(vec, dim=-1)
        vec = vec.view(len(pending_rows), n_chunks, dim).mean(dim=1)
        if config_encoder.L2_NORMALIZE:
            vec = torch.nn.functional.normalize(vec, dim=-1)
        out[pending_rows[0]:pending_rows[-1] + 1] = (
            vec.to(torch.float16).cpu().numpy())
        done_rows += len(pending_rows)
        pending_rows, pending_texts = [], []

    for start, batch_rows in _iter_rows(cohort, split, cols, mask, lo, hi):
        for offset, values in enumerate(batch_rows):
            pending_rows.append(start + offset)
            pending_texts.extend(chunking.render_chunks(values, cols, plan))
            if len(pending_rows) >= rows_per_pass:
                flush()
                if done_rows % 200_000 < rows_per_pass:
                    el = time.time() - t0
                    print(f"  {done_rows:,}/{hi - lo:,} rows  "
                          f"{done_rows / max(el, 1e-9):.0f} rows/s", flush=True)
    flush()
    out.flush()
    del out

    if done_rows != hi - lo:
        raise AssertionError(f"wrote {done_rows:,} rows, expected {hi - lo:,}")
    if truncated:
        raise AssertionError(
            f"{truncated:,} chunks exceeded max_length={config_encoder.MAX_LEN}. The "
            f"chunk plan is stale relative to the template. Recalibrate; do "
            f"NOT let this pass -- truncation silently drops fields.")

    elapsed = time.time() - t0
    with open(marker, "w") as fh:
        json.dump({"rows": [lo, hi], "seconds": round(elapsed, 1),
                   "rows_per_sec": round(done_rows / max(elapsed, 1e-9), 1),
                   "config_sha256": config_encoder.config_sha256(),
                   "revision": spec["revision"]}, fh)
    print(f"[done] {done_rows:,} rows in {elapsed / 60:.1f} min "
          f"({done_rows / elapsed:.0f} rows/s), 0 truncated", flush=True)


def _iter_rows(cohort, split, cols, mask, lo, hi):
    """Yield (absolute_start_row, [row_values, ...]) restricted to [lo, hi)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(serialize.parquet_path(cohort, split))
    pos = 0
    for batch in pf.iter_batches(batch_size=4096, columns=cols):
        n = batch.num_rows
        if pos + n <= lo or pos >= hi:
            pos += n
            continue
        data = {c: batch.column(i).to_pylist() for i, c in enumerate(cols)}
        a, b = max(lo - pos, 0), min(hi - pos, n)
        rows = [[None if c in mask else serialize._fmt(data[c][r]) for c in cols]
                for r in range(a, b)]
        yield pos + a, rows
        pos += n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(config_encoder.ENCODERS))
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    embed_shard(a.encoder, a.cohort, a.split, a.shard, a.n_shards, a.force)


if __name__ == "__main__":
    main()
