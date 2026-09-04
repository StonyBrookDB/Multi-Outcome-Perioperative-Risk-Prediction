"""
Independently recompute a few rows and compare them against what was written.

    python spotcheck.py --encoder bio_clinicalbert --cohort sbuh --split test
    python spotcheck.py --encoder medembed_large --cohort puf --split train -n 5

WHY THIS EXISTS SEPARATELY FROM verify.py

  verify.py checks that the arrays have the right shape, no all-zero rows, unit
  norms, complete shard coverage and a matching config hash. All of that can be
  true of embeddings that are quietly WRONG: chunks pooled in the wrong order,
  a row written at the wrong offset, the wrong pooling branch taken, a
  serialisation that drifted from the one the plan was calibrated on. Those
  failures produce well-formed arrays and plausible downstream metrics, and
  they would only ever be caught by someone recomputing the answer.

  So this recomputes it. It goes back to the parquet, renders the chunks,
  loads the encoder fresh, pools, and compares to the stored vector. It shares
  chunking.render_chunks with the pipeline -- that is the specification -- but
  nothing else: the tokenisation, the forward pass, the pooling and the
  normalisation are all written out again here, so a bug in embed.py's version
  of them shows up as a mismatch.

  The stored vectors were produced in fp16 on a GPU; this recomputes in fp32 on
  whatever device is available. Bitwise equality is therefore not expected and
  not required. Cosine similarity is the test, and the threshold is tight
  enough that a genuine logic error cannot slip under it.
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import chunking     # noqa: E402
import config_encoder      # noqa: E402
import serialize    # noqa: E402

THRESHOLD = 0.999


def recompute(tok, mdl, texts, pooling, device):
    enc = tok(texts, padding=True, truncation=True,
              max_length=config_encoder.MAX_LEN, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        h = mdl(**enc).last_hidden_state
        if pooling == "cls":
            v = h[:, 0]
        elif pooling == "mean":
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        else:
            raise ValueError(pooling)
    v = torch.nn.functional.normalize(v.float(), dim=-1)
    v = v.mean(0)
    if config_encoder.L2_NORMALIZE:
        v = torch.nn.functional.normalize(v, dim=-1)
    return v.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(config_encoder.ENCODERS))
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("-n", type=int, default=5, help="rows to check")
    ap.add_argument("--rows", type=str, default="",
                    help="explicit comma-separated row indices")
    a = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    spec = config_encoder.ENCODERS[a.encoder]
    plan = chunking.load_plan()
    stored = np.load(config_encoder.emb_path(a.encoder, a.cohort, a.split), mmap_mode="r")

    if a.rows:
        rows = [int(x) for x in a.rows.split(",")]
    else:
        rows = list(range(a.n))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"])
    mdl = AutoModel.from_pretrained(spec["hf_id"],
                                    revision=spec["revision"]).eval().to(device)

    cols = serialize.columns(a.cohort, a.split)
    mask = (set(config_encoder.TRAINING_EMPTY_COLS)
            if config_encoder.MASK_TRAINING_EMPTY else set())

    import pyarrow.parquet as pq
    need = max(rows) + 1
    pf = pq.ParquetFile(serialize.parquet_path(a.cohort, a.split))
    got, data = 0, {c: [] for c in cols}
    for batch in pf.iter_batches(batch_size=4096, columns=cols):
        for i, c in enumerate(cols):
            data[c].extend(batch.column(i).to_pylist())
        got += batch.num_rows
        if got >= need:
            break

    print(f"{a.encoder} / {a.cohort}_{a.split} / pooling={spec['pooling']} / "
          f"{plan['n_chunks']} chunks / device={device}")
    print(f"recomputed fp32 vs stored fp16, threshold cosine > {THRESHOLD}\n")

    worst, bad = 1.0, 0
    for r in rows:
        vals = [None if c in mask else serialize._fmt(data[c][r]) for c in cols]
        v = recompute(tok, mdl, chunking.render_chunks(vals, cols, plan),
                      spec["pooling"], device)
        s = np.asarray(stored[r], dtype=np.float32)
        if np.abs(s).sum() == 0:
            print(f"  row {r:>8}: stored row is all zeros -- not written yet")
            continue
        cos = float(v @ s / (np.linalg.norm(v) * np.linalg.norm(s)))
        worst = min(worst, cos)
        ok = cos > THRESHOLD
        bad += not ok
        print(f"  row {r:>8}: cosine={cos:.6f}  maxabsdiff={np.abs(v - s).max():.5f}"
              f"  {'ok' if ok else '*** MISMATCH ***'}")

    print(f"\nworst cosine {worst:.6f} over {len(rows)} row(s)")
    if bad:
        print(f"SPOTCHECK FAILED: {bad} row(s) below threshold")
        sys.exit(1)
    print("SPOTCHECK PASS")


if __name__ == "__main__":
    main()
