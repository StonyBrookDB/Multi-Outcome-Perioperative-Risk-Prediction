"""
The gates from PLAN.md sec 9. Nothing long may be queued until these pass.

    python gate_check.py --calibrate     # build and freeze chunk_plan.json
    python gate_check.py --verify        # zero truncation, on a large sample
    python gate_check.py --throughput    # rows/s per encoder, needs a GPU

`--verify` is the gate that matters. The earlier arm failed exactly here and
nobody noticed, because a tokenizer asked to truncate does so silently and
every downstream number still looks reasonable. So this checks the real
tokenizers against the real rendered text on real rows of all three cohorts,
and it checks the WORST row, not the average.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import chunking     # noqa: E402
import config_encoder      # noqa: E402
import serialize    # noqa: E402


def _tokenizers():
    from transformers import AutoTokenizer
    toks, names = [], []
    for slug, spec in config_encoder.ENCODERS.items():
        kw = {"revision": spec["revision"]} if spec.get("revision") else {}
        toks.append(AutoTokenizer.from_pretrained(spec["hf_id"], **kw))
        names.append(slug)
    return toks, names


def calibrate(n_sample):
    toks, names = _tokenizers()
    print(f"calibrating on {n_sample:,} PUF training rows, "
          f"worst case across {', '.join(names)} ...")
    plan = chunking.build_plan(toks, names, n_sample=n_sample)
    path = chunking.save_plan(plan)
    print(chunking.describe(plan))
    print(f"\nwritten: {path}")
    print(f"config_sha256 = {config_encoder.config_sha256()}")
    return plan


def verify(n_sample):
    """Tokenize real rendered chunks and assert none exceeds MAX_LEN."""
    plan = chunking.load_plan()
    toks, names = _tokenizers()
    print(f"verifying zero truncation at max_length={config_encoder.MAX_LEN}, "
          f"{plan['n_chunks']} chunks/row\n")

    ok = True
    for cohort, split in config_encoder.COHORTS:
        cols = serialize.columns(cohort, split)
        mask = (set(config_encoder.TRAINING_EMPTY_COLS)
                if config_encoder.MASK_TRAINING_EMPTY else set())
        worst = {n: 0 for n in names}
        over = {n: 0 for n in names}
        seen = 0
        for start, rows in _rows(cohort, split, cols, mask, n_sample):
            for values in rows:
                for text in chunking.render_chunks(values, cols, plan):
                    for tk, name in zip(toks, names):
                        n = len(tk(text, add_special_tokens=True)["input_ids"])
                        worst[name] = max(worst[name], n)
                        over[name] += n > config_encoder.MAX_LEN
                seen += 1
            if seen >= n_sample:
                break
        for name in names:
            flag = "FAIL" if over[name] else "ok  "
            headroom = config_encoder.MAX_LEN - worst[name]
            print(f"  [{flag}] {cohort}_{split:5s} {name:18s} "
                  f"longest chunk {worst[name]:3d} tokens "
                  f"(headroom {headroom:3d})  over-limit {over[name]}")
            ok &= not over[name]
        print(f"         rows checked: {seen:,}\n")

    print("GATE 1:", "PASS" if ok else "FAIL -- do not queue the full run")
    return ok


def throughput(n_sample):
    """rows/s per encoder on this GPU, and the extrapolated wall clock."""
    import torch

    import embed as embed_mod

    if not torch.cuda.is_available():
        raise SystemExit("--throughput needs a GPU; submit it, do not run it here")
    plan = chunking.load_plan()
    cols = serialize.columns("puf", "test")
    mask = (set(config_encoder.TRAINING_EMPTY_COLS)
            if config_encoder.MASK_TRAINING_EMPTY else set())
    total_rows = sum(config_encoder.N_ROWS[c] for c in config_encoder.COHORTS)
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"total rows to embed per encoder: {total_rows:,} "
          f"x {plan['n_chunks']} chunks\n")

    for slug in config_encoder.ENCODERS:
        tok, model, spec = embed_mod.load_encoder(slug, device)
        rows_per_pass = max(1, config_encoder.EMBED_BATCH // plan["n_chunks"])
        texts, n_done = [], 0
        t0 = None
        for start, rows in _rows("puf", "test", cols, mask, n_sample):
            for values in rows:
                texts.extend(chunking.render_chunks(values, cols, plan))
                n_done += 1
                if n_done % rows_per_pass == 0:
                    if t0 is None:            # first pass warms the kernels
                        _forward(tok, model, spec, texts, device)
                        texts = []
                        t0 = time.time()
                        n_done = 0
                        continue
                    _forward(tok, model, spec, texts, device)
                    texts = []
            if t0 is not None and n_done >= n_sample:
                break
        el = time.time() - t0
        rate = n_done / el
        hours = total_rows / rate / 3600
        print(f"  {slug:18s} {rate:7.0f} rows/s  ->  "
              f"{hours:5.2f} h on 1 GPU, {hours / 4:5.2f} h on 4")
        del model
        torch.cuda.empty_cache()


def _forward(tok, model, spec, texts, device):
    import torch

    import embed as embed_mod

    enc = tok(texts, padding=True, truncation=True,
              max_length=config_encoder.MAX_LEN, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        h = model(**enc).last_hidden_state
        embed_mod.pool(h, enc["attention_mask"], spec["pooling"])


def _rows(cohort, split, cols, mask, limit):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(serialize.parquet_path(cohort, split))
    pos = 0
    for batch in pf.iter_batches(batch_size=2048, columns=cols):
        data = {c: batch.column(i).to_pylist() for i, c in enumerate(cols)}
        rows = [[None if c in mask else serialize._fmt(data[c][r]) for c in cols]
                for r in range(batch.num_rows)]
        yield pos, rows
        pos += batch.num_rows
        if pos >= limit:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--throughput", action="store_true")
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    if not (a.calibrate or a.verify or a.throughput):
        ap.error("pick at least one of --calibrate / --verify / --throughput")
    if a.calibrate:
        calibrate(a.n)
    if a.verify and not verify(a.n):
        sys.exit(1)
    if a.throughput:
        throughput(min(a.n, 20000))


if __name__ == "__main__":
    main()
