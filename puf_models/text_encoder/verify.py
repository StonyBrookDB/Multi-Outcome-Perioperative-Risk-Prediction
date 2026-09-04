"""
Check that what is on disk is what the run claims it is.

    python verify.py --embeddings     after the embedding array
    python verify.py --heads          after the head array
    python verify.py --all

Every check here exists because its failure mode is SILENT. A missing shard
leaves zeros that train perfectly happily; a stale chunk plan changes what the
encoder saw without changing any shape; a prediction file with the tabular arm's schema
but the wrong row count still merges into a table. None of these announce
themselves, and all of them produce numbers that look reasonable.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config_encoder  # noqa: E402

OK, BAD = "ok  ", "FAIL"
_fail = []


def check(cond, msg):
    print(f"  [{OK if cond else BAD}] {msg}")
    if not cond:
        _fail.append(msg)
    return cond


def verify_embeddings(sample=20000):
    print("\n=== embeddings ===")
    cfg_sha = config_encoder.config_sha256()
    for enc, spec in config_encoder.ENCODERS.items():
        print(f"\n{enc} ({spec['hf_id']})")
        for cohort, split in config_encoder.COHORTS:
            p = config_encoder.emb_path(enc, cohort, split)
            name = f"{cohort}_{split}"
            if not os.path.exists(p):
                check(False, f"{name}: missing {p}")
                continue
            a = np.load(p, mmap_mode="r")
            n = config_encoder.N_ROWS[(cohort, split)]
            check(a.shape == (n, spec["dim"]),
                  f"{name}: shape {a.shape} == ({n:,}, {spec['dim']})")
            check(str(a.dtype) == config_encoder.EMB_DTYPE,
                  f"{name}: dtype {a.dtype} == {config_encoder.EMB_DTYPE}")

            # Shard completion. A crashed shard leaves its rows as zeros, so
            # markers -- not file existence -- are what says the array finished.
            sd = os.path.join(config_encoder.EMB_DIR, enc, "_shards")
            marks = [m for m in os.listdir(sd) if m.startswith(f"{name}.")] \
                if os.path.isdir(sd) else []
            covered = np.zeros(n, dtype=bool)
            stale = []
            for m in marks:
                with open(os.path.join(sd, m)) as fh:
                    meta = json.load(fh)
                lo, hi = meta["rows"]
                covered[lo:hi] = True
                if meta.get("config_sha256") != cfg_sha:
                    stale.append(m)
            check(covered.all(),
                  f"{name}: all {n:,} rows covered by shard markers "
                  f"({len(marks)} marker(s), {int((~covered).sum()):,} uncovered)")
            check(not stale,
                  f"{name}: all shards built under the current config_sha256"
                  + (f" -- stale: {stale}" if stale else ""))

            # Zero rows would mean a shard wrote nothing there. Checked on a
            # spread sample rather than 5M rows.
            idx = np.linspace(0, n - 1, min(sample, n)).astype(int)
            block = np.asarray(a[idx], dtype=np.float32)
            zero = int((np.abs(block).sum(1) == 0).sum())
            check(zero == 0, f"{name}: no all-zero rows in {len(idx):,} sampled")
            if config_encoder.L2_NORMALIZE:
                norms = np.linalg.norm(block, axis=1)
                check(bool(np.all(np.abs(norms - 1.0) < 5e-2)),
                      f"{name}: L2 norms ~1 (min {norms.min():.4f}, "
                      f"max {norms.max():.4f})")


def verify_splits():
    print("\n=== the tabular arm split reuse ===")
    import pyarrow.parquet as pq

    per_label = {}
    for lab in config_encoder.LABELS:
        p = config_encoder.split_path(lab)
        if not os.path.exists(p):
            check(False, f"{lab}: missing {p}")
            continue
        t = pd.read_parquet(p)
        fit = np.sort(t.loc[t["split"] == "fit", "row"].to_numpy())
        val = np.sort(t.loc[t["split"] == "val", "row"].to_numpy())
        per_label[lab] = val
        n = config_encoder.N_ROWS[("puf", "train")]
        check(len(fit) + len(val) == n and not np.intersect1d(fit, val).size,
              f"{lab}: fit {len(fit):,} + val {len(val):,} = {n:,}, disjoint")

    # The split is identical across the six families for one outcome and
    # DIFFERENT across outcomes. Asserting both ways round catches the classic
    # error of comparing the `split` column, which is identical everywhere.
    if len(per_label) == len(config_encoder.LABELS):
        labs = list(per_label)
        diffs = [not np.array_equal(per_label[labs[i]], per_label[labs[j]])
                 for i in range(len(labs)) for j in range(i + 1, len(labs))]
        check(all(diffs),
              "the four outcomes have genuinely different val sets "
              "(they are stratified per outcome)")
    other = "xgboost"
    p2 = os.path.join(config_encoder.TABULAR_OUT_DIR, other, config_encoder.LABELS[0],
                      "split_indices.parquet")
    if os.path.exists(p2):
        a = np.sort(pd.read_parquet(p2).query("split=='val'")["row"].to_numpy())
        check(np.array_equal(a, per_label[config_encoder.LABELS[0]]),
              f"{config_encoder.LABELS[0]}: split identical between "
              f"{config_encoder.SPLIT_SOURCE_MODEL} and {other}")


def verify_heads():
    print("\n=== heads ===")
    tabular_schema = None
    p5 = os.path.join(config_encoder.TABULAR_OUT_DIR, "mlp", "mortality",
                      "test_pred.parquet")
    if os.path.exists(p5):
        tabular_schema = list(pd.read_parquet(p5).columns)

    for enc in config_encoder.ENCODERS:
        for lab in config_encoder.LABELS:
            d = config_encoder.out_dir(enc, lab)
            tag = f"{enc}/{lab}"
            if not os.path.exists(os.path.join(d, "DONE")):
                check(False, f"{tag}: no DONE marker")
                continue
            for name, n in (("test", config_encoder.N_ROWS[("puf", "test")]),
                            ("sbuh", config_encoder.N_ROWS[("sbuh", "test")])):
                f = os.path.join(d, f"{name}_pred.parquet")
                if not os.path.exists(f):
                    check(False, f"{tag}: missing {name}_pred.parquet")
                    continue
                t = pd.read_parquet(f)
                check(len(t) == n, f"{tag}/{name}: {len(t):,} rows == {n:,}")
                if tabular_schema:
                    check(list(t.columns) == tabular_schema,
                          f"{tag}/{name}: schema matches the tabular arm {tabular_schema}")
                check(t["score_raw"].notna().all() and np.isfinite(
                          t["score_raw"]).all(),
                      f"{tag}/{name}: scores all finite")

            m = os.path.join(d, "train_meta.json")
            if os.path.exists(m):
                import head_config
                meta = json.load(open(m))
                check(meta.get("head_sha256") == head_config.head_sha256(),
                      f"{tag}: trained under the current head_sha256")
                check(meta.get("calibration") == "none",
                      f"{tag}: no calibration applied")
                # The embedding-stage hash is recorded but NOT compared against
                # the live one: config_encoder.py may legitimately move on after the
                # embeddings are built. What must agree is this cell's hash and
                # the hash its own embeddings were built under, and that is what
                # the shard markers carry.
                if meta.get("standardize_input"):
                    sp = head_config.scaler_path(enc, lab)
                    check(os.path.exists(sp),
                          f"{tag}: fitted scaler archived at {os.path.basename(sp)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--heads", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not (a.embeddings or a.heads or a.all):
        a.all = True

    print(f"config_sha256 = {config_encoder.config_sha256()}")
    if a.embeddings or a.all:
        verify_embeddings()
        verify_splits()
    if a.heads or a.all:
        verify_heads()

    print("\n" + "=" * 60)
    if _fail:
        print(f"{len(_fail)} CHECK(S) FAILED:")
        for m in _fail:
            print(f"  - {m}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
