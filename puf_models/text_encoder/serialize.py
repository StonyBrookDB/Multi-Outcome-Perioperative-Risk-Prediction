"""
PUF rows -> one sentence per case.

Streaming, because the development cohort is 4,995,670 x 73 and there is no
reason to hold the text for all of it at once: embed.py consumes batches and
keeps only the resulting vectors.

VALUE FORMATTING IS PART OF THE FROZEN SPEC

  Not cosmetic. `Age in years at surgery is 58.0` is not a sentence a clinical
  corpus contains, and the tokenizer spends real tokens on the `.0`. The rules
  below are fixed here, applied identically to every cohort, and hashed into
  config_encoder.config_sha256():

    missing (NaN / None)     -> config_encoder.MISSING_TOKEN
    integral float           -> rendered without the decimal   58.0  -> 58
    other float              -> %.10g, which drops trailing noise
                                0.7300000000000001 -> 0.73
    int                      -> as-is
    string                   -> as-is, whitespace-stripped

  PUF's -99 sentinel is already resolved to NaN upstream in the feature_73
  build (verified 2026-09-01: no -99 survives in any numeric column), so there
  is no sentinel handling here. If a future rebuild changes that, this file
  must gain it -- a serialised `-99` would read to the encoder as a real
  measurement.
"""

import os
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config_encoder      # noqa: E402
import labels73     # noqa: E402


def parquet_path(cohort, split):
    return os.path.join(config_encoder.DATA_DIR, f"{cohort}_X_{split}.parquet")


def columns(cohort="puf", split="test"):
    """The 73 columns, in the parquet's own order.

    Asserted against labels73 rather than trusted: a column added, removed or
    reordered upstream would otherwise silently shift every label onto the
    wrong variable, and every downstream number would still look plausible.
    """
    cols = list(pq.ParquetFile(parquet_path(cohort, split)).schema_arrow.names)
    missing = [c for c in cols if c not in labels73.LABELS]
    unused = [c for c in labels73.LABELS if c not in cols]
    if missing or unused:
        raise AssertionError(
            f"labels73 does not match {cohort}_{split}: "
            f"columns with no label {missing}; labels with no column {unused}")
    if len(cols) != 73:
        raise AssertionError(f"expected 73 columns, found {len(cols)}")
    return cols


def _fmt(v):
    """One cell -> the string that goes in the sentence, or None if missing."""
    if v is None:
        return None
    if isinstance(v, float):
        if np.isnan(v):
            return None
        return str(int(v)) if v.is_integer() else "%.10g" % v
    if isinstance(v, (np.floating,)):
        f = float(v)
        if np.isnan(f):
            return None
        return str(int(f)) if f.is_integer() else "%.10g" % f
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    s = str(v).strip()
    return s if s else None


def iter_batches(cohort, split, batch_size=8192, mask_cols=None):
    """Yield (start_row, [sentence, ...]) in parquet row order.

    Row order is the contract everything downstream depends on: embeddings are
    written into a preallocated memmap at these offsets, and the outcome
    parquets are aligned by position. Nothing here sorts, filters or reorders.
    """
    cols = columns(cohort, split)
    if mask_cols is None:
        mask_cols = (set(config_encoder.TRAINING_EMPTY_COLS)
                     if config_encoder.MASK_TRAINING_EMPTY else set())
    unknown = mask_cols - set(cols)
    if unknown:
        raise AssertionError(f"mask columns not in the data: {sorted(unknown)}")

    pf = pq.ParquetFile(parquet_path(cohort, split))
    start = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        # to_pylist() gives native Python scalars and None for nulls, which is
        # what _fmt expects; going through pandas here would reintroduce NaN
        # and, for the string columns, dtype-dependent null handling.
        data = {c: batch.column(i).to_pylist() for i, c in enumerate(cols)}
        n = batch.num_rows
        texts = []
        for r in range(n):
            vals = [None if c in mask_cols else _fmt(data[c][r]) for c in cols]
            texts.append(labels73.render(vals, cols, config_encoder.MISSING_TOKEN))
        yield start, texts
        start += n


def n_rows(cohort, split):
    n = pq.ParquetFile(parquet_path(cohort, split)).metadata.num_rows
    expected = config_encoder.N_ROWS.get((cohort, split))
    if expected is not None and n != expected:
        raise AssertionError(
            f"{cohort}_{split}: parquet has {n:,} rows, config_encoder expects "
            f"{expected:,}. The cohort changed; the tabular arm's split and outcome "
            f"parquets no longer align.")
    return n


def sample(cohort, split, k=3):
    """First k sentences, for eyeballing and for the gate scripts."""
    for _, texts in iter_batches(cohort, split, batch_size=max(k, 64)):
        return texts[:k]
    return []


if __name__ == "__main__":
    coh, spl = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("puf", "test")
    print(f"{coh}_{spl}: {n_rows(coh, spl):,} rows, {len(columns(coh, spl))} columns")
    print(f"mask_training_empty={config_encoder.MASK_TRAINING_EMPTY} "
          f"({', '.join(config_encoder.TRAINING_EMPTY_COLS)})")
    for i, t in enumerate(sample(coh, spl, 2)):
        print(f"\n--- row {i} ({len(t)} chars) ---\n{t}")
