"""
Metric computation + artifact saving, shared by every training script.

For each (model, label) run we persist to outputs/<model>/<label>/:
    test_pred.parquet   y_true + y_prob on the held-out test set
    metrics.json        AUROC / AUPRC / Brier / log-loss / prevalence / n / time
and append a one-line summary to outputs/summary.csv.
"""

import csv
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

import config


def best_f1(y_true, y_prob):
    """F1 at the threshold that maximises F1 (optimal operating point).

    Threshold-dependent, so we report the best achievable F1 over all
    thresholds rather than F1 at a fixed 0.5 cut -- this is comparable across
    models regardless of probability scale/calibration (it depends only on the
    ranking). Returns (f1, threshold)."""
    p, r, thr = precision_recall_curve(y_true, y_prob)
    denom = p + r
    f1 = np.where(denom > 0, 2 * p * r / np.maximum(denom, 1e-12), 0.0)
    idx = int(np.argmax(f1))
    thr_best = float(thr[idx]) if idx < len(thr) else 1.0
    return float(f1[idx]), thr_best


def compute(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1 - 1e-7)
    n = int(len(y_true))
    n_pos = int(y_true.sum())
    # F1 is reported separately via f1_crossfit.py (cross-fitted threshold), not
    # here, to avoid an optimistic max-over-thresholds F1 in the summary.
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n - n_pos,
        "prevalence": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }


def save_run(model, label, y_true, y_prob, train_seconds, extra=None):
    d = config.out_dir(model, label)

    # full per-patient predictions at float64 so no precision is lost; any
    # threshold-based metric can be recomputed from this later.
    pd.DataFrame({"y_true": np.asarray(y_true).astype(int),
                  "y_prob": np.asarray(y_prob, dtype=np.float64)}).to_parquet(
        os.path.join(d, "test_pred.parquet"))

    m = compute(y_true, y_prob)
    m.update({"model": model, "label": label,
              "train_seconds": round(float(train_seconds), 1)})
    if extra:
        m.update(extra)
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)

    _append_summary(m)
    print(f"[{model} / {label}]  AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
          f"Brier={m['brier']:.4f}  ({m['train_seconds']}s)")
    return m


_SUMMARY_COLS = ["model", "label", "auroc", "auprc", "brier", "logloss",
                 "prevalence", "n", "n_pos", "n_neg", "train_seconds"]


def _append_summary(m):
    os.makedirs(config.OUT_DIR, exist_ok=True)
    path = os.path.join(config.OUT_DIR, "summary.csv")
    exists = os.path.exists(path)
    row = {k: m.get(k) for k in _SUMMARY_COLS}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_COLS)
        if not exists:
            w.writeheader()
        w.writerow(row)


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.seconds = time.time() - self.t0
