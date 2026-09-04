"""
Experiment 5 evaluation. Reads only saved predictions -- no retraining, no
re-inference -- and scores both cohorts through one code path:

    python evaluate.py --cohort puf     reads <model>/<label>/test_pred.parquet
    python evaluate.py --cohort sbuh    reads <model>/<label>/sbuh_pred.parquet

Both write results/tabular/summary_<cohort>.csv (tidy: one row per model x
outcome x rule x metric, every one with a bootstrap 95% CI) plus a wide
AUROC/AUPRC convenience pivot.

WHAT IS AND IS NOT REPORTED
---------------------------
There is no calibration in the tabular arm, so there are no calibrated quantities. Brier,
log-loss, ECE and calibration slope are deliberately absent: all six families
train with class re-weighting, `score_raw` is not a probability, and computing
those numbers on a re-weighted raw scale would produce something that looks like
a calibration result and is not one. For the same reason there is no
"risk >= 5%" style threshold anywhere. Every operating point is rank-based.

  RANKING (the primary result -- unaffected by the class re-weighting)
      AUROC, AUPRC.

  THRESHOLD (sensitivity / specificity / PPV / NPV / F1 / % flagged)
      top k%          flag the k% highest-scoring patients. A capacity rule:
                      no calibration needed, and the SAME k in both cohorts, so
                      the PUF and SBUH rows answer the same question. The cut is
                      that cohort's own score quantile.
      F1-max @ PUF val
                      one threshold, F1-maximising on the PUF VALIDATION split,
                      then frozen and applied unchanged to the PUF test year and
                      to SBUH. This is the transferable operating point: the
                      threshold is fitted on neither evaluation set, so nothing
                      selects and reports on the same rows, and SBUH gets a
                      genuinely external cut.
      F1-max (cross-fitted)
                      threshold learned on the half of the cohort that excludes
                      each patient. PUF ONLY -- on SBUH this would be fitting a
                      threshold on the external cohort, which the brief forbids.
                      Reported for continuity across runs.

Every threshold_source is written into the output, so no reader has to guess
which of the two sanctioned SBUH rules produced a row.

BOOTSTRAP
---------
Patients are resampled with replacement, 1000 draws by default (--n-boot 2000 is
supported), with ONE shared set of resamples across all metrics so paired
comparisons stay possible from the saved draws. Thresholds are held FIXED at the
values selected on the real cohort: re-selecting them inside each draw would
leak the resample into the threshold and bias the threshold metrics upward.

SBUH is 3,490 patients with ~23 deaths. Its intervals are wide, and they are the
point: a point estimate alone would invite conclusions that sample cannot
support.
"""

import argparse
import glob
import json
import os
import re
import sys
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config_tabular

CI_LO, CI_HI = 2.5, 97.5
N_JOBS = min(int(os.environ.get("BOOT_N_JOBS", str(config_tabular.N_THREADS))),
             os.cpu_count() or 1)

RANK_METRICS = ["auroc", "auprc"]
THR_METRICS = ["sensitivity", "specificity", "ppv", "npv", "f1", "flagged_pct"]


def _slug(rule):
    """Column-safe name for one operating point.

    Keyed on the RULE, not the kind: all four top-k cuts share kind="topk", so
    naming the bootstrap columns by kind collides four ways and the draws file
    cannot be written at all.
    """
    return re.sub(r"[^0-9a-z]+", "_", rule.lower()).strip("_")


# ── primitive metrics ────────────────────────────────────────────────────────

def confusion(yb, mask):
    """Counts and rates from a boolean outcome and a boolean flag mask."""
    tp = int(np.count_nonzero(mask & yb))
    fp = int(np.count_nonzero(mask & ~yb))
    fn = int(np.count_nonzero(~mask & yb))
    tn = int(np.count_nonzero(~mask & ~yb))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_flagged": tp + fp,
            "flagged_pct": 100.0 * (tp + fp) / max(len(yb), 1),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1),
            "npv": tn / max(tn + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def best_f1_threshold(y, s):
    p, r, thr = precision_recall_curve(y, s)
    d = p + r
    f1 = np.where(d > 0, 2 * p * r / np.maximum(d, 1e-12), 0.0)
    i = int(np.argmax(f1))
    return float(thr[i]) if i < len(thr) else 1.0


def crossfit_mask(y, s, label):
    """F1 flag mask with the threshold learned on the complementary half.

    The cohort is split in two; the threshold maximising F1 on half A is applied
    to half B and vice versa, so no patient is ever scored by a threshold chosen
    with their own outcome in it. Same procedure and same seeding throughout,
    so the numbers remain comparable across experiments.
    """
    rng = np.random.default_rng(config_tabular.SEED + zlib.crc32(label.encode()) % 10_000)
    perm = rng.permutation(len(y))
    a = np.zeros(len(y), dtype=bool)
    a[perm[:len(y) // 2]] = True
    b = ~a
    t_a, t_b = best_f1_threshold(y[a], s[a]), best_f1_threshold(y[b], s[b])
    mask = np.zeros(len(y), dtype=bool)
    mask[b] = s[b] >= t_a
    mask[a] = s[a] >= t_b
    return mask, t_a, t_b


# ── the rule set ─────────────────────────────────────────────────────────────

def build_rules(cohort, y, s, label, val_thr):
    """Every operating point for one cell, as (description, flag mask) pairs."""
    rules = []
    for k in config_tabular.TOP_K_PERCENT:
        cut = float(np.percentile(s, 100 - k))
        rules.append({"rule": f"top{k:g}%", "kind": "topk", "threshold": cut,
                      "threshold_source": f"score quantile within this cohort "
                                          f"(same k={k:g}% in both cohorts)",
                      "mask": s >= cut})

    rules.append({"rule": "F1-max @ PUF val", "kind": "f1_val",
                  "threshold": float(val_thr),
                  "threshold_source": "F1-maximising threshold fitted once on "
                                      "the PUF validation split; frozen and "
                                      "applied unchanged to both cohorts",
                  "mask": s >= val_thr})

    if cohort == "puf":
        mask, t_a, t_b = crossfit_mask(y, s, label)
        rules.append({"rule": "F1-max (cross-fitted)", "kind": "f1_crossfit",
                      "threshold": float(np.nan),
                      "threshold_source": f"cross-fitted within the PUF test set "
                                          f"(t_A={t_a:.6g}, t_B={t_b:.6g}); not "
                                          f"reported for SBUH, where fitting a "
                                          f"threshold is not permitted",
                      "mask": mask, "t_A": t_a, "t_B": t_b})
    return rules


# ── bootstrap ────────────────────────────────────────────────────────────────

def _metric_vector(yb, s, masks):
    if not yb.any() or yb.all():                 # degenerate resample
        return [np.nan] * (len(RANK_METRICS) + len(THR_METRICS) * len(masks))
    out = [roc_auc_score(yb, s), average_precision_score(yb, s)]
    for m in masks:
        c = confusion(yb, m)
        out += [c[k] for k in THR_METRICS]
    return out


def _boot_chunk(yb, s, masks, seeds):
    n = len(yb)
    width = len(RANK_METRICS) + len(THR_METRICS) * len(masks)
    out = np.empty((len(seeds), width))
    for i, sd in enumerate(seeds):
        idx = np.random.default_rng(sd).integers(0, n, n)
        out[i] = _metric_vector(yb[idx], s[idx], [m[idx] for m in masks])
    return out


def bootstrap(yb, s, masks, n_boot):
    seeds = np.random.SeedSequence(config_tabular.SEED).generate_state(n_boot)
    chunks = [c for c in np.array_split(seeds, N_JOBS) if len(c)]
    draws = np.vstack(Parallel(n_jobs=min(N_JOBS, len(chunks)))(
        delayed(_boot_chunk)(yb, s, masks, c) for c in chunks))
    return draws


def _ci(col):
    col = col[np.isfinite(col)]
    if len(col) < 2:
        return np.nan, np.nan, np.nan, int(len(col))
    return (float(np.percentile(col, CI_LO)), float(np.percentile(col, CI_HI)),
            float(col.std(ddof=1)), int(len(col)))


# ── curves ───────────────────────────────────────────────────────────────────

def curve_points(y, s, k=2000):
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)

    def thin(x, yv):
        if len(x) <= k:
            return x, yv
        i = np.unique(np.linspace(0, len(x) - 1, k).astype(int))
        return x[i], yv[i]

    fpr, tpr = thin(fpr, tpr)
    rec, prec = thin(rec, prec)
    n = max(len(fpr), len(rec))
    pad = lambda a: np.pad(np.asarray(a, dtype=np.float64),  # noqa: E731
                           (0, n - len(a)), constant_values=np.nan)
    return pd.DataFrame({"roc_fpr": pad(fpr), "roc_tpr": pad(tpr),
                         "pr_recall": pad(rec), "pr_precision": pad(prec)})


# ── driver ───────────────────────────────────────────────────────────────────

def evaluate_run(model, label, cohort, n_boot):
    d = config_tabular.out_dir(model, label)
    pred = pd.read_parquet(os.path.join(d, config_tabular.PRED_FILE[cohort]))
    y = pred["y_true"].to_numpy()
    yb = y == 1
    s = pred["score_raw"].to_numpy()

    # The transferable threshold. Read from the PUF validation split for BOTH
    # cohorts, so the SBUH cut is genuinely external and identical to the PUF one.
    val = pd.read_parquet(os.path.join(d, "val_pred.parquet"))
    val_thr = best_f1_threshold(val["y_true"].to_numpy(), val["score_raw"].to_numpy())

    rules = build_rules(cohort, y, s, label, val_thr)
    masks = [r["mask"] for r in rules]

    point = _metric_vector(yb, s, masks)
    draws = bootstrap(yb, s, masks, n_boot)

    base = {"cohort": cohort, "model": model, "label": label, "n": int(len(y)),
            "n_pos": int(yb.sum()), "prevalence": float(yb.mean())}
    rows = []
    for j, m in enumerate(RANK_METRICS):
        lo, hi, sd, nv = _ci(draws[:, j])
        rows.append({**base, "rule": "ranking", "kind": "ranking",
                     "threshold": np.nan,
                     "threshold_source": "none (rank metric, threshold-free)",
                     "metric": m, "value": point[j], "ci_lo": lo, "ci_hi": hi,
                     "boot_std": sd, "n_boot_valid": nv})
    for ri, r in enumerate(rules):
        c = confusion(yb, r["mask"])
        for mi, m in enumerate(THR_METRICS):
            j = len(RANK_METRICS) + ri * len(THR_METRICS) + mi
            lo, hi, sd, nv = _ci(draws[:, j])
            rows.append({**base, "rule": r["rule"], "kind": r["kind"],
                         "threshold": r["threshold"],
                         "threshold_source": r["threshold_source"],
                         "metric": m, "value": c[m], "ci_lo": lo, "ci_hi": hi,
                         "boot_std": sd, "n_boot_valid": nv,
                         "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                         "tn": c["tn"], "n_flagged": c["n_flagged"]})

    cols = (RANK_METRICS
            + [f"{_slug(r['rule'])}__{m}" for r in rules for m in THR_METRICS])
    pd.DataFrame(draws, columns=cols).to_parquet(
        os.path.join(d, f"bootstrap_{cohort}.parquet"), index=False)
    curve_points(y, s).to_parquet(os.path.join(d, f"curves_{cohort}.parquet"),
                                  index=False)
    with open(os.path.join(d, f"metrics_{cohort}.json"), "w") as f:
        json.dump({**base, "val_f1_threshold": val_thr,
                   "auroc": point[0], "auprc": point[1],
                   "rules": [{k: v for k, v in r.items() if k != "mask"}
                             for r in rules]}, f, indent=2, default=float)

    auroc_ci = _ci(draws[:, 0])
    auprc_ci = _ci(draws[:, 1])
    print(f"  [{cohort}/{model}/{label}] n={base['n']:,} pos={base['n_pos']:,}  "
          f"AUROC={point[0]:.4f} [{auroc_ci[0]:.4f},{auroc_ci[1]:.4f}]  "
          f"AUPRC={point[1]:.4f} [{auprc_ci[0]:.4f},{auprc_ci[1]:.4f}]", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="puf", choices=sorted(config_tabular.PRED_FILE))
    ap.add_argument("--n-boot", type=int, default=config_tabular.N_BOOT)
    args = ap.parse_args()

    pf = config_tabular.PRED_FILE[args.cohort]
    runs = sorted((p.split(os.sep)[-3], p.split(os.sep)[-2])
                  for p in glob.glob(os.path.join(config_tabular.OUT_DIR, "*", "*", pf)))
    if not runs:
        raise SystemExit(f"no {pf} found under {config_tabular.OUT_DIR}")
    print(f"[evaluate] cohort={args.cohort}  {len(runs)} cells  "
          f"n_boot={args.n_boot}  n_jobs={N_JOBS}", flush=True)

    rows = []
    for model, label in runs:
        rows.extend(evaluate_run(model, label, args.cohort, args.n_boot))

    df = pd.DataFrame(rows)
    order = ["cohort", "model", "label", "n", "n_pos", "prevalence", "kind",
             "rule", "metric", "value", "ci_lo", "ci_hi", "boot_std",
             "n_boot_valid", "threshold", "threshold_source",
             "tp", "fp", "fn", "tn", "n_flagged"]
    df = df[[c for c in order if c in df.columns]].sort_values(
        ["label", "kind", "metric", "value"], ascending=[True, True, True, False])
    p = os.path.join(config_tabular.OUT_DIR, f"summary_{args.cohort}.csv")
    df.to_csv(p, index=False)

    # A wide AUROC/AUPRC pivot, because that is the table people actually read
    # first. It is a view of summary_<cohort>.csv, not a second source of truth.
    w = df[df["kind"] == "ranking"].pivot_table(
        index=["model", "label", "n", "n_pos"], columns="metric",
        values=["value", "ci_lo", "ci_hi"])
    w.columns = [f"{m}_{v}" if v != "value" else m for v, m in w.columns]
    pw = os.path.join(config_tabular.OUT_DIR, f"summary_{args.cohort}_ranking.csv")
    w.reset_index().to_csv(pw, index=False)

    print(f"\nDone -> {p}\n        {pw}")


if __name__ == "__main__":
    main()
