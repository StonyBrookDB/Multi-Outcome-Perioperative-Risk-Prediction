"""
Experiment 6 evaluation. Reads only saved predictions -- no retraining, no
re-inference.

    python evaluate_encoder.py --cohort puf     reads <encoder>/<label>/test_pred.parquet
    python evaluate_encoder.py --cohort sbuh    reads <encoder>/<label>/sbuh_pred.parquet

THIS FILE DELIBERATELY CONTAINS ALMOST NO MATHEMATICS

  It imports tabular/evaluate.py and calls that module's own functions -- the same
  `build_rules`, `_metric_vector`, `bootstrap`, `_ci`, `confusion`,
  `best_f1_threshold` and `curve_points` that produced every tabular row in the
  manuscript. Only the directory resolution differs.

  Reimplementing the bootstrap here would be the easy way to end up with rows
  that are not comparable with the tabular arm's after all -- a different resampling
  scheme, a different tie rule in AUPRC, a threshold re-selected inside each
  draw. The whole point of the text-encoder arm is that its rows can sit in the tabular arm's table, and
  the strongest available guarantee of that is running the tabular arm's code.

  What that inherits, unchanged: 1,000 draws, one shared set of resamples
  across all metrics so paired comparisons remain possible; thresholds held
  fixed at the values selected on the real cohort; no calibration and therefore
  no Brier, log-loss, ECE or calibration slope; every operating point
  rank-based.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tabular"))

import config_encoder  # noqa: E402

# the tabular arm's evaluation module. Imported for its functions, not run.
sys.path.insert(0, os.path.join(REPO, "tabular"))
import evaluate as E  # noqa: E402


def evaluate_run(encoder, label, cohort, n_boot):
    """the tabular arm's evaluate.evaluate_run, with the text-encoder arm's paths and `encoder` for `model`."""
    d = config_encoder.out_dir(encoder, label)
    pred_file = config_encoder.PRED_FILE[cohort]
    pred = pd.read_parquet(os.path.join(d, pred_file))
    y = pred["y_true"].to_numpy()
    yb = y == 1
    s = pred["score_raw"].to_numpy()

    # The transferable threshold: taken from the PUF validation split for BOTH
    # cohorts, so the single-institution cut is genuinely external and is the
    # same rule as the PUF one.
    val = pd.read_parquet(os.path.join(d, "val_pred.parquet"))
    val_thr = E.best_f1_threshold(val["y_true"].to_numpy(),
                                  val["score_raw"].to_numpy())

    rules = E.build_rules(cohort, y, s, label, val_thr)
    masks = [r["mask"] for r in rules]
    point = E._metric_vector(yb, s, masks)
    draws = E.bootstrap(yb, s, masks, n_boot)

    base = {"cohort": cohort, "model": encoder, "label": label,
            "n": int(len(y)), "n_pos": int(yb.sum()),
            "prevalence": float(yb.mean())}
    rows = []
    for j, m in enumerate(E.RANK_METRICS):
        lo, hi, sd, nv = E._ci(draws[:, j])
        rows.append({**base, "rule": "ranking", "kind": "ranking",
                     "threshold": np.nan,
                     "threshold_source": "none (rank metric, threshold-free)",
                     "metric": m, "value": point[j], "ci_lo": lo, "ci_hi": hi,
                     "boot_std": sd, "n_boot_valid": nv})
    for ri, r in enumerate(rules):
        c = E.confusion(yb, r["mask"])
        for mi, m in enumerate(E.THR_METRICS):
            j = len(E.RANK_METRICS) + ri * len(E.THR_METRICS) + mi
            lo, hi, sd, nv = E._ci(draws[:, j])
            rows.append({**base, "rule": r["rule"], "kind": r["kind"],
                         "threshold": r["threshold"],
                         "threshold_source": r["threshold_source"],
                         "metric": m, "value": c[m], "ci_lo": lo, "ci_hi": hi,
                         "boot_std": sd, "n_boot_valid": nv,
                         "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                         "tn": c["tn"], "n_flagged": c["n_flagged"]})

    cols = (E.RANK_METRICS
            + [f"{E._slug(r['rule'])}__{m}" for r in rules for m in E.THR_METRICS])
    pd.DataFrame(draws, columns=cols).to_parquet(
        os.path.join(d, f"bootstrap_{cohort}.parquet"), index=False)
    E.curve_points(y, s).to_parquet(
        os.path.join(d, f"curves_{cohort}.parquet"), index=False)
    with open(os.path.join(d, f"metrics_{cohort}.json"), "w") as f:
        json.dump({**base, "experiment": 6, "val_f1_threshold": val_thr,
                   "auroc": point[0], "auprc": point[1],
                   "config_sha256": config_encoder.config_sha256(),
                   "rules": [{k: v for k, v in r.items() if k != "mask"}
                             for r in rules]}, f, indent=2, default=float)

    a0, a1 = E._ci(draws[:, 0]), E._ci(draws[:, 1])
    print(f"  [{cohort}/{encoder}/{label}] n={base['n']:,} pos={base['n_pos']:,}  "
          f"AUROC={point[0]:.4f} [{a0[0]:.4f},{a0[1]:.4f}]  "
          f"AUPRC={point[1]:.4f} [{a1[0]:.4f},{a1[1]:.4f}]", flush=True)
    return rows


def discover(cohort):
    pred_file = config_encoder.PRED_FILE[cohort]
    out = []
    for enc in config_encoder.ENCODERS:
        for lab in config_encoder.LABELS:
            d = config_encoder.out_dir(enc, lab)
            if os.path.exists(os.path.join(d, pred_file)):
                out.append((enc, lab))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="puf", choices=sorted(config_encoder.PRED_FILE))
    ap.add_argument("--n-boot", type=int, default=config_encoder.N_BOOT)
    a = ap.parse_args()

    runs = discover(a.cohort)
    if not runs:
        raise SystemExit(f"no {config_encoder.PRED_FILE[a.cohort]} found under "
                         f"{config_encoder.OUT_DIR} -- train the heads first")
    print(f"evaluating {len(runs)} run(s) on {a.cohort}, "
          f"{a.n_boot} bootstrap draws")
    rows = []
    for enc, lab in runs:
        rows += evaluate_run(enc, lab, a.cohort, a.n_boot)

    df = pd.DataFrame(rows)
    out = os.path.join(config_encoder.OUT_DIR, f"summary_{a.cohort}.csv")
    df.to_csv(out, index=False)
    rank = df[df["kind"] == "ranking"]
    rank.to_csv(os.path.join(config_encoder.OUT_DIR,
                             f"summary_{a.cohort}_ranking.csv"), index=False)
    print(f"\nwrote {out}  ({len(df):,} rows, {len(rank):,} ranking)")


if __name__ == "__main__":
    main()
