"""
Experiment 5 external validation: score SBUH 2024-2025 with the FROZEN models.

This step fits NOTHING. For each (model, label) it loads

    results/tabular/<model>/<label>/model/<artifact>          the model
    results/tabular/<model>/<label>/model/prep_state.joblib   the preprocessing

applies the preprocessing to sbuh_X_test with apply_state -- no medians, no
category vocabularies, no one-hot layout recomputed from SBUH -- predicts, and
writes the raw scores next to the PUF ones as sbuh_pred.parquet. That is exactly
what another hospital would receive: a frozen model plus a frozen transform.
Refitting either on SBUH would answer a different and much easier question.

Thresholds are not fitted here either, and none are stored in this file:
evaluate.py derives every operating point from the saved scores, using only
rules that are legal on an external cohort (a top-k% capacity cut taken with the
same k in both cohorts, and one F1 threshold fitted on the PUF validation split
and applied unchanged).

XGBoost needs one extra step: train.py saves the full booster while
XGBClassifier.predict_proba truncates to the early-stopped tree count, so the
saved best_iteration is replayed here. Without it SBUH would be scored by a
different model than the PUF test set was.

Usage:  python external.py [--cohort sbuh] [--split test] [--models ...] [--labels ...]
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import config_tabular
import data_tabular


def predict(model, rundir, rep, state):
    art = os.path.join(rundir, config_tabular.ARTIFACT_FOR[model])

    if model in ("logistic_regression", "random_forest"):
        return joblib.load(art).predict_proba(rep)[:, 1]

    if model == "xgboost":
        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(art)
        with open(os.path.join(rundir, "train_meta.json")) as f:
            best = json.load(f).get("best_iteration", -1)
        dm = xgb.DMatrix(rep, enable_categorical=True)
        # A bare Booster ignores early stopping; XGBClassifier.predict_proba does
        # not. Truncate to the same trees or SBUH is scored by a different model.
        if best is not None and best >= 0:
            return bst.predict(dm, iteration_range=(0, int(best) + 1))
        return bst.predict(dm)

    if model == "lightgbm":
        import lightgbm as lgb
        # save_model already truncated to best_iteration_.
        return lgb.Booster(model_file=art).predict(rep)

    import torch
    import train_torch as T
    xnum, xcat = rep
    net = (T.EmbedMLP(xnum.shape[1], state["cardinalities"]) if model == "mlp"
           else T.FTTransformer(xnum.shape[1], state["cardinalities"]))
    net.load_state_dict(torch.load(art, map_location="cpu"))
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xnum), 4096):
            out.append(torch.sigmoid(
                net(torch.from_numpy(xnum[i:i + 4096]),
                    torch.from_numpy(xcat[i:i + 4096]))).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=config_tabular.EXT_COHORT)
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", default=None, help="comma-separated subset")
    ap.add_argument("--labels", default=None, help="comma-separated subset")
    args = ap.parse_args()

    from sklearn.metrics import average_precision_score, roc_auc_score

    models = args.models.split(",") if args.models else config_tabular.MODELS
    labels = args.labels.split(",") if args.labels else config_tabular.LABELS
    pred_file = config_tabular.PRED_FILE[args.cohort]

    keep = data_tabular.model_features()
    X = data_tabular.load_X(args.cohort, args.split, keep)
    print(f"[external] {args.cohort}_{args.split}: {X.shape[0]:,} rows x "
          f"{X.shape[1]} model columns", flush=True)

    rows = []
    for label in labels:
        y = data_tabular.load_y(args.cohort, args.split, label)
        if len(y) != len(X):
            raise SystemExit(f"{label}: Y has {len(y)} rows, X has {len(X)}")
        cache = {}
        for model in models:
            rundir = config_tabular.out_dir(model, label)
            art = os.path.join(rundir, config_tabular.ARTIFACT_FOR[model])
            prep = os.path.join(rundir, "model", "prep_state.joblib")
            if not (os.path.exists(art) and os.path.exists(prep)):
                print(f"[external] SKIP {model}/{label}: missing artifact or "
                      f"prep state", flush=True)
                continue

            view = config_tabular.VIEW_FOR[model]
            if view not in cache:
                # One prep state per (label, view): xgboost/lightgbm share the
                # tree view and mlp/ft_transformer share embed, so the transform
                # runs twice per label rather than six times. The states are
                # loaded from each model's own directory and asserted equal by
                # verify.py, not assumed equal here.
                st = data_tabular.load_prep(prep)
                cache[view] = (st, data_tabular.apply_prep(st, X))
            state, rep = cache[view]

            s = np.asarray(predict(model, rundir, rep, state), dtype=np.float64)
            pd.DataFrame({"y_true": np.asarray(y).astype(int),
                          "score_raw": s}).to_parquet(
                os.path.join(rundir, pred_file), index=False)

            meta_p = os.path.join(rundir, "external_meta.json")
            with open(meta_p, "w") as f:
                json.dump({"model": model, "label": label, "experiment": 5,
                           "cohort": f"{args.cohort}_{args.split}",
                           "pred_file": pred_file,
                           "n": int(len(y)), "n_pos": int(y.sum()),
                           "prevalence": float(y.mean()),
                           "inference_only": True,
                           "note": "frozen PUF model + frozen PUF preprocessing "
                                   "state; nothing -- model, transform or "
                                   "threshold -- was fitted on this cohort"},
                          f, indent=2)

            r = {"model": model, "label": label, "n": int(len(s)),
                 "n_pos": int(y.sum()), "prevalence": float(y.mean())}
            if 0 < y.sum() < len(y):
                r["auroc"] = float(roc_auc_score(y, s))
                r["auprc"] = float(average_precision_score(y, s))
            rows.append(r)
            print(f"[external] {model:20s} {label:13s} n={len(s):,} "
                  f"pos={int(y.sum()):4d}  AUROC={r.get('auroc', float('nan')):.4f} "
                  f"AUPRC={r.get('auprc', float('nan')):.4f}", flush=True)

    p = os.path.join(config_tabular.OUT_DIR, f"inference_summary_{args.cohort}.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"\n[external] {len(rows)} cells -> {p}")
    print(f"[external] point estimates only; CIs and operating points come from "
          f"evaluate.py --cohort {args.cohort}")


if __name__ == "__main__":
    main()
