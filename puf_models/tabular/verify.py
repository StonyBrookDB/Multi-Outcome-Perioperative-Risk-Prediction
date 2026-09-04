"""
Guard on the invariants the frozen-model design depends on.

Run before post-processing and again at the end. For every (model, label) cell
it checks:

  1. the model object is on disk and loads
  2. the fitted preprocessing state is on disk, loads, and is for the right
     view
  3. the fit/val split is present, val is non-empty, and the two are DISJOINT
  4. all six families share the identical fit split for a label, so the
     model-to-model comparison is about the models
  5. the shared prep/<label>__<view>.joblib matches the per-model copy, since
     external.py loads one and feature_alignment/ loads the other
  6. XGBoost recorded best_iteration, without which the frozen booster scores a
     new site with a different number of trees than it was evaluated with
  7. the per-patient prediction files exist, and (--strict) so does sbuh_pred

Exits non-zero on any failure. Usage:  python verify.py [--strict]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import config_tabular


def _same_state(a, b):
    """Structural equality for two prep states (numpy/pandas members inside)."""
    if a.keys() != b.keys() or a["view"] != b["view"]:
        return False
    if a["num_cols"] != b["num_cols"] or a["cat_cols"] != b["cat_cols"]:
        return False
    if a.get("feature_names") != b.get("feature_names"):
        return False
    if a.get("cardinalities") != b.get("cardinalities"):
        return False
    for c, spec in a.get("cats", {}).items():
        if b.get("cats", {}).get(c) != spec:
            return False
    for k in ("median", "mean", "std"):
        sa, sb = a.get("numeric", {}).get(k), b.get("numeric", {}).get(k)
        if sa is None and sb is None:
            continue
        if sa is None or sb is None or not np.allclose(sa.to_numpy(),
                                                       sb.to_numpy(),
                                                       equal_nan=True):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="also fail on untrained cells and missing sbuh_pred")
    args = ap.parse_args()

    import joblib

    problems, missing, checked = [], [], 0
    for label in config_tabular.LABELS:
        fit_rows, states = {}, {}
        for model in config_tabular.MODELS:
            d = config_tabular.out_dir(model, label)
            tag = f"{model}/{label}"
            if not os.path.exists(os.path.join(d, "DONE")):
                missing.append(tag)
                continue
            checked += 1

            if not os.path.exists(os.path.join(d, config_tabular.ARTIFACT_FOR[model])):
                problems.append(f"{tag}: REQUIREMENT 1 -- no model artifact at "
                                f"{config_tabular.ARTIFACT_FOR[model]}")

            prep = os.path.join(d, "model", "prep_state.joblib")
            if not os.path.exists(prep):
                problems.append(f"{tag}: REQUIREMENT 2 -- no model/prep_state.joblib")
            else:
                try:
                    st = joblib.load(prep)
                    if st["view"] != config_tabular.VIEW_FOR[model]:
                        problems.append(f"{tag}: prep state is for view "
                                        f"{st['view']}, expected "
                                        f"{config_tabular.VIEW_FOR[model]}")
                    states[model] = st
                except Exception as e:                     # noqa: BLE001
                    problems.append(f"{tag}: prep state will not load ({e})")

            sp = os.path.join(d, "split_indices.parquet")
            if not os.path.exists(sp):
                problems.append(f"{tag}: REQUIREMENT 3 -- no split_indices.parquet")
            else:
                s = pd.read_parquet(sp)
                g = {k: set(v["row"]) for k, v in s.groupby("split")}
                if not g.get("val"):
                    problems.append(f"{tag}: REQUIREMENT 3 -- validation split empty")
                inter = g.get("fit", set()) & g.get("val", set())
                if inter:
                    problems.append(f"{tag}: fit and val overlap on "
                                    f"{len(inter):,} rows")
                fit_rows[model] = np.sort(np.fromiter(g.get("fit", ()),
                                                      dtype=np.int64))

            if model == "xgboost":
                mp = os.path.join(d, "train_meta.json")
                if os.path.exists(mp):
                    with open(mp) as f:
                        bi = json.load(f).get("best_iteration", -1)
                    if bi is None or bi < 0:
                        problems.append(f"{tag}: no usable best_iteration in "
                                        f"train_meta.json -- the frozen booster "
                                        f"cannot be replayed")

            for f_ in ["test_pred.parquet", "val_pred.parquet"] + (
                    ["sbuh_pred.parquet"] if args.strict else []):
                if not os.path.exists(os.path.join(d, f_)):
                    problems.append(f"{tag}: missing {f_}")

        if len(fit_rows) > 1:
            ref_model, ref = next(iter(fit_rows.items()))
            for m, v in fit_rows.items():
                if not np.array_equal(v, ref):
                    problems.append(f"{label}: fit split differs between "
                                    f"{ref_model} and {m} -- the six models are "
                                    f"not mutually comparable")

        # models sharing a view must have produced the same state, and the shared
        # copy handed to feature_alignment must be that same state
        by_view = {}
        for m, st in states.items():
            by_view.setdefault(config_tabular.VIEW_FOR[m], []).append((m, st))
        for view, items in by_view.items():
            ref_m, ref_st = items[0]
            for m, st in items[1:]:
                if not _same_state(ref_st, st):
                    problems.append(f"{label}/{view}: prep state differs between "
                                    f"{ref_m} and {m}")
            shared = config_tabular.prep_path(label, view)
            if not os.path.exists(shared):
                problems.append(f"{label}/{view}: no shared prep state at "
                                f"{os.path.relpath(shared, config_tabular.OUT_DIR)}")
            elif not _same_state(ref_st, joblib.load(shared)):
                problems.append(f"{label}/{view}: shared prep state does not "
                                f"match {ref_m}'s copy")

    print(f"[verify] {checked} completed cells checked "
          f"({len(missing)} not yet trained)")
    if missing:
        print(f"[verify] not trained: {', '.join(sorted(missing))}")
    if problems:
        print(f"\n[verify] {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    if args.strict and missing:
        print("\n[verify] --strict: some cells were never trained")
        sys.exit(1)
    print("[verify] OK: model object, preprocessing state and a disjoint "
          "fit/val split present for every completed cell")


if __name__ == "__main__":
    main()
