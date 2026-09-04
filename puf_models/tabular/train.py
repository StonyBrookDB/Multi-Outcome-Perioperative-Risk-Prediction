"""
Experiment 5 training. One (model, label) cell per invocation.

    python train.py --model xgboost --label mortality
    python train.py --model mlp                       # all four outcomes

Per cell this writes into results/tabular/<model>/<label>/:

    model/<artifact>         the fitted model itself              [REQUIREMENT 1]
    model/prep_state.joblib  the fitted preprocessing state       [REQUIREMENT 2]
    split_indices.parquet    which training row went to fit / val [REQUIREMENT 3]
    val_pred.parquet         y_true + score_raw, validation split
    test_pred.parquet        y_true + score_raw, PUF 2024 test year
    train_meta.json          split sizes, prevalences, timings, hyperparameters
    DONE                     written last; the resume check keys off it

plus a shared copy of the preprocessing state at
results/tabular/prep/<label>__<view>.joblib, in the layout feature_alignment/
expects. (sbuh_pred.parquet lands in the same cell directory, written later by
external.py -- training never touches SBUH.)

Those three requirements are what make the frozen bundle scoreable on a new
cohort without the original 5M-row fit split.

`score_raw` is the uncalibrated model output at float64, saved exactly as the
model produced it. Every family here trains with class re-weighting, so that
number RANKS patients but is not a risk -- do not read it as a probability.
There is no calibration step by design. Nothing is thresholded and no metric is
computed here: evaluate.py derives everything from these files, so a new metric
never costs a retrain.

Neural nets checkpoint after every epoch and resume from the checkpoint, so a
task killed by a 2-hour gpu_short wall clock picks up where it stopped instead
of starting over.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import config_tabular
import data_tabular


def _pos_weight(y):
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / max(pos, 1.0)


# ── sklearn families ─────────────────────────────────────────────────────────

def fit_sklearn(model, reps, y_fit, y_val, outdir):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    clf = (LogisticRegression(**config_tabular.LR_PARAMS) if model == "logistic_regression"
           else RandomForestClassifier(**config_tabular.RF_PARAMS))
    # Neither family early-stops, so the validation rows are simply not used for
    # fitting. They are still scored, so every model has a val_pred on file (the
    # transferable F1 threshold is read off it) and all six sit on identical
    # fit rows.
    clf.fit(reps["fit"], y_fit)

    joblib.dump(clf, os.path.join(outdir, "model", "model.joblib"), compress=3)
    extra = {"uses_validation_for_early_stopping": False,
             "class_reweighting": "class_weight=balanced"}
    if model == "logistic_regression":
        # saga at max_iter=200 does not always reach tol on this many rows. That
        # limit is inherited from ../config.py and is left alone so the LogReg
        # baseline stays comparable across experiments -- but a coefficient
        # vector that stopped early is a fact about the result, so record it
        # here instead of leaving it in a warning nobody reads.
        n_iter = int(np.max(clf.n_iter_))
        extra.update({"lr_n_iter": n_iter,
                      "lr_max_iter": int(config_tabular.LR_PARAMS["max_iter"]),
                      "lr_converged": bool(n_iter < config_tabular.LR_PARAMS["max_iter"])})
        if not extra["lr_converged"]:
            print(f"    NOTE: LogisticRegression hit max_iter="
                  f"{extra['lr_max_iter']} without reaching tol="
                  f"{config_tabular.LR_PARAMS['tol']}", flush=True)

    p = lambda X: clf.predict_proba(X)[:, 1]  # noqa: E731
    return p(reps["val"]), p(reps["test"]), extra


# ── gradient-boosted trees ───────────────────────────────────────────────────

def fit_gbdt(model, reps, y_fit, y_val, cat_cols, outdir):
    spw = _pos_weight(y_fit)

    if model == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(enable_categorical=True, scale_pos_weight=spw,
                            **config_tabular.XGB_PARAMS)
        clf.fit(reps["fit"], y_fit, eval_set=[(reps["val"], y_val)], verbose=50)
        clf.get_booster().save_model(os.path.join(outdir, "model", "xgboost.json"))
        # best_iteration MUST be recorded: XGBClassifier.predict_proba truncates
        # to the early-stopped tree count, a bare Booster does not. Scoring a new
        # site through the raw Booster without this number silently uses all 800
        # trees and disagrees with everything reported here.
        extra = {"scale_pos_weight": round(spw, 2),
                 "best_iteration": int(getattr(clf, "best_iteration", -1)),
                 "n_trees_saved": int(config_tabular.XGB_PARAMS["n_estimators"])}
    else:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        clf = LGBMClassifier(scale_pos_weight=spw, **config_tabular.LGBM_PARAMS)
        clf.fit(reps["fit"], y_fit, eval_set=[(reps["val"], y_val)],
                eval_metric="average_precision",
                categorical_feature=list(cat_cols),
                callbacks=[early_stopping(50, first_metric_only=True, verbose=False),
                           log_evaluation(100)])
        # save_model defaults to num_iteration=best_iteration_, so the saved
        # booster is already truncated -- no replay parameter needed at scoring.
        clf.booster_.save_model(os.path.join(outdir, "model", "lightgbm.txt"))
        best = int(clf.best_iteration_ or -1)
        # A GBM that early-stopped in the first handful of rounds has not been
        # trained; it is one tree wearing a 1500-tree label. That is exactly what
        # the untracked binary_logloss metric used to cause here, so fail loudly
        # rather than write a plausible-looking artifact.
        if 0 <= best < 20:
            raise SystemExit(
                f"lightgbm/{label}: early stopping fired at iteration {best} of "
                f"{config_tabular.LGBM_PARAMS['n_estimators']}; the tracked metrics were "
                f"{list(clf.evals_result_.get('valid_0', {}))}. Expected "
                f"average_precision only -- check config_tabular.LGBM_PARAMS['metric'].")
        extra = {"scale_pos_weight": round(spw, 2), "best_iteration": best,
                 "eval_metrics_tracked": list(clf.evals_result_.get("valid_0", {}))}

    p = lambda X: clf.predict_proba(X)[:, 1]  # noqa: E731
    extra.update({"uses_validation_for_early_stopping": True,
                  "class_reweighting": "scale_pos_weight"})
    return p(reps["val"]), p(reps["test"]), extra


# ── neural nets ──────────────────────────────────────────────────────────────

def _torch_loader(rep, y, batch, shuffle, device):
    """DataLoader over (X_num, X_cat, y) that hands the dataset whole batches.

    The stock per-sample TensorDataset path collates one row at a time, which at
    ~4.7M rows per epoch costs more than the forward pass -- badly so on CPU,
    where the tabular arm has to run if no GPU frees up. Indexing the tensors with a batch
    of indices at once is the same data in the same order, without the per-row
    Python overhead.
    """
    import torch
    xn = torch.from_numpy(rep[0])
    xc = torch.from_numpy(rep[1])
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(xn)

        def __getitem__(self, i):
            return xn[i], xc[i], yt[i]

    inner = (torch.utils.data.RandomSampler(range(len(xn))) if shuffle
             else torch.utils.data.SequentialSampler(range(len(xn))))
    sampler = torch.utils.data.BatchSampler(inner, batch_size=batch, drop_last=False)
    return torch.utils.data.DataLoader(
        _DS(), sampler=sampler, batch_size=None,
        num_workers=0, pin_memory=(device.type == "cuda"))


def _torch_predict(net, loader, device):
    import torch
    net.eval()
    out = []
    with torch.no_grad():
        for xn, xc, _ in loader:
            xn = xn.to(device, non_blocking=True)
            xc = xc.to(device, non_blocking=True)
            out.append(torch.sigmoid(net(xn, xc)).float().cpu().numpy())
    return np.concatenate(out)


def fit_torch(model, reps, y_fit, y_val, cardinalities, outdir, resume=True):
    import torch
    from sklearn.metrics import roc_auc_score

    import train_torch as T   # ../train_torch.py: EmbedMLP / FTTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(config_tabular.N_THREADS)
    print(f"    device={device} threads={torch.get_num_threads()}", flush=True)

    n_num = reps["fit"][0].shape[1]
    net = (T.EmbedMLP(n_num, cardinalities) if model == "mlp"
           else T.FTTransformer(n_num, cardinalities)).to(device)

    tr = config_tabular.TORCH_TRAIN
    batch = (config_tabular.FT_PARAMS["batch_size"] if model == "ft_transformer"
             else tr["batch_size"])
    opt = torch.optim.AdamW(net.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])

    pos, neg = float((y_fit == 1).sum()), float((y_fit == 0).sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))

    # ── resume ───────────────────────────────────────────────────────────────
    # Epoch-level checkpoint. gpu_short caps a job at 2 hours; without this an
    # FT-Transformer that needs longer could never finish there, and every
    # requeue would throw away the epochs already paid for.
    ckpt_path = os.path.join(outdir, "model", "ckpt.pt")
    start_epoch, best_auc, best_epoch, bad, best_state = 1, -1.0, 0, 0, None
    if resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        best_auc, best_epoch, bad = ck["best_auc"], ck["best_epoch"], ck["bad"]
        best_state = ck["best_state"]
        print(f"    resumed after epoch {ck['epoch']} "
              f"(best val_auroc={best_auc:.4f} @ {best_epoch})", flush=True)

    train_loader = _torch_loader(reps["fit"], y_fit, batch, True, device)
    val_loader = _torch_loader(reps["val"], y_val, batch, False, device)

    # `epoch` must exist even if the loop never runs -- a task requeued after its
    # final epoch resumes with start_epoch > max_epochs and goes straight to
    # saving.
    stopped_early, epoch = False, start_epoch - 1
    for epoch in range(start_epoch, tr["max_epochs"] + 1):
        t0 = time.time()
        net.train()
        for xn, xc, yb in train_loader:
            xn = xn.to(device, non_blocking=True)
            xc = xc.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss_fn(net(xn, xc), yb).backward()
            opt.step()
        auc = float(roc_auc_score(y_val, _torch_predict(net, val_loader, device)))
        print(f"    epoch {epoch:2d}  val_auroc={auc:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

        if auc > best_auc + 1e-4:
            best_auc, bad, best_epoch = auc, 0, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
        else:
            bad += 1

        torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "best_auc": best_auc,
                    "best_epoch": best_epoch, "bad": bad,
                    "best_state": best_state}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

        if bad >= tr["patience"]:
            print(f"    early stop (best val_auroc={best_auc:.4f} @ epoch "
                  f"{best_epoch})", flush=True)
            stopped_early = True
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    torch.save(net.state_dict(), os.path.join(outdir, "model", "state_dict.pt"))

    def pred(rep):
        return _torch_predict(
            net, _torch_loader(rep, np.zeros(len(rep[0]), np.float32),
                               batch, False, device), device)

    return (pred(reps["val"]), pred(reps["test"]),
            {"uses_validation_for_early_stopping": True,
             "class_reweighting": "BCEWithLogitsLoss pos_weight",
             "best_val_auroc": round(best_auc, 6), "best_epoch": best_epoch,
             "epochs_run": epoch, "early_stopped": stopped_early,
             "device": device.type})


# ── driver ───────────────────────────────────────────────────────────────────

def run(model, label, resume=True):
    outdir = config_tabular.out_dir(model, label)
    os.makedirs(os.path.join(outdir, "model"), exist_ok=True)
    view = config_tabular.VIEW_FOR[model]
    keep = data_tabular.model_features()

    t_load = time.time()
    X_train = data_tabular.load_X("puf", "train", keep)
    y_train = data_tabular.load_y("puf", "train", label)
    if len(X_train) != len(y_train):
        raise SystemExit(f"X_train {len(X_train)} != Y_train {len(y_train)} rows")

    fit_idx, val_idx = data_tabular.split_indices(y_train, label)
    pd.DataFrame({"row": np.concatenate([fit_idx, val_idx]),
                  "split": ["fit"] * len(fit_idx) + ["val"] * len(val_idx)}
                 ).to_parquet(os.path.join(outdir, "split_indices.parquet"),
                              index=False)
    y_fit, y_val = y_train[fit_idx], y_train[val_idx]
    print(f"  split  fit={len(fit_idx):,} val={len(val_idx):,}  "
          f"prevalence fit={y_fit.mean():.5f} val={y_val.mean():.5f}", flush=True)

    # ── preprocessing: fit ONCE on the fit split, then apply-only everywhere ──
    t0 = time.time()
    X_fit = X_train.iloc[fit_idx].reset_index(drop=True)
    state = data_tabular.fit_prep(view, X_fit)
    data_tabular.save_prep(state, os.path.join(outdir, "model", "prep_state.joblib"))
    data_tabular.save_prep(state, config_tabular.prep_path(label, view))

    reps = {"fit": data_tabular.apply_prep(state, X_fit)}
    del X_fit
    reps["val"] = data_tabular.apply_prep(state, X_train.iloc[val_idx].reset_index(drop=True))
    n_train_cols = X_train.shape[1]
    del X_train

    X_test = data_tabular.load_X("puf", "test", keep)
    y_test = data_tabular.load_y("puf", "test", label)
    reps["test"] = data_tabular.apply_prep(state, X_test)
    del X_test
    prep_seconds = time.time() - t0
    print(f"  prep   view={view} in {prep_seconds:.0f}s "
          f"(load+prep {time.time() - t_load:.0f}s)", flush=True)

    t0 = time.time()
    if model in ("logistic_regression", "random_forest"):
        s_val, s_te, extra = fit_sklearn(model, reps, y_fit, y_val, outdir)
    elif model in ("xgboost", "lightgbm"):
        s_val, s_te, extra = fit_gbdt(model, reps, y_fit, y_val,
                                      state["cat_cols"], outdir)
    else:
        s_val, s_te, extra = fit_torch(model, reps, y_fit, y_val,
                                       state["cardinalities"], outdir, resume)
    train_seconds = time.time() - t0

    # float64, untransformed: these files are the only thing standing between a
    # new metric and a retrain, so nothing is rounded, clipped or rescaled.
    for name, y_, s_ in (("test", y_test, s_te), ("val", y_val, s_val)):
        pd.DataFrame({"y_true": np.asarray(y_).astype(int),
                      "score_raw": np.asarray(s_, dtype=np.float64)}).to_parquet(
            os.path.join(outdir, f"{name}_pred.parquet"), index=False)

    info = {"model": model, "label": label, "experiment": 5, "view": view,
            "n_fit": int(len(fit_idx)), "n_val": int(len(val_idx)),
            "n_test": int(len(y_test)),
            "fit_frac": config_tabular.FIT_FRAC, "val_frac": config_tabular.VAL_FRAC,
            "prevalence_fit": float(y_fit.mean()),
            "prevalence_val": float(y_val.mean()),
            "prevalence_test": float(y_test.mean()),
            "n_features_model": int(len(keep)),
            "n_features_file": int(n_train_cols),
            "n_prep_columns": len(state["feature_names"]),
            "artifact": config_tabular.ARTIFACT_FOR[model],
            "prep_state": "model/prep_state.joblib",
            "prep_state_shared": os.path.relpath(
                config_tabular.prep_path(label, view), config_tabular.OUT_DIR),
            "calibrated": False,
            "score_note": "uncalibrated, class-reweighted: ranks patients, "
                          "is NOT a risk probability",
            "seed": config_tabular.SEED,
            "prep_seconds": round(prep_seconds, 1),
            "train_seconds": round(train_seconds, 1)}
    info.update(extra)
    with open(os.path.join(outdir, "train_meta.json"), "w") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(outdir, "DONE"), "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))

    print(f"[{model} / {label}] saved  fit={info['n_fit']:,} val={info['n_val']:,} "
          f"test={info['n_test']:,}  ({info['train_seconds']}s)", flush=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=config_tabular.MODELS)
    ap.add_argument("--label", default=None, choices=config_tabular.LABELS,
                    help="omit to run all four outcomes sequentially")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing neural-net checkpoint and restart")
    ap.add_argument("--force", action="store_true",
                    help="rerun a cell that already has a DONE marker")
    args = ap.parse_args()

    np.random.seed(config_tabular.SEED)
    try:
        import torch
        torch.manual_seed(config_tabular.SEED)
    except ImportError:
        pass

    for label in ([args.label] if args.label else config_tabular.LABELS):
        if os.path.exists(os.path.join(config_tabular.out_dir(args.model, label), "DONE")) \
                and not args.force:
            print(f"=== {args.model} / {label}: already DONE, skipping "
                  f"(--force to rerun) ===", flush=True)
            continue
        print(f"\n=== {args.model} / {label} ===", flush=True)
        run(args.model, label, resume=not args.no_resume)


if __name__ == "__main__":
    main()
