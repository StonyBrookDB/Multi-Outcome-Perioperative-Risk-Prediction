# Tabular arm

Six model families — logistic regression, random forest, XGBoost, LightGBM, a
multilayer perceptron and the FT-Transformer — trained on ACS NSQIP PUF
2018–2022 and reported twice: on the PUF 2024 test year, and, with the *same
frozen models*, on the single-institution 2024–2025 cohort. No other comparison
is made here.

```bash
source env.sh
bash submit.sh          # from a login node; sbatch is not on the compute nodes' PATH
```

## Design requirements

`verify.py` fails the job if any of these is broken.

1. **Every model object is serialised.** `model/model.joblib` (sklearn),
   `model/xgboost.json` plus `best_iteration` in `train_meta.json`,
   `model/lightgbm.txt`, `model/state_dict.pt` (torch).
2. **Every fitted preprocessing state is serialised**, and everything after the
   fit split goes through an apply-only transform — two copies:
   `<model>/<label>/model/prep_state.joblib` and a shared
   `prep/<label>__<view>.joblib`. This is what lets a frozen model score a new
   cohort without dragging the multi-million-row fit split along.
3. **The fit/validation split is saved per outcome and shared by all six
   families**, so a model-to-model comparison is about the models and not about
   the rows they happened to see.

## Data

Reads a prepared 73-column feature build; it never rebuilds or re-splits it.
Sixty-nine of the 73 prespecified variables have observed values and are
instantiated as model inputs. Required under `$TABULAR_DATA_DIR`:

```
puf_X_train.parquet      development rows (2018–2022)
puf_X_test.parquet       temporal test rows (2024)
sbuh_X_test.parquet      single-institution cohort
features/features_model_69.csv
```

Preprocessing is the fit-once/apply-everywhere pair in
`$TABULAR_ALIGN_DIR/lib/prep_apply.py`:

```python
state = fit_state(view, X_fit)      # once, on the fit split, and serialised
rep   = apply_state(state, X_any)   # everywhere else — fits nothing
```

## No calibration, and what follows from it

All six families train with class re-weighting (`class_weight="balanced"` /
`scale_pos_weight` / BCE `pos_weight`). `score_raw` therefore **ranks patients
but is not a risk**. Nothing here is calibrated, and Brier, log-loss, ECE and
calibration slope are consequently not reported — on a re-weighted raw scale
those numbers would look like calibration results without being any. For the
same reason there is no `risk >= 5%` style threshold anywhere; every operating
point is rank-based.

## Metrics

AUROC and AUPRC, identical in both summaries, with bootstrap 95% CIs (1,000
draws by default; `N_BOOT=2000 sbatch run_post.sbatch` for more). One shared set
of patient resamples per cell, so paired comparisons remain possible from the
saved draws.

## Hyperparameters

Prespecified, not searched: one configuration per family, fixed before any model
was fit and identical across the four outcomes. Values live in
`config_tabular.py`, inherited from `../config.py`. The validation split is used
for one purpose only — early stopping for the families that support it.

**LightGBM's early-stopping metric is named explicitly, and that matters.**
Under class re-weighting, validation binary log loss is best at the first
iteration and degrades thereafter. Left at its default, LightGBM tracks log loss
alongside average precision and stops at a single 127-leaf tree. Fixed by
setting `metric="average_precision"` in `LGBM_PARAMS` with
`first_metric_only=True`; `train.py` additionally aborts if LightGBM stops
inside the first 20 rounds, and records `eval_metrics_tracked`.

## Layout

```
<TABULAR_OUT_DIR>/
  <model>/<label>/
    model/            serialised model + prep_state.joblib
    train_meta.json   hyperparameters, timings, best_iteration
    puf_pred.parquet  test-year predictions (y_true, score_raw)
    sbuh_pred.parquet single-institution predictions, written by external.py
    split_indices.parquet
  prep/<label>__<view>.joblib
  summary_puf.csv
  summary_sbuh.csv
```

## Jobs

| Script | What it runs |
|---|---|
| `run_cpu.sbatch` | sklearn and gradient-boosting families, array over model x outcome |
| `run_cpu_torch.sbatch` | MLP / FT-Transformer when no GPU is free |
| `run_gpu.sbatch` | MLP / FT-Transformer on GPU |
| `run_post.sbatch` | evaluation, bootstrap CIs, summaries |

`submit.sh` sequences them. `TABULAR_FORCE=1` retrains a cell that already
carries a DONE marker.

`#SBATCH --output=` paths in the job scripts are absolute and point at the
authors' cluster; edit them for another site.

## Downstream

The frozen models and their preprocessing states are reused by `external.py` to
score the single-institution cohort, and by the text-encoder arm, which reads
this arm's per-outcome fit/validation split rather than deriving its own.
