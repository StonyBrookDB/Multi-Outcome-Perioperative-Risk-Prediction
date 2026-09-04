# Text-encoder arm

Three frozen encoders — MedEmbed-large-v0.1, ClinicalBERT and Bio\_ClinicalBERT
— turn each serialized patient record into one vector, and the tabular arm's
multilayer perceptron is trained on those vectors. No encoder weight is updated.

## Before you start

Run the tabular arm first. This arm reads its saved per-outcome fit/validation
split rather than deriving its own, and `env.sh` aborts if it is missing.

```bash
source env.sh
```

`env.sh` sets `ENCODER_DATA_DIR`, `ENCODER_OUT_DIR`,
`ENCODER_TABULAR_OUT_DIR`, `HF_HOME` and `HF_HUB_OFFLINE=1`, and fails loudly if
an expected input is absent.

## 1. Fetch the encoders

```bash
python fetch_models.py
```

Run this on a login node — compute nodes have no route to huggingface.co. It
downloads the three checkpoints into `$HF_HOME` and writes the resolved commit
SHA for each back into `config_encoder.py`. Nothing else will run until every
encoder has a pinned revision.

## 2. Run everything

```bash
bash submit.sh
```

`submit.sh` calls `sbatch` and does no work itself. It chains, in order:

| Stage | Script | What it does |
|---|---|---|
| gate | `run_gate.sbatch` | four pre-flight checks; refuses to continue if any fails |
| embed | `run_embed.sbatch` | array job, one task per encoder x cohort x shard |
| head | `run_head.sbatch` | trains the downstream MLP, all four outcomes per encoder |
| verify | `run_verify.sbatch` | config hash and schema checks over the finished run |
| eval | `run_eval.sbatch` | bootstrap CIs on both cohorts |

The gate stage is deliberate: `ENCODER_SKIP_GATE=1` overrides it, and you should
know why before you do.

## Running stages by hand

```bash
python gate_check.py                                   # pre-flight only
python embed.py --encoder medembed_large --cohort puf --split train \
                --shard 0 --n-shards 16
python train_head.py --encoder medembed_large --label all
python verify.py --all
python evaluate_encoder.py --cohort puf  --n-boot 1000
python evaluate_encoder.py --cohort sbuh --n-boot 1000
```

Encoder names are `medembed_large`, `clinicalbert`, `bio_clinicalbert`.
Outcome labels are `mortality`, `any_compl`, `unplanned_or`, `readmission`, or
`all`.

`ENCODER_FORCE=1` recomputes a shard or retrains a head that already carries a
DONE marker. `ENCODER_TRAIN_SHARDS` and `ENCODER_TEST_SHARDS` change the array
width.

## Reporting

```bash
python make_table.py     # merges both arms into one table
python make_figure.py    # AUROC figure, both cohorts
```

Both read saved summaries only and recompute nothing. `make_figure.py` expects
fonts under a path set at the top of the file; edit it for another site.

## Outputs

```
<ENCODER_OUT_DIR>/
  emb/<encoder>/<cohort>/<split>/    embedding shards + markers
  <encoder>/<label>/
    train_meta.json                  config_sha256, timings, best epoch
    puf_pred.parquet                 y_true, score_raw
    sbuh_pred.parquet
  summary_puf.csv
  summary_sbuh.csv
```

`score_raw` is the uncalibrated head output. The head trains with class
re-weighting, so it ranks patients but is not a risk — do not read it as a
probability.
