# Multi-Outcome Perioperative Risk Prediction

Model source code for *From National Data to Local Evaluation: Benchmarking
Machine Learning Models for Perioperative Risk Prediction*.

Nine modelling configurations are benchmarked for four 30-day postoperative
outcomes — unplanned readmission, unplanned return to the operating room,
mortality, and any complication — holding one prespecified preoperative
predictor set fixed across all of them.

| Arm | Configurations |
|---|---|
| [`puf_models/tabular`](puf_models/tabular) | Logistic regression, random forest, XGBoost, LightGBM, multilayer perceptron, FT-Transformer |
| [`puf_models/text_encoder`](puf_models/text_encoder) | MedEmbed-large-v0.1, ClinicalBERT, Bio\_ClinicalBERT — each frozen, with the tabular arm's multilayer perceptron as the downstream classifier |

Development used 4,995,670 ACS NSQIP PUF cases from 2018–2022; temporal
evaluation used 963,565 cases from 2024, with 2023 held out as a prespecified
one-year gap. A single-institution cohort of 3,490 cases was scored with the
same frozen models as a deployment check.

## Data availability

**No data is included in this repository, and none may be added to it.**

The ACS NSQIP Participant Use File is distributed by the American College of
Surgeons under a Data Use Agreement and is not redistributable by the authors.
Obtain it directly from ACS. The single-institution cohort is not shareable.
`.gitignore` keeps data, model outputs and logs out of version control; please
keep it that way.

Both arms expect a prepared 73-column feature build:

```
puf_X_train.parquet      development rows (2018–2022)
puf_X_test.parquet       temporal test rows (2024)
sbuh_X_test.parquet      single-institution cohort
features/features_model_69.csv
```

Sixty-nine of the 73 prespecified variables have observed values and are
instantiated as tabular model inputs; all 73 field slots are serialized to text
for the encoder arm, with the remaining four written as `undefined`.

Variable availability and coding changed across NSQIP annual files and between
cohorts. [`docs/harmonization.md`](docs/harmonization.md) is the crosswalk, with
machine-readable copies in `docs/harmonization_table.csv` and
`docs/puf73_build_spec.csv`.

## Layout

```
puf_models/
  config.py            shared paths and hyperparameter dictionaries
  data.py              data loading used by the neural models
  metrics.py           AUROC / AUPRC and bootstrap helpers
  train_torch.py       EmbedMLP and FTTransformer definitions
  tabular/             six tabular families — training, evaluation, jobs
  text_encoder/        three frozen encoders — serialization, chunking,
                       embedding, downstream heads
feature_alignment/
  lib/prep_apply.py    apply-only preprocessing; fitted parameters are never
                       re-fitted on evaluation data
docs/
  harmonization.md     predictor availability and coding differences across
                       NSQIP years and cohorts, and how each was handled
  Table S1_TRIPOD+AI Checklist.docx
                       completed TRIPOD+AI reporting checklist for the
                       manuscript
```

Neither arm is self-contained. Both add the repository root to `sys.path` and
import `config.py`; `tabular/data_tabular.py` imports `prep_apply` from
`feature_alignment/lib`; and `text_encoder/evaluate_encoder.py` imports the
tabular arm's `evaluate.py` so that both arms are scored by the same code. Keep
the two top-level directories side by side.

## Configuration

Every path is an environment variable with a default; override the variables
rather than editing the source. `tabular/env.sh` and `text_encoder/env.sh` set
them and fail loudly if an expected input is missing.

| Variable | Meaning |
|---|---|
| `TABULAR_DATA_DIR`, `ENCODER_DATA_DIR` | the 73-column feature build |
| `TABULAR_ALIGN_DIR` | directory containing `lib/prep_apply.py` |
| `TABULAR_OUT_DIR`, `ENCODER_OUT_DIR` | where model outputs are written |
| `ENCODER_TABULAR_OUT_DIR` | tabular outputs; the encoder arm reuses their fit/validation split |
| `TABULAR_XGB_DEVICE` | `cpu` or `cuda` for XGBoost |
| `TABULAR_PYTHON`, `ENCODER_PYTHON` | interpreter for each arm |
| `HF_HOME` | HuggingFace cache; the encoder arm runs with `HF_HUB_OFFLINE=1` |

The `#SBATCH --output=` paths inside the job scripts are absolute and point at
the authors' cluster. SLURM does not expand shell variables in `#SBATCH`
directives, so these must be edited by hand for another site.

## Running

Both arms are driven by SLURM array jobs.

```bash
source puf_models/tabular/env.sh
bash puf_models/tabular/submit.sh            # 6 families x 4 outcomes

source puf_models/text_encoder/env.sh
python puf_models/text_encoder/fetch_models.py   # login node only; pins commit SHAs
bash puf_models/text_encoder/submit.sh           # 3 encoders x 4 outcomes
```

The tabular arm must finish first: the encoder arm reads its saved per-outcome
fit/validation split rather than deriving its own, which is what keeps the two
sets of rows comparable.

`fetch_models.py` must run where huggingface.co is reachable. It pins each
encoder to an exact commit, because authors update weights in place under an
unchanged name:

| Encoder | HuggingFace ID | Commit |
|---|---|---|
| MedEmbed-large-v0.1 | `abhinand/MedEmbed-large-v0.1` | `963121b` |
| ClinicalBERT | `medicalai/ClinicalBERT` | `f7c7f65` |
| Bio\_ClinicalBERT | `emilyalsentzer/Bio_ClinicalBERT` | `d5892b3` |

## Verification

Hyperparameters were prespecified, not searched: one configuration per family,
fixed before any model was fit and identical across the four outcomes. The
validation split is used only for early stopping.

`tabular/verify.py` checks that every model object and every fitted
preprocessing state is on disk and loads, and that all six families share an
identical fit split per outcome.

The encoder arm verifies itself before any embedding is generated:

- `gate_check.py` — zero truncation on all three cohorts, throughput, label
  transcription against the ACS NSQIP data dictionary, end-to-end dry run
- `verify.py` — `config_sha256` over `config_encoder.py`, `labels73.py` and
  `chunk_plan.json`, recorded in every shard marker and every `train_meta.json`
- `spotcheck.py` — independent recomputation of sampled embeddings

## Citation

Citation details will be added once the manuscript is published.
