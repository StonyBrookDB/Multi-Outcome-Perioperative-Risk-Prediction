# Predictor harmonization across NSQIP years and cohorts

Variable availability and coding changed across NSQIP annual files and between
the national and single-institution cohorts. This is the crosswalk: what
differed, and what was done about it. Values are the percentage of cases with a
recorded value in each cohort.

The prespecified predictor set was retained despite these differences.
Unavailable values are handled through the fitted preprocessing pipeline rather
than by dropping predictors.

| Difference | Variable(s) | PUF 2018–22 | PUF 2024 | Single-site | Harmonization / handling |
|---|---|---:|---:|---:|---|
| Discontinued | DYSPNEA, WNDINF, WTLOSS | 60.1 | 0.0 | 0.0 | Collection discontinued after 2020; retained in the predictor set, unavailable values handled as missing |
| Introduced or expanded | OXYGEN_SUPPORT, PREOP_COVID | 40.0 | 100.0 | 100.0 | Introduced mid-window; earlier-year values handled as missing |
| | RENAFAIL | 80.3 | 100.0 | 100.0 | Availability increased during the study period |
| | HXDEMENTIA, HXFALL | 6.2 | 16.7 | 21.9 | Increased over time but remained limited |
| | HOMESUP | 5.8 | 15.6 | 21.1 | Increased over time but remained limited |
| Recoded across eras | CASETYPE, EMERGNCY, ELECTSURG | 100.0 | 100.0 | 100.0 | Mapped to a common coding scheme across annual files and retained as distinct predictors |
| | HEMO, PRHEMO_A1C | 10.1 | 30.3 | 35.0 | Harmonized across coding eras; availability increased over time |
| Cross-cohort measurement | PRHEMOGLOBIN | 83.7 | 84.0 | 93.4 | Derived from PRHCT in PUF; directly measured at the single site\* |
| Very low availability | PREOP_CREAT_MSINCR | 0.1 | 0.2 | 0.2 | Retained; observed values present in the training years |
| | IMMUNO_CAT | 1.7 | 4.6 | 6.4 | Retained; observed values present in the training years |
| No training values | DPRHEMOGLOBIN, DPRHEMO_A1C, PRPT, DPRPT | 0.0 | 0.0 | 0.0 | Not instantiated as model inputs |

\* In PUF, `PRHEMOGLOBIN` is approximated from the recorded haematocrit as
`0.344 * PRHCT - 0.56` (R² = 0.74, fitted on 169,354 paired 2023 observations);
at the single institution it is measured directly.

## Files

| File | Contents |
|---|---|
| [`harmonization_table.csv`](harmonization_table.csv) | the table above, machine-readable |
| [`puf73_build_spec.csv`](puf73_build_spec.csv) | all 73 prespecified variables: which annual files they appear in, and how each was built |

`puf73_build_spec.csv` columns:

| Column | Meaning |
|---|---|
| `name` | PUF variable name |
| `availability` | `present in all years`, `absent in some years`, `partially derivable`, `fully derivable` |
| `years_present` | operative years in which the raw variable appears |
| `n_years` | count of those years |
| `build_action` | `read directly` (52 variables), `missing years left as NaN` (15), `missing years approximated by rule` (4), `missing years derived by rule` (2) |
| `rule` | the derivation, where one applies |

Six variables carry an explicit derivation rule: `CASETYPE`, `ELECTSURG`,
`EMERGNCY`, `PRHEMOGLOBIN`, `HEMO` and `PRHEMO_A1C`.

Sixty-nine of the 73 have observed values in the training years and are
instantiated as tabular model inputs. All 73 field slots are serialized to text
for the encoder arm, with the remaining four written as `undefined`.
