"""
Feature availability across the three cohorts -- the report the the tabular arm brief
requires in writing rather than in code.

Three of the 69 model columns are populated in the training years and NOT
COLLECTED AT PREDICTION TIME:

    DYSPNEA, WNDINF, WTLOSS   60.1% populated in PUF 2018-2022, 0.0% in PUF
                              2024 and 0.0% in SBUH 2024-2025 -- ACS stopped
                              collecting them after 2020.

They stay in the model input because the clinician-signed list says 73 columns
and the only removals allowed are the 4 that are 100% empty in training. But a
model is then free to lean on a variable that will always be missing when the
model is actually used, and every reader of the results has to know that. This
script produces the evidence table; do not silently drop the columns instead.

Several columns move the other way -- OXYGEN_SUPPORT, PREOP_COVID, RENAFAIL,
HOMESUP, HXDEMENTIA, HXFALL are far better populated in 2024 than in the pooled
training years, because they were introduced mid-window. Same issue, opposite
sign: the fitted imputation defaults were set on a training pool where most rows
had no value.

Usage:  python feature_report.py
Writes: results/tabular/feature_availability.csv
        results/tabular/feature_availability.md
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import config_tabular
import data_tabular

FLAGGED = ["DYSPNEA", "WNDINF", "WTLOSS"]
SHIFT_PP = 25.0     # populated-rate gap, in points, worth flagging


def main():
    keep = data_tabular.model_features()
    # schema only -- reading all 73 columns of 5M rows just to list them would
    # cost ~7 GB for a column name lookup
    import pyarrow.parquet as pq
    all73 = pq.ParquetFile(
        os.path.join(config_tabular.DATA_DIR, "puf_X_train.parquet")).schema_arrow.names

    frames = {
        "puf_train": data_tabular.load_X("puf", "train", keep),
        "puf_test": data_tabular.load_X("puf", "test", keep),
        "sbuh_test": data_tabular.load_X("sbuh", "test", keep),
    }
    rows = []
    for c in keep:
        r = {"column": c, "in_model_69": True,
             "dtype": str(frames["puf_train"][c].dtype)}
        for name, X in frames.items():
            r[f"pct_populated_{name}"] = round(100.0 * X[c].notna().mean(), 2)
        r["train_minus_test_pp"] = round(
            r["pct_populated_puf_train"] - r["pct_populated_puf_test"], 2)
        r["unusable_at_prediction_time"] = bool(
            r["pct_populated_puf_test"] == 0.0 and r["pct_populated_puf_train"] > 0.0)
        r["large_shift"] = bool(abs(r["train_minus_test_pp"]) >= SHIFT_PP)
        rows.append(r)

    for c in [c for c in all73 if c not in keep]:
        rows.append({"column": c, "in_model_69": False, "dtype": "all-missing",
                     "pct_populated_puf_train": 0.0, "pct_populated_puf_test": 0.0,
                     "pct_populated_sbuh_test": 0.0, "train_minus_test_pp": 0.0,
                     "unusable_at_prediction_time": False, "large_shift": False})

    df = pd.DataFrame(rows).sort_values("train_minus_test_pp", ascending=False)
    os.makedirs(config_tabular.OUT_DIR, exist_ok=True)
    p = os.path.join(config_tabular.OUT_DIR, "feature_availability.csv")
    df.to_csv(p, index=False)

    unusable = df.loc[df["unusable_at_prediction_time"], "column"].tolist()
    shifted = df.loc[df["large_shift"] & ~df["unusable_at_prediction_time"],
                     "column"].tolist()

    md = [
        "# the tabular arm feature availability",
        "",
        f"73 clinician-signed columns in the data files; {int(df['in_model_69'].sum())} "
        "used as model input. Populated = non-missing.",
        "",
        "## Populated at training time, absent at prediction time",
        "",
        "These columns carry information the model can learn from and will "
        "never have in production. They are retained because the signed feature "
        "list is not modifiable in code; the limitation is reported, not "
        "silently patched.",
        "",
        "| column | PUF train | PUF test 2024 | SBUH test |",
        "|---|---|---|---|",
    ]
    for c in unusable:
        r = df.loc[df["column"] == c].iloc[0]
        md.append(f"| `{c}` | {r['pct_populated_puf_train']:.1f}% | "
                  f"{r['pct_populated_puf_test']:.1f}% | "
                  f"{r['pct_populated_sbuh_test']:.1f}% |")
    md += [
        "",
        f"## Large availability shift (>= {SHIFT_PP:.0f} points, either direction)",
        "",
        "Introduced or retired mid-window, so the fitted imputation default was "
        "set on a training pool with a very different missing rate.",
        "",
        "| column | PUF train | PUF test 2024 | SBUH test | train - test |",
        "|---|---|---|---|---|",
    ]
    for c in shifted:
        r = df.loc[df["column"] == c].iloc[0]
        md.append(f"| `{c}` | {r['pct_populated_puf_train']:.1f}% | "
                  f"{r['pct_populated_puf_test']:.1f}% | "
                  f"{r['pct_populated_sbuh_test']:.1f}% | "
                  f"{r['train_minus_test_pp']:+.1f} pp |")
    md += [
        "",
        "## Not model input",
        "",
        "`DPRHEMOGLOBIN`, `DPRHEMO_A1C`, `PRPT`, `DPRPT` are 100% empty in the "
        "PUF training pool -- nothing to learn, and XGBoost raises "
        "`cannot call vectorize on size 0 inputs` on an empty category array. "
        "Judged on the training set alone; they remain in the data files.",
        "",
        "## Measurement caveat",
        "",
        "`PRHEMOGLOBIN` is not the same quantity on both sides: PUF is the "
        "approximation `0.344 x PRHCT - 0.56` (R2 = 0.74, fitted on 169,354 "
        "2023 pairs), SBUH is a measured value. Any interpretation of that "
        "variable's contribution has to say so.",
        "",
    ]
    pmd = os.path.join(config_tabular.OUT_DIR, "feature_availability.md")
    with open(pmd, "w") as f:
        f.write("\n".join(md))

    print(f"unusable at prediction time ({len(unusable)}): {', '.join(unusable)}")
    print(f"large availability shift ({len(shifted)}): {', '.join(shifted)}")
    print(f"\nDone -> {p}\n        {pmd}")


if __name__ == "__main__":
    main()
