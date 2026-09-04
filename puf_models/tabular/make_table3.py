"""Table 3 — ranking performance, six tabular models x four outcomes.

Four panels: (PUF 2024, SBUH) x (AUROC, AUPRC). One row per model, one column
per outcome, cell = point estimate (95% bootstrap CI). Bold marks the highest
point estimate in each column of each panel; ties at the printed precision are
all bolded.

Reads results/tabular/summary_{puf,sbuh}.csv (the tidy file), never the wide
pivot, so the CI columns come from the same rows the metrics do.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import config_tabular  # noqa: E402

DEC = 3
MODEL_NAME = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "mlp": "MLP",
    "ft_transformer": "FT-Transformer",
}
MODEL_ORDER = list(MODEL_NAME)
LABEL_NAME = {
    "mortality": "Mortality",
    "any_compl": "Any complication",
    "unplanned_or": "Unplanned reoperation",
    "readmission": "Readmission",
}
LABEL_ORDER = list(LABEL_NAME)
COHORT_NAME = {"puf": "PUF 2024", "sbuh": "External validation (SBUH 2024-2025)"}


def load(cohort):
    df = pd.read_csv(os.path.join(config_tabular.OUT_DIR, f"summary_{cohort}.csv"))
    return df[df["kind"] == "ranking"]


def cell(v, lo, hi, bold):
    s = f"{v:.{DEC}f} ({lo:.{DEC}f}-{hi:.{DEC}f})"
    return f"**{s}**" if bold else s


def panel(df, metric):
    d = df[df["metric"] == metric].set_index(["model", "label"])
    # bold on the printed value so what a reader compares is what got bolded
    best = {lab: round(d.xs(lab, level="label")["value"].max(), DEC)
            for lab in LABEL_ORDER}
    lines = ["| Model | " + " | ".join(LABEL_NAME[l] for l in LABEL_ORDER) + " |",
             "|---" * (len(LABEL_ORDER) + 1) + "|"]
    for m in MODEL_ORDER:
        cells = []
        for lab in LABEL_ORDER:
            r = d.loc[(m, lab)]
            cells.append(cell(r["value"], r["ci_lo"], r["ci_hi"],
                              round(r["value"], DEC) >= best[lab]))
        lines.append(f"| {MODEL_NAME[m]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def counts(df):
    d = df[df["metric"] == "auroc"].drop_duplicates("label").set_index("label")
    return ", ".join(f"{LABEL_NAME[l].lower()} {int(d.loc[l, 'n_pos']):,}"
                     f" ({d.loc[l, 'prevalence'] * 100:.1f}%)" for l in LABEL_ORDER)


def main():
    puf, sbuh = load("puf"), load("sbuh")
    nb = int(puf["n_boot_valid"].max())
    n_puf = int(puf["n"].iloc[0])
    n_sbuh = int(sbuh["n"].iloc[0])

    out = [
        "**Table 3. Performance of six tabular models in the 2024 PUF temporal "
        "test cohort and the external validation cohort.** Values are AUROC and "
        "AUPRC point estimates with 95% confidence intervals from "
        f"{nb:,} nonparametric bootstrap resamples of patients; one shared set "
        "of resamples was used for every model within a cohort, so the draws "
        "support paired between-model comparison. Both metrics are threshold-"
        "free and are unaffected by the class re-weighting used in training. "
        "All models were frozen after development on PUF 2018-2022 and applied "
        "without refitting to both evaluation cohorts. Bold indicates the "
        "highest point estimate for each outcome within each panel; ties at the "
        "printed precision are bolded. AUPRC should be read against the outcome "
        "prevalence in that cohort, which is the AUPRC of a random classifier. "
        f"PUF 2024 n = {n_puf:,} (events: {counts(puf)}); external validation "
        f"n = {n_sbuh:,} (events: {counts(sbuh)}). AUROC, area under the "
        "receiver operating characteristic curve; AUPRC, area under the "
        "precision-recall curve.",
        "",
        f"**A. PUF 2024 - AUROC**", "", panel(puf, "auroc"), "",
        f"**B. PUF 2024 - AUPRC**", "", panel(puf, "auprc"), "",
        f"**C. External validation - AUROC**", "", panel(sbuh, "auroc"), "",
        f"**D. External validation - AUPRC**", "", panel(sbuh, "auprc"), "",
    ]
    txt = "\n".join(out)
    dest = os.path.join(config_tabular.OUT_DIR, "table3_ranking.md")
    with open(dest, "w") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"\n[written] {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
