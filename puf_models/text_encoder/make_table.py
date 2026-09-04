"""
Merge the tabular arm's six tabular families with the text-encoder arm's encoders into one table.

    python make_table.py                 markdown to stdout
    python make_table.py --latex         LaTeX, in the manuscript's table5 shape
    python make_table.py --out DIR       write both to DIR

Reads only the two summaries -- results/tabular/summary_{puf,sbuh}.csv and
results/text_encoder/summary_{puf,sbuh}.csv -- and never recomputes a metric. Both
were produced by the SAME evaluation code (the text-encoder arm imports the tabular arm's evaluate module),
so the numbers on a row are comparable by construction rather than by assertion.

Rows with kind == 'ranking' are AUROC and AUPRC, the threshold-free metrics, all
carrying a 1,000-draw bootstrap interval. Nothing threshold-dependent is merged
here: those rules differ between cohorts by design and do not belong in a single
cross-model table.

THE ACS REFERENCE ROW is read from results/acs_benchmark, unchanged --
NSQIP's own MORTPROB and MORBPROB scored on the identical 963,565 cases. It is
excluded from bolding: it is a deployed-tool reference, not a competing model.
"""

import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import config_encoder  # noqa: E402

TABULAR_ROOT = config_encoder.TABULAR_OUT_DIR
ENCODER_ROOT = config_encoder.OUT_DIR
ACS_DIR = os.path.join(REPO, "results", "acs_benchmark")

TABULAR = ["logistic_regression", "random_forest", "xgboost", "lightgbm",
           "mlp", "ft_transformer"]
PRETTY = {
    "logistic_regression": "Logistic regression", "random_forest": "Random forest",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP",
    "ft_transformer": "FT-Transformer",
    "medembed_large": "MedEmbed-large-v0.1", "clinicalbert": "ClinicalBERT",
    "bio_clinicalbert": "Bio\\_ClinicalBERT",
}
LABEL_ORDER = ["mortality", "any_compl", "unplanned_or", "readmission"]
LABEL_PRETTY = {"mortality": "Mortality", "any_compl": "Any complication",
                "unplanned_or": "Unplanned return to OR",
                "readmission": "Readmission"}


def load(cohort):
    frames = []
    for root, arm in ((TABULAR_ROOT, "tabular"), (ENCODER_ROOT, "text_encoder")):
        p = os.path.join(root, f"summary_{cohort}.csv")
        if not os.path.exists(p):
            print(f"  (missing {p} -- skipped)", file=sys.stderr)
            continue
        d = pd.read_csv(p)
        d = d[(d["kind"] == "ranking") & d["metric"].isin(["auroc", "auprc"])]
        d["arm"] = arm
        frames.append(d)
    if not frames:
        raise SystemExit("neither summary found; run evaluate_encoder.py first")
    return pd.concat(frames, ignore_index=True)


def acs_reference(cohort):
    """NSQIP's own risk estimates, for the two outcomes it publishes."""
    if cohort != "puf" or not os.path.isdir(ACS_DIR):
        return {}
    out = {}
    for lab, field in (("mortality", "MORTPROB"), ("any_compl", "MORBPROB")):
        p = os.path.join(ACS_DIR, lab, "bootstrap_ci.json")
        if os.path.exists(p):
            with open(p) as fh:
                d = json.load(fh)
            # The file also carries brier and other calibration quantities.
            # They are deliberately not merged: the manuscript reports no
            # calibration metrics, because every trained model here is fitted
            # with class re-weighting and its scores are not probabilities.
            # Reporting them for the ACS row alone would invite a comparison
            # that cannot be made.
            out[lab] = {k: d[k] for k in ("auroc", "auprc") if k in d}
    return out


def cell(row):
    return f"{row['value']:.3f} ({row['ci_lo']:.3f}–{row['ci_hi']:.3f})"


def build(cohort):
    df = load(cohort)
    present = [m for m in TABULAR if m in set(df["model"])]
    encoders = [e for e in config_encoder.ENCODERS if e in set(df["model"])]
    order = present + encoders
    out = {}
    for metric in ("auroc", "auprc"):
        sub = df[df["metric"] == metric]
        tbl = pd.DataFrame(index=order, columns=LABEL_ORDER, dtype=object)
        best = {}
        for lab in LABEL_ORDER:
            s = sub[sub["label"] == lab].set_index("model")
            if s.empty:
                continue
            best[lab] = s["value"].max()
            for m in order:
                if m in s.index:
                    tbl.loc[m, lab] = cell(s.loc[m])
        out[metric] = (tbl, best, sub)
    return out, encoders


def to_markdown(cohort):
    built, encoders = build(cohort)
    lines = [f"### {cohort}", ""]
    for metric, (tbl, best, sub) in built.items():
        lines += [f"**{metric.upper()}**", "",
                  "| Model | " + " | ".join(LABEL_PRETTY[l] for l in LABEL_ORDER) + " |",
                  "|---|" + "---|" * len(LABEL_ORDER)]
        for m in tbl.index:
            cells = []
            for lab in LABEL_ORDER:
                v = tbl.loc[m, lab]
                if pd.isna(v):
                    cells.append("—")
                    continue
                s = sub[(sub["label"] == lab) & (sub["model"] == m)]
                mark = "**" if (not s.empty and lab in best
                                and abs(float(s.iloc[0]["value"]) - best[lab]) < 5e-7) else ""
                cells.append(f"{mark}{v}{mark}")
            name = PRETTY.get(m, m).replace("\\_", "_")
            tag = " *(LLM)*" if m in encoders else ""
            lines.append(f"| {name}{tag} | " + " | ".join(cells) + " |")
        acs = acs_reference(cohort)
        if acs and metric in ("auroc", "auprc"):
            cells = []
            for lab in LABEL_ORDER:
                d = acs.get(lab, {}).get(metric)
                cells.append(
                    f"_{d['point']:.3f} ({d['ci_lo']:.3f}–{d['ci_hi']:.3f})_"
                    if d else "—")
            lines.append("| _ACS calculator_ | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def to_latex(cohort):
    """Rows only, in the shape of tables/table5_tabular_performance.tex."""
    built, encoders = build(cohort)
    rows = []
    for metric, (tbl, best, sub) in built.items():
        for i, m in enumerate(tbl.index):
            cells = []
            for lab in LABEL_ORDER:
                v = tbl.loc[m, lab]
                if pd.isna(v):
                    cells.append("---")
                    continue
                s = sub[(sub["label"] == lab) & (sub["model"] == m)]
                bold = (not s.empty and lab in best
                        and abs(float(s.iloc[0]["value"]) - best[lab]) < 5e-7)
                v = v.replace("–", "--")
                cells.append(f"\\textbf{{{v}}}" if bold else v)
            lead = (f"  {cohort} & {metric.upper()} & " if i == 0 else "   &  & ")
            rows.append(lead + PRETTY.get(m, m) + "\n    & "
                        + " & ".join(cells) + " \\\\")
        rows.append("  \\addlinespace")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    for cohort in ("puf", "sbuh"):
        text = to_latex(cohort) if a.latex else to_markdown(cohort)
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            ext = "tex" if a.latex else "md"
            p = os.path.join(a.out, f"table_{cohort}.{ext}")
            with open(p, "w") as fh:
                fh.write(text + "\n")
            print(f"wrote {p}")
        else:
            print(text)


if __name__ == "__main__":
    main()
