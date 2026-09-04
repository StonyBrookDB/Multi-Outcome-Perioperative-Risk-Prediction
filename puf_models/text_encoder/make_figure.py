"""
Figure 1: discrimination of all nine models, national cohort against the single
institution.

  Terminology: the three encoder rows are 110M-335M parameter encoder-only
  models (two BERT checkpoints and one BGE-derived embedding model), not
  generative large language models. They are called text encoders throughout;
  "LLM" would overstate what they are.

    python make_figure.py --out ../../amiaAmplify2027/figures

WHAT THE FIGURE IS FOR

  Table 5 has every number; a figure that repeated them would earn nothing. This
  one shows the two things the table makes a reader compute for themselves:

    - across the top row, nine models sit inside a span narrower than most
      readers would guess from a nine-row table -- the paper's central claim,
      that architecture stops mattering once the predictor set is fixed;
    - between the top row and the bottom, the intervals grow by an order of
      magnitude, and the ordering that looked clean nationally dissolves. That
      is the "national data to local evaluation" of the title, and it is the
      one thing a table of point estimates actively hides.

  AUROC only. AUPRC and the threshold-based rules stay in Table 5: a second
  metric would double the panel count and halve the size of the marks, and the
  rank instability between the two metrics is a sentence in the Results, not
  something a reader should have to extract from sixteen panels.

DESIGN

  Colour encodes model CLASS, not model identity. Nine categorical hues would be
  unreadable and would imply nine things worth telling apart; the y-axis label
  already names each model, so colour is free to carry the only distinction the
  paper argues about -- tabular family versus frozen text-encoder embedding. Two
  hues, validated for colour-vision deficiency (worst adjacent separation 24.7
  OKLab dE, against a target of 8).

  The ACS calculator is a dashed rule, not a tenth row: it is a deployed
  reference scored on the same cases, not a model this study fitted, and giving
  it a marker would invite it into the ranking.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import config_encoder  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib import font_manager          # noqa: E402
from matplotlib.lines import Line2D          # noqa: E402

# Match the manuscript's body face. amia.cls sets TeX Gyre Termes (the
# metric-compatible Times clone; Times New Roman itself is not on this
# cluster), and a figure set in the plotting library's default sans-serif
# announces itself as an import on every page it appears. The OTFs are
# registered directly rather than by family name so this does not depend on a
# fontconfig cache having been rebuilt.
FONT_DIR = "/vast/projects/akumar-group/NSQIP/zihan/fonts"
_registered = False
for _f in ("texgyretermes-regular.otf", "texgyretermes-bold.otf",
           "texgyretermes-italic.otf", "texgyretermes-bolditalic.otf"):
    _p = os.path.join(FONT_DIR, _f)
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)
        _registered = True
if _registered:
    plt.rcParams["font.family"] = "TeX Gyre Termes"
    # Maths in the same family, so an axis label with a symbol does not switch
    # face mid-line.
    plt.rcParams["mathtext.fontset"] = "custom"
    for _k in ("mathtext.rm", "mathtext.it", "mathtext.bf"):
        plt.rcParams[_k] = "TeX Gyre Termes"
    plt.rcParams["mathtext.it"] = "TeX Gyre Termes:italic"
    plt.rcParams["mathtext.bf"] = "TeX Gyre Termes:bold"
else:
    print(f"warning: no TeX Gyre Termes under {FONT_DIR}; "
          f"figure will not match the body text", file=sys.stderr)
# Keep glyphs as text in the PDF rather than outlines, so the figure stays
# searchable and copy-pasteable in the compiled paper.
plt.rcParams["pdf.fonttype"] = 42

# Validated categorical slots 1 and 2 (light mode). Text and grid wear ink
# tokens, never a series colour.
C_TABULAR = "#2a78d6"
C_ENC     = "#eb6834"
INK       = "#0b0b0b"
INK_2     = "#52514e"
GRID      = "#d8d7d2"
SURFACE   = "#ffffff"   # pure white: the figure sits on a white page, and the
                        # palette's off-white chart surface reads as a grey box
                        # against it. Re-validated at this surface.

TABULAR = ["logistic_regression", "random_forest", "xgboost", "lightgbm",
           "mlp", "ft_transformer"]
ENCODERS_ = ["medembed_large", "clinicalbert", "bio_clinicalbert"]
PRETTY = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
          "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP",
          "ft_transformer": "FT-Transformer", "medembed_large": "MedEmbed-large",
          "clinicalbert": "ClinicalBERT", "bio_clinicalbert": "Bio_ClinicalBERT"}
OUTCOMES = ["mortality", "any_compl", "unplanned_or", "readmission"]
OUT_TITLE = {"mortality": "Mortality", "any_compl": "Any complication",
             "unplanned_or": "Unplanned return to OR", "readmission": "Readmission"}
COHORTS = [("puf", "PUF 2024 (n = 963,565)"),
           ("sbuh", "Single institution (n = 3,490)")]


def load(cohort):
    frames = []
    for root in (config_encoder.TABULAR_OUT_DIR, config_encoder.OUT_DIR):
        p = os.path.join(root, f"summary_{cohort}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            frames.append(d[(d["kind"] == "ranking") & (d["metric"] == "auroc")])
    if not frames:
        raise SystemExit(f"no summary_{cohort}.csv found")
    return pd.concat(frames, ignore_index=True)


def acs_line(outcome):
    import json
    p = os.path.join(REPO, "results", "acs_benchmark", outcome,
                     "bootstrap_ci.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)["auroc"]["point"]


def build(path):
    order = TABULAR + ENCODERS_
    ypos = {m: len(order) - 1 - i for i, m in enumerate(order)}   # top-down

    # 6.5 in is the class's exact \textwidth (US Letter, 1 in margins), so the
    # figure is included at 1:1 and the point sizes below are the point sizes
    # that reach the page. At the previous 9.6 in the whole figure was scaled to
    # 0.68 on inclusion and every label lost a third of its size.
    fig, axes = plt.subplots(
        2, 4, figsize=(6.5, 4.8), sharey=True,
        gridspec_kw=dict(hspace=0.38, wspace=0.16,
                         left=0.225, right=0.995, top=0.865, bottom=0.185))
    fig.patch.set_facecolor(SURFACE)

    # One x range per OUTCOME, shared by both cohorts.
    #
    # Letting each panel autoscale is what a plotting library does by default,
    # and it destroys the figure's whole point: the single-institution
    # intervals are an order of magnitude wider than the national ones, and
    # independent axes rescale that difference away, leaving two rows that look
    # equally precise. Sharing the range within a column puts the two on one
    # ruler, so the growth is the first thing the eye gets. The cost is that the
    # national row compresses into a tight cluster -- which is itself the
    # result, and the exact values are in Table 5 for anyone who needs the
    # ordering.
    data = {c: load(c) for c, _ in COHORTS}
    xlim = {}
    for outcome in OUTCOMES:
        lo = min(float(data[c][data[c]["label"] == outcome]["ci_lo"].min())
                 for c, _ in COHORTS)
        hi = max(float(data[c][data[c]["label"] == outcome]["ci_hi"].max())
                 for c, _ in COHORTS)
        if (a := acs_line(outcome)) is not None:
            lo, hi = min(lo, a), max(hi, a)
        pad = (hi - lo) * 0.06
        xlim[outcome] = (lo - pad, hi + pad)

    for r, (cohort, cohort_label) in enumerate(COHORTS):
        d = data[cohort]
        for c, outcome in enumerate(OUTCOMES):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            s = d[d["label"] == outcome].set_index("model")

            if cohort == "puf" and (a := acs_line(outcome)) is not None:
                ax.axvline(a, color=INK_2, lw=1.0, ls=(0, (4, 3)), zorder=1)

            for m in order:
                if m not in s.index:
                    continue
                row = s.loc[m]
                col = C_TABULAR if m in TABULAR else C_ENC
                y = ypos[m]
                ax.plot([row["ci_lo"], row["ci_hi"]], [y, y],
                        color=col, lw=1.6, solid_capstyle="butt", zorder=2)
                # 2px surface ring so markers stay separable where they overlap
                ax.plot([row["value"]], [y], "o", ms=5.2, color=col,
                        mec=SURFACE, mew=1.1, zorder=3)

            ax.set_xlim(*xlim[outcome])
            ax.set_ylim(-0.7, len(order) - 0.3)

            ax.xaxis.set_major_locator(plt.MaxNLocator(3, prune="both"))
            ax.tick_params(axis="x", labelsize=7.6, colors=INK_2,
                           length=2.5, width=0.6, pad=2)
            ax.tick_params(axis="y", length=0, pad=3)
            ax.grid(axis="x", color=GRID, lw=0.55, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.spines["bottom"].set_linewidth(0.6)

            if r == 0:
                ax.set_title(OUT_TITLE[outcome], fontsize=9.2, color=INK,
                             pad=5, fontweight="medium")
            if c == 0:
                ax.set_yticks(range(len(order)))
                ax.set_yticklabels([PRETTY[m] for m in order[::-1]],
                                   fontsize=8.6, color=INK)
                # colour the tick text? no -- identity is carried by the marker
                # beside it; text stays in ink.
                ax.text(-0.34, 1.16 if r == 0 else 1.06, cohort_label,
                        transform=ax.transAxes, fontsize=9.6, color=INK,
                        fontweight="semibold", ha="left", va="bottom")

    fig.text(0.61, 0.075, "AUROC (point estimate and 95% bootstrap CI)",
             ha="center", fontsize=8.8, color=INK_2)

    handles = [
        Line2D([], [], color=C_TABULAR, lw=1.6, marker="o", ms=5.2,
               mec=SURFACE, mew=1.1, label="Tabular model families (6)"),
        Line2D([], [], color=C_ENC, lw=1.6, marker="o", ms=5.2,
               mec=SURFACE, mew=1.1, label="Text-encoder embedding classifiers (3)"),
        Line2D([], [], color=INK_2, lw=1.0, ls=(0, (4, 3)),
               label="ACS NSQIP calculator"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.4, labelcolor=INK_2, bbox_to_anchor=(0.55, 0.004),
               handlelength=1.9, columnspacing=1.2, handletextpad=0.6)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{os.path.splitext(path)[0]}.{ext}", dpi=300,
                    facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(REPO)), "amiaAmplify2027", "figures"))
    a = ap.parse_args()
    p = build(os.path.join(a.out, "fig1_discrimination.pdf"))
    print(f"wrote {os.path.splitext(p)[0]}.pdf and .png")


if __name__ == "__main__":
    main()
