"""
The 73 label strings that turn a PUF row into a sentence, and the template
that assembles them.

WHY LABELS AND NOT COLUMN NAMES

  The premise of this arm is that a clinical language model brings prior
  knowledge to the variables. `DPRALKPH` defeats that: no pretraining corpus
  contains it, and WordPiece shatters it into subword fragments that carry no
  meaning. `Days before surgery the preoperative alkaline phosphatase was
  drawn` is the same variable in words the encoder has actually seen.

PROVENANCE, AND WHAT WAS AND WAS NOT A JUDGEMENT CALL

  These strings are transcriptions of the official ACS NSQIP PUF data
  dictionary definitions, cross-checked against the wording used in the
  collaborator's earlier template (tkurc/puf/preop_puf_prompt.txt). They carry
  no clinical judgement of their own -- the clinical decision was WHICH 73
  variables to include, and that was made and signed off once, upstream, for
  the tabular arm; nothing here revisits it.

  The one thing that can go wrong here is a mistranscription, and it is
  invisible downstream: label PRALKPH as a day-offset and the encoder is told a
  lab value is a number of days, while every metric downstream still looks
  entirely reasonable. Gate 3 in PLAN.md is a line-by-line diff of these 73
  against the data dictionary. Do it before the full run, not after.

ORDER

  Fixed, and equal to the column order of feature_73/*_X_*.parquet. serialize.py
  asserts this rather than trusting it.
"""

# Column -> label. Order is the feature_73 column order and is load-bearing:
# the serialised sentence follows it, and the assertion in serialize.py fails
# if the parquet ever ships columns in a different order.
LABELS = {
    "Age": "Age in years at surgery (top-coded, '90+')",
    "ASACLAS": "ASA physical status classification",
    "ASCITES": "Ascites within 30 days before surgery",
    "BLEEDDIS": "Bleeding disorder",
    "CPT": "CPT code of the principal procedure",
    "DIABETES": "Diabetes mellitus status/treatment",
    "DIALYSIS": "Currently on dialysis preoperatively",
    "DISCANCR": "Disseminated cancer",
    "DPRALBUM": "Days before surgery the preoperative serum albumin was drawn",
    "DPRALKPH": "Days before surgery the preoperative alkaline phosphatase was drawn",
    "DPRBILI": "Days before surgery the preoperative total bilirubin was drawn",
    "DPRBUN": "Days before surgery the preoperative blood urea nitrogen was drawn",
    "DPRCREAT": "Days before surgery the preoperative serum creatinine was drawn",
    "DPRHCT": "Days before surgery the preoperative haematocrit was drawn",
    "DPRINR": "Days before surgery the preoperative INR was drawn",
    "DPRNA": "Days before surgery the preoperative serum sodium was drawn",
    "DPRPLATE": "Days before surgery the preoperative platelet count was drawn",
    "DPRPTT": "Days before surgery the preoperative partial thromboplastin time was drawn",
    "DPRSGOT": "Days before surgery the preoperative AST/SGOT was drawn",
    "DPRWBC": "Days before surgery the preoperative white blood cell count was drawn",
    "ETHNICITY_HISPANIC": "Hispanic/Latino ethnicity",
    "FNSTATUS2": "Functional health status prior to surgery",
    "HEIGHT": "Height, inches",
    "HtoODay": "Days from hospital admission to operation",
    "HXCHF": "Congestive heart failure within 30 days before surgery",
    "HXCOPD": "History of severe COPD",
    "HYPERMED": "Hypertension requiring medication",
    "INOUT": "Inpatient vs outpatient case designation",
    "PRALBUM": "Preoperative serum albumin",
    "PRALKPH": "Preoperative alkaline phosphatase",
    "PRBILI": "Preoperative total bilirubin",
    "PRBUN": "Preoperative blood urea nitrogen",
    "PRCREAT": "Preoperative serum creatinine",
    "PRHCT": "Preoperative haematocrit",
    "PRINR": "Preoperative INR",
    "PRNCPTX": "Free-text description of the principal procedure",
    "PRPLATE": "Preoperative platelet count",
    "PRPTT": "Preoperative partial thromboplastin time",
    "PRSEPIS": "Preoperative SIRS / sepsis / septic shock status",
    "PRSGOT": "Preoperative AST/SGOT",
    "PRSODM": "Preoperative serum sodium",
    "PRWBC": "Preoperative white blood cell count",
    "RACE_NEW": "Patient race (multi-race as comma-separated combination)",
    "SEX": "Patient sex",
    "SMOKE": "Current smoker within 1 year of surgery",
    "STEROID": "Chronic steroid / immunosuppressant use",
    "SURGSPEC": "Surgical specialty of the operating surgeon",
    "TRANSFUS": ">=1 unit of packed red blood cells or whole blood transfused "
                "in the 72 hours before surgery",
    "TRANST": "Origin/transfer status at admission",
    "VENTILAT": "Ventilator dependent at time of surgery",
    "WEIGHT": "Weight, pounds",
    "WORKRVU": "Work RVU of the principal procedure",
    "CASETYPE": "Case acuity: elective, urgent, or emergent",
    "DPRHEMOGLOBIN": "Days before surgery the preoperative haemoglobin was drawn",
    "DPRHEMO_A1C": "Days before surgery the preoperative haemoglobin A1c was drawn",
    "DPRPT": "Days before surgery the preoperative prothrombin time was drawn",
    "DYSPNEA": "Dyspnea before surgery: none, on moderate exertion, or at rest",
    "ELECTSURG": "Elective surgery: the procedure was neither emergent nor urgent",
    "EMERGNCY": "Emergency case, as determined by the surgeon or anaesthesiologist",
    # HEMO and PRHEMO_A1C are the SAME measurement under two PUF coding eras --
    # the manuscript's harmonization table groups them on one row and keeps them
    # as distinct predictors. The labels say so rather than repeating one
    # string twice, which would read as a copy-paste error to anyone checking.
    "HEMO": "Preoperative haemoglobin A1c, percent, in the earlier PUF coding era",
    "HOMESUP": "For patients 75 or older admitted from home, whether the patient "
               "lives alone",
    "HXDEMENTIA": "History of dementia or cognitive impairment",
    "HXFALL": "Fall within the 6 months before surgery",
    "IMMUNO_CAT": "Class or classes of immunosuppressive therapy taken before surgery",
    "OXYGEN_SUPPORT": "Requires supplemental oxygen before surgery",
    "PREOP_COVID": "Preoperative COVID-19 diagnosis, lab-confirmed or suspected",
    "PREOP_CREAT_MSINCR": "Severity of preoperative acute kidney injury, by KDIGO "
                          "serum-creatinine increase",
    # In PUF this is derived from haematocrit as 0.344*PRHCT - 0.56; at the
    # single institution it is measured directly. The label describes the
    # quantity, not its provenance -- the derivation is a Methods matter and
    # putting it in the sentence would make PUF and SBUH text differ for a
    # reason that is not about the patient.
    "PRHEMOGLOBIN": "Preoperative haemoglobin, g/dL",
    "PRHEMO_A1C": "Preoperative haemoglobin A1c, percent, in the later PUF coding era",
    "PRPT": "Preoperative prothrombin time, seconds",
    "RENAFAIL": "Preoperative acute renal failure (2018-2020) or acute kidney "
                "injury (2022+)",
    "WNDINF": "Open wound at the time of surgery, with or without infection",
    "WTLOSS": "Malnourishment or unintentional weight loss before surgery",
}

PREFIX = "This patient's "
JOIN = ", "
SUFFIX = "."


def render(values, columns, missing_token):
    """One patient row -> one sentence.

    `values` is a sequence parallel to `columns`; a value that is None is
    already-resolved missingness (serialize.py normalises NaN and the PUF -99
    sentinel before calling). Every column is emitted, in order, always.
    """
    parts = [f"{LABELS[c]} is {missing_token if v is None else v}"
             for c, v in zip(columns, values)]
    return PREFIX + JOIN.join(parts) + SUFFIX
