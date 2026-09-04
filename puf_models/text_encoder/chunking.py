"""
Partition the 73 fields into contiguous groups that each fit in 512 tokens.

WHY A FIXED PARTITION AND NOT PER-ROW PACKING

  A greedy per-row pack would use the budget better, but it would put different
  fields in different chunks for different patients. Two patients would then get
  representations built from differently-composed passages, and any difference
  between them would be partly an artefact of how their text happened to pack.
  One partition, applied to every patient in every cohort under every encoder,
  keeps that out of the comparison entirely.

  The cost of the fixed partition is headroom: the plan is sized on the
  WORST-CASE row, so the median row leaves some of each 512-window empty. That
  is the right trade -- padding is cheap, non-comparability is not.

WHY CONTIGUOUS

  The field order is the parquet's own and is already fixed. Contiguous groups
  mean the plan is describable in one sentence ("fields 1-25, 26-49, 50-73"),
  which matters when it has to go in a Methods section.

WHAT THE PLAN IS DERIVED FROM

  Per-field worst-case token cost, maximised across all three tokenizers, over
  a calibration sample of the TRAINING cohort. No outcome is read at any point,
  and the plan is a deterministic function of the template and the tokenizers.
  It is written once to chunk_plan.json, folded into config_encoder.config_sha256(),
  and asserted unchanged at the end of the run.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config_encoder      # noqa: E402
import labels73     # noqa: E402


def _field_text(col, value, missing_token):
    """Exactly the substring render() contributes for one field."""
    return f"{labels73.LABELS[col]} is {missing_token if value is None else value}"


def measure_field_costs(tokenizers, cohort="puf", split="train", n_sample=20000):
    """Worst-case token cost per field, maximised over rows and tokenizers.

    Costs are measured on the field's own rendered substring plus the joiner, so
    they sum to slightly more than the true cost of a concatenated chunk (a
    tokenizer can merge across a boundary). Over-estimating is the safe
    direction: it can only make chunks shorter than the budget, never longer.
    """
    import serialize

    cols = serialize.columns(cohort, split)
    worst = {c: 0 for c in cols}
    seen = 0
    mask = (set(config_encoder.TRAINING_EMPTY_COLS)
            if config_encoder.MASK_TRAINING_EMPTY else set())

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(serialize.parquet_path(cohort, split))
    for batch in pf.iter_batches(batch_size=4096, columns=cols):
        data = {c: batch.column(i).to_pylist() for i, c in enumerate(cols)}
        for r in range(batch.num_rows):
            for c in cols:
                v = None if c in mask else serialize._fmt(data[c][r])
                s = _field_text(c, v, config_encoder.MISSING_TOKEN) + labels73.JOIN
                n = max(len(tk(s, add_special_tokens=False)["input_ids"])
                        for tk in tokenizers)
                if n > worst[c]:
                    worst[c] = n
            seen += 1
            if seen >= n_sample:
                return cols, worst, seen
    return cols, worst, seen


def pack(cols, costs, budget, overhead):
    """Greedy contiguous packing: fill a chunk until the next field would burst it."""
    groups, cur, cur_cost = [], [], overhead
    for c in cols:
        if cur and cur_cost + costs[c] > budget:
            groups.append(cur)
            cur, cur_cost = [], overhead
        cur.append(c)
        cur_cost += costs[c]
    if cur:
        groups.append(cur)
    return groups


def build_plan(tokenizers, tokenizer_names, n_sample=20000):
    """Measure, pack, and return the plan dict. Does not write it."""
    prefix_overhead = max(
        len(tk(labels73.PREFIX, add_special_tokens=True)["input_ids"])
        for tk in tokenizers)
    cols, costs, seen = measure_field_costs(tokenizers, n_sample=n_sample)
    groups = pack(cols, costs, config_encoder.CHUNK_BUDGET, prefix_overhead)
    return {
        "n_chunks": len(groups),
        "groups": groups,
        "field_worst_case_tokens": costs,
        "prefix_overhead_tokens": prefix_overhead,
        "budget": config_encoder.CHUNK_BUDGET,
        "max_len": config_encoder.MAX_LEN,
        "calibration": {
            "cohort": "puf", "split": "train", "rows": seen,
            "tokenizers": list(tokenizer_names),
        },
        "mask_training_empty": config_encoder.MASK_TRAINING_EMPTY,
        "missing_token": config_encoder.MISSING_TOKEN,
    }


def save_plan(plan, path=None):
    path = path or config_encoder.CHUNK_PLAN_FILE
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(plan, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def load_plan(path=None):
    path = path or config_encoder.CHUNK_PLAN_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing. Run:  python gate_check.py --calibrate")
    with open(path) as fh:
        plan = json.load(fh)
    if plan["mask_training_empty"] != config_encoder.MASK_TRAINING_EMPTY:
        raise AssertionError(
            "chunk_plan.json was built with mask_training_empty="
            f"{plan['mask_training_empty']} but config_encoder now says "
            f"{config_encoder.MASK_TRAINING_EMPTY}. Recalibrate; do not mix.")
    return plan


def render_chunks(values, cols, plan, missing_token=None):
    """One patient row -> n_chunks sentences, in plan order.

    Each chunk is a standalone passage carrying the same PREFIX, so the encoder
    sees a well-formed clinical sentence rather than a fragment.
    """
    missing_token = missing_token or config_encoder.MISSING_TOKEN
    by_col = dict(zip(cols, values))
    out = []
    for group in plan["groups"]:
        parts = [_field_text(c, by_col[c], missing_token) for c in group]
        out.append(labels73.PREFIX + labels73.JOIN.join(parts) + labels73.SUFFIX)
    return out


def describe(plan):
    lines = [f"{plan['n_chunks']} chunks, budget {plan['budget']} tokens "
             f"(+{plan['prefix_overhead_tokens']} overhead), "
             f"calibrated on {plan['calibration']['rows']:,} rows"]
    costs = plan["field_worst_case_tokens"]
    for i, g in enumerate(plan["groups"]):
        tot = sum(costs[c] for c in g) + plan["prefix_overhead_tokens"]
        lines.append(f"  chunk {i}: fields {g[0]}..{g[-1]}  "
                     f"({len(g)} fields, worst case {tot} tokens)")
    return "\n".join(lines)
