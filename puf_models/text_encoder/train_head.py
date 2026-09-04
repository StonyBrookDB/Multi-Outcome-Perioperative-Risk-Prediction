"""
Train one downstream head: (encoder, outcome) -> scores on val, PUF test, SBUH.

    python train_head.py --encoder bio_clinicalbert --label mortality

WHAT MAKES THIS COMPARABLE TO the tabular arm

  Everything except the input representation. The head is the tabular arm's MLP -- same
  widths, same dropout, same AdamW, same batch size, same BCEWithLogitsLoss
  with pos_weight, same 30-epoch cap, same early stop on val AUROC with
  patience 5 and best-epoch restore. The fit/val rows are the tabular arm's own, read from
  its saved split rather than re-derived. The inputs are standardized on the fit
  split, which is what the tabular arm does to its numeric predictors (see head_config).

  So `MLP-on-embedding` against the tabular arm's `MLP-on-tabular` differs in the
  representation and in nothing else, which is the only reason the two can
  share a table.

  Deliberately NOT focal loss and NOT a balanced sampler. The earlier arm used
  both; they are extra free parameters and they distort the score scale a
  second time on top of the re-weighting the manuscript already discloses.

OUTPUT ORDER MATTERS

  The three *_pred.parquet files are written BEFORE any metric is computed. Per
  -case scores at full float64 precision are the artifact that makes every
  downstream number recomputable without a retrain; the earlier arm was thought
  lost precisely because they were missing. If this script dies after training
  but before evaluation, the scores still survive.

NO CALIBRATION, exactly as in the tabular arm. Class re-weighting deliberately rescales
the output, so score_raw ranks patients and is not a risk.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config_encoder       # noqa: E402
import head_config   # noqa: E402


# ── data ─────────────────────────────────────────────────────────────────────

def load_embeddings(encoder, cohort, split):
    """Read the matrix, into RAM by default.

    Indexing the memmap per batch is not viable here: measured on this
    filesystem a random 8,192-row gather out of the 10 GB training matrix costs
    ~2.4 s, i.e. 23 min per epoch. Read sequentially into RAM it is seconds, and
    fp16 keeps the fit split at 9.7 GB.
    """
    p = config_encoder.emb_path(encoder, cohort, split)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} -- run the embedding array first")
    a = np.load(p, mmap_mode="r")
    n_expected = config_encoder.N_ROWS[(cohort, split)]
    if a.shape[0] != n_expected:
        raise AssertionError(f"{p}: {a.shape[0]:,} rows, expected {n_expected:,}")
    if not head_config.LOAD_TO_RAM:
        return a
    out = np.empty(a.shape, dtype=np.dtype(head_config.RAM_DTYPE))
    step = 500_000
    for i in range(0, a.shape[0], step):          # sequential, not a gather
        out[i:i + step] = a[i:i + step]
    return out


def load_y(cohort, split, label):
    p = os.path.join(config_encoder.DATA_DIR, f"{cohort}_Y_{split}_{label}.parquet")
    return pd.read_parquet(p).iloc[:, 0].to_numpy().astype(np.float32)


def load_split(label):
    """the tabular arm's fit/val row indices for this outcome.

    Read from the `row` column, never the `split` column: `split` is 'fit' x
    4,745,886 then 'val' x 249,784 for every model and every outcome alike, so
    comparing it would report a match that is not there. The four outcomes have
    genuinely different splits (val overlap ~12,700 of 249,784).
    """
    p = config_encoder.split_path(label)
    if not os.path.exists(p):
        raise SystemExit(f"missing the tabular arm split {p}")
    t = pd.read_parquet(p)
    fit_idx = np.sort(t.loc[t["split"] == "fit", "row"].to_numpy())
    val_idx = np.sort(t.loc[t["split"] == "val", "row"].to_numpy())
    n = config_encoder.N_ROWS[("puf", "train")]
    if len(fit_idx) + len(val_idx) != n:
        raise AssertionError(
            f"split covers {len(fit_idx) + len(val_idx):,} rows, expected {n:,}")
    if np.intersect1d(fit_idx, val_idx).size:
        raise AssertionError("fit and val overlap in the tabular arm's split file")
    return fit_idx, val_idx


# ── standardization, fitted on the fit split only ────────────────────────────

def fit_scaler(X, fit_idx, chunk=250_000):
    """Per-dimension mean and scale over the fit rows, in float32.

    Two passes rather than a sum-of-squares shortcut: the components are ~0.004
    and squaring them in one pass loses the precision that matters here.

    chunk is kept modest because each pass materialises the block in float64:
    at 250k rows x 1024 dims that is 2 GB, and the centre-and-square step holds
    two more like it.
    """
    d = X.shape[1]
    total = np.zeros(d, dtype=np.float64)
    for i in range(0, len(fit_idx), chunk):
        total += np.asarray(X[fit_idx[i:i + chunk]], dtype=np.float64).sum(0)
    mean = (total / len(fit_idx)).astype(np.float32)
    var = np.zeros(d, dtype=np.float64)
    for i in range(0, len(fit_idx), chunk):
        b = np.asarray(X[fit_idx[i:i + chunk]], dtype=np.float64) - mean
        var += (b * b).sum(0)
    scale = np.sqrt(var / len(fit_idx)).astype(np.float32)
    # A constant dimension carries no information; leaving its scale at 0 would
    # produce inf. 1.0 maps it to a constant 0 after centring.
    scale[scale < 1e-8] = 1.0
    return mean, scale


def apply_scaler(block, mean, scale):
    return (block - mean) / scale


# ── model ────────────────────────────────────────────────────────────────────

class Head(torch.nn.Module):
    """the tabular arm's MLP trunk, with a dense embedding in place of the tabular view.

    the tabular arm's EmbedMLP is an entity-embedding layer for the categorical columns
    followed by 512-256-128 with dropout. Here the encoder has already produced
    the dense vector, so only the trunk remains -- identical widths, identical
    dropout, identical final linear.
    """

    def __init__(self, in_dim, hidden_dims, dropout):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden_dims:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU(),
                       torch.nn.Dropout(dropout)]
            d = h
        layers.append(torch.nn.Linear(d, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _batches(X, idx, batch, mean, scale, shuffle=False, rng=None):
    """Yield (row indices, standardized float32 tensor) in RAM.

    `sel` carries ORIGINAL row numbers, so callers index their full-length label
    array with it directly -- no position bookkeeping to get wrong.
    """
    order = idx.copy()
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), batch):
        sel = order[i:i + batch]
        blk = np.asarray(X[sel], dtype=np.float32)
        if mean is not None:
            blk = apply_scaler(blk, mean, scale)
        yield sel, torch.from_numpy(blk)


@torch.inference_mode()
def predict(net, X, idx, batch, mean, scale, device):
    net.eval()
    out = np.empty(len(idx), dtype=np.float64)
    pos = 0
    for sel, xb in _batches(X, idx, batch, mean, scale):
        out[pos:pos + len(sel)] = net(xb.to(device)).float().cpu().numpy()
        pos += len(sel)
    return out


def _clear_stale_claim(claim, force):
    """Drop a claim left behind by a job that is no longer running.

    A killed job (wall clock, node failure) never reaches the line that removes
    its claim, so without this the cell stays blocked and the only way past it
    is --force -- which also retrains every cell that already finished. Asking
    Slurm whether the claiming job still exists makes recovery just
    "resubmit": live claims are respected, dead ones are cleared.
    """
    if not os.path.exists(claim):
        return
    if force:
        os.remove(claim)
        return
    try:
        with open(claim) as fh:
            txt = fh.read()
        jid = txt.split("job", 1)[1].split()[0]
    except Exception:
        return                                  # unreadable: leave it alone
    if jid == "local":
        return
    import subprocess
    r = subprocess.run(["squeue", "-j", jid, "-h", "-o", "%T"],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return                                  # still running: respect it
    print(f"    clearing stale claim from job {jid} (no longer in the queue)")
    os.remove(claim)


def train(encoder, label, force=False, _shared=None):
    from sklearn.metrics import roc_auc_score

    outdir = config_encoder.out_dir(encoder, label)
    if os.path.exists(os.path.join(outdir, "DONE")) and not force:
        print(f"[skip] {encoder}/{label} already DONE")
        return

    # Atomic claim. Two jobs can legitimately target the same cell -- an ad-hoc
    # run for the encoders whose embeddings finished early, and the chained
    # array that covers all three -- and without this they would both train it
    # and interleave their writes to the same *_pred.parquet. O_EXCL makes the
    # claim indivisible, so exactly one process proceeds.
    #
    # A claim left behind by a killed job blocks the cell until --force clears
    # it; the file names the job that made it so it is obvious which.
    claim = os.path.join(outdir, "IN_PROGRESS")
    _clear_stale_claim(claim, force)
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        with open(claim) as fh:
            who = fh.read().strip()
        print(f"[skip] {encoder}/{label} claimed by {who} -- "
              f"another job is training it. Use --force to override.")
        return
    with os.fdopen(fd, "w") as fh:
        fh.write(f"job {os.environ.get('SLURM_JOB_ID', 'local')} "
                 f"task {os.environ.get('SLURM_ARRAY_TASK_ID', '-')} "
                 f"pid {os.getpid()}\n")
    os.makedirs(os.path.join(outdir, "model"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, hp = config_encoder.HEAD_TRAIN, config_encoder.HEAD_PARAMS
    torch.manual_seed(config_encoder.SEED)
    rng = np.random.default_rng(config_encoder.SEED)
    t0 = time.time()

    # The three matrices do not depend on the outcome, so when several outcomes
    # run in one process they are loaded once and handed on through _shared.
    if _shared is None:
        _shared = {}
    if "Xtr" not in _shared:
        _shared["Xtr"] = load_embeddings(encoder, "puf", "train")
        _shared["Xte"] = load_embeddings(encoder, "puf", "test")
        _shared["Xsb"] = load_embeddings(encoder, "sbuh", "test")
    Xtr, Xte, Xsb = _shared["Xtr"], _shared["Xte"], _shared["Xsb"]
    y_all = load_y("puf", "train", label)          # indexed by original row
    y_test = load_y("puf", "test", label)
    y_sbuh = load_y("sbuh", "test", label)
    fit_idx, val_idx = load_split(label)
    y_fit, y_val = y_all[fit_idx], y_all[val_idx]
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    if head_config.STANDARDIZE_INPUT:
        mean, scale = fit_scaler(Xtr, fit_idx)
        np.savez(head_config.scaler_path(encoder, label), mean=mean, scale=scale)
    else:
        mean = scale = None

    pos, neg = float((y_fit == 1).sum()), float((y_fit == 0).sum())
    pos_weight = neg / max(pos, 1.0)
    net = Head(Xtr.shape[1], hp["hidden_dims"], hp["hidden_dropout"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device))

    print(f"[{encoder}/{label}] dim={Xtr.shape[1]} device={device} "
          f"fit={len(fit_idx):,} val={len(val_idx):,} "
          f"pos_weight={pos_weight:.1f} "
          f"standardized={head_config.STANDARDIZE_INPUT}", flush=True)

    y_all_t = torch.from_numpy(y_all)
    best_auc, best_epoch, bad, best_state = -1.0, 0, 0, None
    for epoch in range(1, tr["max_epochs"] + 1):
        te = time.time()
        net.train()
        for sel, xb in _batches(Xtr, fit_idx, tr["batch_size"], mean, scale,
                                shuffle=True, rng=rng):
            yb = y_all_t[sel].to(device)           # sel holds original rows
            opt.zero_grad()
            loss_fn(net(xb.to(device)), yb).backward()
            opt.step()
        s_val = predict(net, Xtr, val_idx, tr["batch_size"], mean, scale, device)
        auc = float(roc_auc_score(y_val, s_val))
        print(f"    epoch {epoch:2d}  val_auroc={auc:.4f}  "
              f"({time.time() - te:.0f}s)", flush=True)
        if auc > best_auc + 1e-4:
            best_auc, bad, best_epoch = auc, 0, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
        else:
            bad += 1
        if bad >= tr["patience"]:
            print(f"    early stop (best val_auroc={best_auc:.4f} @ epoch "
                  f"{best_epoch})", flush=True)
            break

    if best_state is not None:
        net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(net.state_dict(), os.path.join(outdir, "model", "state_dict.pt"))

    s_val = predict(net, Xtr, val_idx, tr["batch_size"], mean, scale, device)
    s_test = predict(net, Xte, np.arange(Xte.shape[0]), tr["batch_size"],
                     mean, scale, device)
    s_sbuh = predict(net, Xsb, np.arange(Xsb.shape[0]), tr["batch_size"],
                     mean, scale, device)

    # float64, untransformed, written before any metric is computed.
    for name, y_, s_ in (("test", y_test, s_test),
                         ("val", y_val, s_val),
                         ("sbuh", y_sbuh, s_sbuh)):
        pd.DataFrame({"y_true": np.asarray(y_).astype(int),
                      "score_raw": np.asarray(s_, dtype=np.float64)}).to_parquet(
            os.path.join(outdir, f"{name}_pred.parquet"), index=False)

    info = {
        "encoder": encoder, "label": label, "experiment": 6,
        "hf_id": config_encoder.ENCODERS[encoder]["hf_id"],
        "revision": config_encoder.ENCODERS[encoder]["revision"],
        "pooling": config_encoder.ENCODERS[encoder]["pooling"],
        "embedding_dim": int(Xtr.shape[1]),
        "n_fit": int(len(fit_idx)), "n_val": int(len(val_idx)),
        "n_test": int(len(y_test)), "n_sbuh": int(len(y_sbuh)),
        "split_source": config_encoder.split_path(label),
        "prevalence_fit": float(y_fit.mean()),
        "prevalence_val": float(y_val.mean()),
        "pos_weight": round(pos_weight, 4),
        "class_reweighting": "BCEWithLogitsLoss pos_weight",
        "standardize_input": head_config.STANDARDIZE_INPUT,
        "best_epoch": best_epoch, "best_val_auroc": round(best_auc, 6),
        "head": {"hidden_dims": list(hp["hidden_dims"]),
                 "dropout": hp["hidden_dropout"], **tr},
        "calibration": "none",
        "config_sha256": config_encoder.config_sha256(),
        "head_sha256": head_config.head_sha256(),
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(outdir, "train_meta.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    with open(os.path.join(outdir, "DONE"), "w") as fh:
        fh.write(f"epoch {best_epoch} auc {best_auc:.6f}\n")
    if os.path.exists(claim):
        os.remove(claim)
    print(f"[done] {encoder}/{label} best val_auroc={best_auc:.4f} "
          f"@ epoch {best_epoch}, {time.time() - t0:.0f}s", flush=True)


def train_all(encoder, labels, force=False):
    """All four outcomes for one encoder, loading the embeddings once.

    The matrices are the same for every outcome -- only the fit/val split, the
    labels and the class weight change -- so reloading 12 GB per outcome buys
    nothing. Sharing them also means the head stage needs three GPU allocations
    instead of twelve, which on a busy cluster is the difference that actually
    decides when the results exist.

    Each outcome still writes its own DONE marker, so a job that dies partway
    resumes at the outcome it stopped on.
    """
    shared = {}
    for lab in labels:
        train(encoder, lab, force, _shared=shared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(config_encoder.ENCODERS))
    ap.add_argument("--label", required=True,
                    choices=config_encoder.LABELS + ["all"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.label == "all":
        train_all(a.encoder, config_encoder.LABELS, a.force)
    else:
        train(a.encoder, a.label, a.force)


if __name__ == "__main__":
    main()
