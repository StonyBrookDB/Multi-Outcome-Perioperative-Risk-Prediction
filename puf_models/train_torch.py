"""
Deep tabular models (PyTorch): entity-embedding MLP and FT-Transformer.

Usage:
    python train_torch.py --model mlp [--label mortality]
    python train_torch.py --model ft_transformer

Both consume the 'embed' view (scaled numerics + top-K integer-coded
categoricals). Shared training loop: BCEWithLogitsLoss with pos_weight for
imbalance, AdamW, early stopping on validation AUROC. Requires an env with
torch (e.g. mistral_env); uses CUDA when available.

FT-Transformer follows Gorishniy et al. 2021 ("Revisiting Deep Learning Models
for Tabular Data"): each feature -> one token, a [CLS] token is prepended, a
stack of pre-norm transformer encoder blocks mixes the tokens, and the final
[CLS] embedding is fed to a linear head. Encoder-only; no positional encoding
(feature identity is carried by per-feature tokenizers), no decoder.
"""

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import config
import data
import metrics

MODELS = {"mlp", "ft_transformer"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Models ───────────────────────────────────────────────────────────────────

def _embed_dims(cardinalities):
    # fastai-style rule of thumb, capped
    return [min(600, round(1.6 * c ** 0.56)) for c in cardinalities]


class EmbedMLP(nn.Module):
    def __init__(self, n_num, cardinalities):
        super().__init__()
        dims = _embed_dims(cardinalities)
        self.embs = nn.ModuleList(
            [nn.Embedding(c, d, padding_idx=0) for c, d in zip(cardinalities, dims)]
        )
        self.emb_drop = nn.Dropout(config.MLP_PARAMS["embed_dropout"])
        in_dim = n_num + sum(dims)

        layers, prev = [], in_dim
        for h in config.MLP_PARAMS["hidden_dims"]:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(config.MLP_PARAMS["hidden_dropout"])]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        e = [emb(x_cat[:, j]) for j, emb in enumerate(self.embs)]
        x = torch.cat([x_num] + e, dim=1)
        x = self.emb_drop(x)
        return self.mlp(x).squeeze(1)


class _Block(nn.Module):
    def __init__(self, d, n_heads, attn_drop, ffn_drop, res_drop, ffn_factor):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=attn_drop, batch_first=True)
        self.res1 = nn.Dropout(res_drop)
        self.norm2 = nn.LayerNorm(d)
        hidden = int(d * ffn_factor)
        self.ffn = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                 nn.Dropout(ffn_drop), nn.Linear(hidden, d))
        self.res2 = nn.Dropout(res_drop)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.res1(a)
        x = x + self.res2(self.ffn(self.norm2(x)))
        return x


class FTTransformer(nn.Module):
    def __init__(self, n_num, cardinalities):
        super().__init__()
        d = config.FT_PARAMS["d_token"]
        self.n_num = n_num
        # numeric tokenizer: per-feature weight + bias -> token
        self.num_weight = nn.Parameter(torch.empty(n_num, d))
        self.num_bias = nn.Parameter(torch.empty(n_num, d))
        nn.init.normal_(self.num_weight, std=d ** -0.5)
        nn.init.normal_(self.num_bias, std=d ** -0.5)
        # categorical tokenizer: one embedding table per column, all -> d
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(c, d, padding_idx=0) for c in cardinalities])
        self.cls = nn.Parameter(torch.empty(1, 1, d))
        nn.init.normal_(self.cls, std=d ** -0.5)

        p = config.FT_PARAMS
        self.blocks = nn.ModuleList([
            _Block(d, p["n_heads"], p["attn_dropout"], p["ffn_dropout"],
                   p["residual_dropout"], p["ffn_factor"])
            for _ in range(p["n_blocks"])
        ])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x_num, x_cat):
        b = x_num.shape[0]
        num_tok = x_num.unsqueeze(-1) * self.num_weight + self.num_bias  # (b, n_num, d)
        cat_tok = torch.stack([emb(x_cat[:, j]) for j, emb in enumerate(self.cat_embs)],
                              dim=1) if len(self.cat_embs) else num_tok[:, :0]
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, num_tok, cat_tok], dim=1)  # (b, 1+n_num+n_cat, d)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0])).squeeze(1)


# ── Training loop ────────────────────────────────────────────────────────────

def _loader(Xnum, Xcat, y, batch_size, shuffle):
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xnum), torch.from_numpy(Xcat),
        torch.from_numpy(y.astype(np.float32)))
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=config.TORCH_TRAIN["num_workers"], pin_memory=True,
        drop_last=False)


@torch.no_grad()
def _predict(model, loader):
    model.eval()
    out = []
    for xn, xc, _ in loader:
        xn, xc = xn.to(DEVICE, non_blocking=True), xc.to(DEVICE, non_blocking=True)
        out.append(torch.sigmoid(model(xn, xc)).float().cpu().numpy())
    return np.concatenate(out)


def train_one(model, Xnum_tr, Xcat_tr, ytr, batch_size):
    tr = config.TORCH_TRAIN
    Xn_tr, Xn_va, Xc_tr, Xc_va, y_tr, y_va = train_test_split(
        Xnum_tr, Xcat_tr, ytr, test_size=tr["val_frac"],
        random_state=config.SEED, stratify=ytr)

    train_loader = _loader(Xn_tr, Xc_tr, y_tr, batch_size, shuffle=True)
    val_loader = _loader(Xn_va, Xc_va, y_va, batch_size, shuffle=False)

    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                            weight_decay=tr["weight_decay"])

    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(1, tr["max_epochs"] + 1):
        model.train()
        for xn, xc, yb in train_loader:
            xn = xn.to(DEVICE, non_blocking=True)
            xc = xc.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xn, xc), yb)
            loss.backward()
            opt.step()

        val_prob = _predict(model, val_loader)
        auc = roc_auc_score(y_va, val_prob)
        print(f"    epoch {epoch:2d}  val_auroc={auc:.4f}")
        if auc > best_auc + 1e-4:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tr["patience"]:
                print(f"    early stop (best val_auroc={best_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--label", default=None, choices=config.LABELS)
    args = ap.parse_args()

    labels = [args.label] if args.label else config.LABELS
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    print(f"Device: {DEVICE}")

    print("Loading X + building embed view ...")
    X_train, X_test = data.load_X()
    (Xnum_tr, Xcat_tr), (Xnum_te, Xcat_te), cardinalities, (num_cols, cat_cols) = \
        data.prepare_embed(X_train, X_test)
    print(f"  {len(num_cols)} numeric, {len(cat_cols)} categorical "
          f"(max vocab {max(cardinalities)})")

    batch_size = (config.FT_PARAMS["batch_size"] if args.model == "ft_transformer"
                  else config.TORCH_TRAIN["batch_size"])
    test_loader = _loader(Xnum_te, Xcat_te,
                          np.zeros(len(Xnum_te), np.float32), batch_size, shuffle=False)

    for label in labels:
        print(f"\n=== {args.model} / {label} ===")
        ytr, yte = data.load_y(label)
        if args.model == "mlp":
            model = EmbedMLP(len(num_cols), cardinalities).to(DEVICE)
        else:
            model = FTTransformer(len(num_cols), cardinalities).to(DEVICE)

        with metrics.Timer() as t:
            model = train_one(model, Xnum_tr, Xcat_tr, ytr, batch_size)
            prob = _predict(model, test_loader)
        metrics.save_run(args.model, label, yte, prob, t.seconds)


if __name__ == "__main__":
    main()
