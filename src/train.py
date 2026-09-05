"""
Training loop for the baseline.

MODEL SELECTION. After every epoch we score the model on VALIDATION and keep
the weights that scored best. Test is never used to decide anything. We
select on validation offensive-class F1 rather than on validation loss,
because F1 is the quantity we actually care about and loss on a 96%-negative
task is dominated by the easy class.

EARLY STOPPING. If validation F1 has not improved for `patience` epochs, we
stop. This prevents overfitting and saves time.

REPRODUCIBILITY. set_seed() fixes every source of randomness so the same
seed gives the same result. Report the mean over 5 seeds, as the SOLD paper
did.
"""

from __future__ import annotations

import copy
import random
import time
from typing import Dict, Optional

import numpy as np
import torch

from dataset import make_loader, unpad_predictions
from metrics import token_level_scores


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():   # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    gold_all, pred_all = [], []
    for ids, labels, mask, lengths, _, pid, plen in loader:
        ids, mask = ids.to(device), mask.to(device)
        if pid is not None:
            pid, plen = pid.to(device), plen.to(device)
        preds = model.predict(ids, mask, lengths, pid, plen)
        gold_all.extend(unpad_predictions(labels, mask.cpu()))
        pred_all.extend(preds)
    return token_level_scores(gold_all, pred_all)


def compute_class_weights(df, device) -> torch.Tensor:
    """Inverse-frequency weights, computed from TRAIN only."""
    pos = int(df["n_offensive_tokens"].sum())
    tot = int(df["n_tokens"].sum())
    neg = tot - pos
    w = torch.tensor([tot / (2 * neg), tot / (2 * pos)], dtype=torch.float)
    return w.to(device)


def train_one_seed(
    model_fn,
    train_df,
    val_df,
    vocab,
    seed: int,
    *,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 5,
    weight_decay: float = 0.0,
    grad_clip: float = 5.0,
    class_weights: Optional[torch.Tensor] = None,
    device=None,
    verbose: bool = True,
    sp=None,
    sentence_lambda: float = 0.0,
    distill_loader=None,
    distill_weight: float = 1.0,
):
    """Train once. Returns (best_model, history, best_val_scores, seconds)."""
    device = device or get_device()
    g = set_seed(seed)

    model = model_fn().to(device)
    train_loader = make_loader(train_df, vocab, batch_size, shuffle=True,
                               generator=g, sp=sp)
    val_loader = make_loader(val_df, vocab, batch_size, shuffle=False, sp=sp)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )

    best_f1, best_state, best_scores, bad_epochs = -1.0, None, None, 0
    history = []
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n_batches = 0.0, 0
        distill_iter = iter(distill_loader) if distill_loader is not None else None
        for ids, labels, mask, lengths, sent, pid, plen in train_loader:
            ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
            if pid is not None:
                pid, plen = pid.to(device), plen.to(device)
            opt.zero_grad()
            loss = model.loss(ids, labels, mask, lengths,
                              class_weights=class_weights,
                              piece_ids=pid, piece_lens=plen,
                              sentence_labels=sent.to(device) if sentence_lambda else None,
                              sentence_lambda=sentence_lambda)
            # PIECE 4: one unlabeled SemiSOLD batch per labeled batch.
            if distill_iter is not None:
                try:
                    d = next(distill_iter)
                except StopIteration:
                    distill_iter = iter(distill_loader)
                    d = next(distill_iter)
                d_ids, d_mask, d_len, d_pid, d_plen, d_soft = d
                d_ids, d_mask = d_ids.to(device), d_mask.to(device)
                d_soft = d_soft.to(device)
                if d_pid is not None:
                    d_pid, d_plen = d_pid.to(device), d_plen.to(device)
                loss = loss + distill_weight * model.distillation_loss(
                    d_ids, d_mask, d_len, d_soft, d_pid, d_plen)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total += loss.item()
            n_batches += 1

        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": total / max(n_batches, 1), **val})

        if verbose:
            print(f"    epoch {epoch:>2}  loss {total/max(n_batches,1):.4f}  "
                  f"val P {val['offensive_precision']:.4f}  "
                  f"R {val['offensive_recall']:.4f}  "
                  f"F1 {val['offensive_f1']:.4f}")

        if val["offensive_f1"] > best_f1:
            best_f1 = val["offensive_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_scores = val
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"    early stop at epoch {epoch} (best F1 {best_f1:.4f})")
                break

    model.load_state_dict(best_state)
    return model, history, best_scores, time.time() - t0