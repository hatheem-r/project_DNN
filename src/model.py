"""
The baseline model: BiLSTM (+ optional CRF).

This reproduces the architecture described in the SOLD paper: an embedding
layer, one bidirectional LSTM with 64 units, and a CRF on top.

WHAT EACH PIECE DOES

  Embedding layer
      Looks up the 300-number vector for each word id. Frozen by default,
      meaning the vectors do not change during training - the model uses
      fastText's knowledge as given. This is what the paper did.

  BiLSTM
      Reads the tweet twice, once left-to-right and once right-to-left,
      keeping a running memory. So the representation of word 7 knows about
      every word before AND after it. This matters because offensiveness is
      contextual: the SOLD paper shows most 'offensive' keywords are only
      offensive 30-50% of the time, so no word list can be correct always.
      ('Bidirectional' is the Bi in BiLSTM.)

  Linear layer
      Turns each word's BiLSTM output into 2 scores: one for 'not
      offensive', one for 'offensive'.

  CRF (optional)
      Without it, each word is labelled independently. A CRF instead scores
      the whole label SEQUENCE, learning that offensive words come in runs
      and that some label transitions are unlikely. Usually worth a point
      or two on span-style tasks. Build without it first, confirm the
      pipeline works, then switch it on and check the score goes up.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from embeddings import PAD_ID

try:
    from torchcrf import CRF
    HAS_CRF = True
except ImportError:  # pragma: no cover
    HAS_CRF = False


class BiLSTMTagger(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.5,
        freeze_embeddings: bool = True,
        use_crf: bool = True,
        num_labels: int = 2,
    ):
        super().__init__()
        vocab_size, dim = embedding_matrix.shape

        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            dim, hidden_size, num_layers=num_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

        self.use_crf = use_crf and HAS_CRF
        if use_crf and not HAS_CRF:
            raise ImportError("use_crf=True but pytorch-crf is not installed. "
                              "pip install pytorch-crf")
        self.crf = CRF(num_labels, batch_first=True) if self.use_crf else None
        self.num_labels = num_labels

    # ------------------------------------------------------------------
    def emissions(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Per-token scores, shape (batch, seq, num_labels)."""
        x = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        return self.classifier(self.dropout(out))

    # ------------------------------------------------------------------
    def loss(
        self,
        ids: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        em = self.emissions(ids, lengths)

        if self.use_crf:
            # CRF needs real labels everywhere; padding is excluded by the mask.
            safe = labels.clone()
            safe[~mask] = 0
            return -self.crf(em, safe, mask=mask, reduction="mean")

        loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        return loss_fn(em.reshape(-1, self.num_labels), labels.reshape(-1))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self, ids: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor
    ) -> List[List[int]]:
        """Predicted label per real token, padding already stripped."""
        em = self.emissions(ids, lengths)
        if self.use_crf:
            return self.crf.decode(em, mask=mask)
        best = em.argmax(-1)
        return [[int(v) for v in row[: int(m.sum())]] for row, m in zip(best, mask)]

    # ------------------------------------------------------------------
    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "embedding": self.embedding.weight.numel(),
            "non_embedding_trainable": trainable - (
                self.embedding.weight.numel() if self.embedding.weight.requires_grad else 0
            ),
        }