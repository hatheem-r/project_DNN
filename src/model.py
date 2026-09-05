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


class SubwordEncoder(nn.Module):
    """Turn each word's subword pieces into ONE vector for that word.

    WHY THIS EXISTS
    ---------------
    Measured in Step 3: 48.7% of distinct test words never appeared in training,
    and 14.3% of test word occurrences are unseen. For those words the fastText
    channel holds a random vector - pure noise. But their PIECES were seen, so
    this channel can still say something useful.

    Measured in Step 6: unseen words are split into 2.5-4 pieces, and 75-83% of
    them contain a piece that also occurs inside a training word.

    THE CRITICAL CONTRACT
    ---------------------
    Input  (batch, n_words, n_pieces) piece ids
    Output (batch, n_words, out_dim)  exactly ONE vector per word

    Labels are one per word, so the number of output rows must equal the number
    of words. Anything else silently shifts labels against words.

    POOLING
    -------
    mean   : average the piece vectors. Simple, ignores order.
    bilstm : read the pieces left-to-right and right-to-left, take the final
             states. Can learn that a Sinhala word's ENDING carries grammar
             while its start carries meaning. Slower.

    Which wins is an empirical question and an ablation row.
    """

    def __init__(
        self,
        n_pieces: int,
        piece_dim: int = 50,
        out_dim: int = 100,
        pooling: str = "bilstm",
        dropout: float = 0.3,
        pad_id: int = 0,
    ):
        super().__init__()
        if pooling not in ("mean", "bilstm"):
            raise ValueError(f"pooling must be 'mean' or 'bilstm', got {pooling!r}")
        self.pooling = pooling
        self.out_dim = out_dim
        self.pad_id = pad_id

        self.embedding = nn.Embedding(n_pieces, piece_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)

        if pooling == "bilstm":
            if out_dim % 2:
                raise ValueError("out_dim must be even when pooling='bilstm'")
            self.rnn = nn.LSTM(piece_dim, out_dim // 2,
                               bidirectional=True, batch_first=True)
        else:
            self.proj = nn.Linear(piece_dim, out_dim)

    def forward(self, piece_ids: torch.Tensor, piece_lens: torch.Tensor) -> torch.Tensor:
        B, W, P = piece_ids.shape
        flat_ids = piece_ids.reshape(B * W, P)
        flat_lens = piece_lens.reshape(B * W)

        # Padded WORD positions have zero pieces. pack_padded_sequence rejects
        # length 0, so clamp to 1 and zero the output for those rows afterwards.
        real_word = flat_lens > 0
        safe_lens = flat_lens.clamp(min=1)

        x = self.dropout(self.embedding(flat_ids))

        if self.pooling == "bilstm":
            packed = pack_padded_sequence(
                x, safe_lens.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h, _) = self.rnn(packed)          # h: (2, B*W, out_dim//2)
            out = torch.cat([h[0], h[1]], dim=-1)  # (B*W, out_dim)
        else:
            pad_mask = (flat_ids != self.pad_id).unsqueeze(-1).float()
            summed = (x * pad_mask).sum(dim=1)
            out = self.proj(summed / pad_mask.sum(dim=1).clamp(min=1.0))

        out = out * real_word.unsqueeze(-1).to(out.dtype)
        return out.view(B, W, self.out_dim)


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
        # Phase 2, Piece 1: optional subword channel
        n_pieces: int = 0,
        piece_dim: int = 50,
        subword_dim: int = 100,
        subword_pooling: str = "bilstm",
        use_word_channel: bool = True,
        # Phase 2, Piece 2: optional sentence head
        sentence_head: bool = False,
        # Phase 2, Piece 3: token loss when not using the CRF
        token_loss=None,
    ):
        super().__init__()
        vocab_size, dim = embedding_matrix.shape

        if not use_word_channel and n_pieces == 0:
            raise ValueError("Cannot disable the word channel with no subword channel.")

        self.use_word_channel = use_word_channel
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = not freeze_embeddings

        # The two channels are CONCATENATED, not swapped. For the 80.8% of words
        # with a real fastText vector the word channel already works; for the
        # rest it is noise and the subword channel carries the signal. Keeping
        # both lets the model fall back rather than lose what works.
        self.subword = None
        lstm_input = dim if use_word_channel else 0
        if n_pieces > 0:
            self.subword = SubwordEncoder(
                n_pieces, piece_dim=piece_dim, out_dim=subword_dim,
                pooling=subword_pooling, dropout=dropout,
            )
            lstm_input += subword_dim
        self.lstm_input_dim = lstm_input

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            lstm_input, hidden_size, num_layers=num_layers,
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
        self.token_loss = token_loss

        # PIECE 2. A second output on the SAME encoder: is the whole tweet
        # offensive? Two views of one phenomenon, so forcing one BiLSTM to serve
        # both regularises it - valuable when token labels are only 4% positive
        # and we have 7,500 tweets.
        #
        # It is also the docking port for Piece 4. SemiSOLD's 145k extra tweets
        # carry SENTENCE-level teacher scores only, with no token labels, so
        # distillation cannot reach the token head directly. It trains this
        # head, and the shared encoder carries the benefit across.
        self.sentence_head = None
        if sentence_head:
            self.sentence_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 2),
            )

    # ------------------------------------------------------------------
    def emissions(self, ids, lengths, piece_ids=None, piece_lens=None) -> torch.Tensor:
        """Per-token scores, shape (batch, seq, num_labels)."""
        parts = []
        if self.use_word_channel:
            parts.append(self.dropout(self.embedding(ids)))
        if self.subword is not None:
            if piece_ids is None:
                raise ValueError("Model has a subword channel but got no piece ids. "
                                 "Pass sp=... to make_loader.")
            sub = self.subword(piece_ids, piece_lens)
            if sub.size(1) != ids.size(1):
                raise RuntimeError(
                    f"ALIGNMENT BROKEN: subword encoder returned {sub.size(1)} rows "
                    f"for {ids.size(1)} words. Labels would shift against words."
                )
            parts.append(sub)
        x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        self._last_encoder_out = out
        return self.classifier(self.dropout(out))

    def encode(self, ids, lengths, piece_ids=None, piece_lens=None):
        """Run the shared encoder and return its per-token states."""
        self.emissions(ids, lengths, piece_ids, piece_lens)
        return self._last_encoder_out

    def sentence_logits(self, ids, mask, lengths, piece_ids=None, piece_lens=None):
        """Masked mean-pool the encoder states, then classify the whole tweet."""
        if self.sentence_head is None:
            raise ValueError("Model has no sentence head. Pass sentence_head=True.")
        h = self.encode(ids, lengths, piece_ids, piece_lens)
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.sentence_head(pooled)

    # ------------------------------------------------------------------
    def loss(
        self,
        ids: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        piece_ids: Optional[torch.Tensor] = None,
        piece_lens: Optional[torch.Tensor] = None,
        sentence_labels: Optional[torch.Tensor] = None,
        sentence_lambda: float = 0.0,
    ) -> torch.Tensor:
        em = self.emissions(ids, lengths, piece_ids, piece_lens)

        if self.use_crf:
            # CRF needs real labels everywhere; padding is excluded by the mask.
            # It computes its own sequence likelihood and cannot take per-class
            # weights, which is why the Piece 3 loss comparison runs without it.
            safe = labels.clone()
            safe[~mask] = 0
            total = -self.crf(em, safe, mask=mask, reduction="mean")
        elif self.token_loss is not None:
            total = self.token_loss(em, labels)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
            total = loss_fn(em.reshape(-1, self.num_labels), labels.reshape(-1))

        # PIECE 2: add the sentence task, scaled by lambda.
        if self.sentence_head is not None and sentence_labels is not None:
            m = mask.unsqueeze(-1).to(self._last_encoder_out.dtype)
            pooled = (self._last_encoder_out * m).sum(1) / m.sum(1).clamp(min=1.0)
            slog = self.sentence_head(pooled)
            total = total + sentence_lambda * nn.functional.cross_entropy(
                slog, sentence_labels)
        return total

    # ------------------------------------------------------------------
    def distillation_loss(self, ids, mask, lengths, soft_targets,
                          piece_ids=None, piece_lens=None, temperature: float = 1.0):
        """PIECE 4. Learn from SemiSOLD's saved teacher scores.

        soft_targets is P(offensive) per tweet, precomputed by the SOLD authors
        in 2022 and stored as columns in a public file. No pretrained language
        model is ever loaded, run, or backpropagated through here - we consume a
        published artifact exactly as we consume the gold labels.

        These targets are SENTENCE level. There are no token labels in SemiSOLD,
        so this touches only the sentence head; the shared encoder is what
        carries any benefit to the token head.
        """
        slog = self.sentence_logits(ids, mask, lengths, piece_ids, piece_lens)
        logp = nn.functional.log_softmax(slog / temperature, dim=-1)
        tgt = torch.stack([1.0 - soft_targets, soft_targets], dim=-1)
        return -(tgt * logp).sum(dim=-1).mean() * (temperature ** 2)

    @torch.no_grad()
    def predict(
        self, ids: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor,
        piece_ids: Optional[torch.Tensor] = None,
        piece_lens: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Predicted label per real token, padding already stripped."""
        em = self.emissions(ids, lengths, piece_ids, piece_lens)
        if self.use_crf:
            return self.crf.decode(em, mask=mask)
        best = em.argmax(-1)
        return [[int(v) for v in row[: int(m.sum())]] for row, m in zip(best, mask)]

    # ------------------------------------------------------------------
    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        sub = sum(p.numel() for p in self.subword.parameters()) if self.subword else 0
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "word_embedding": self.embedding.weight.numel(),
            "subword_channel": sub,
            "sentence_head": sum(p.numel() for p in self.sentence_head.parameters())
                              if self.sentence_head is not None else 0,
            "lstm_input_dim": self.lstm_input_dim,
            "non_embedding_trainable": trainable - (
                self.embedding.weight.numel() if self.embedding.weight.requires_grad else 0
            ),
        }