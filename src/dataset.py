"""
Turning tweets into tensors.

A neural network eats fixed-shape blocks of numbers. Tweets have different
lengths, and now words have different numbers of subword pieces. This file
bridges all of that.

PADDING. To put 32 tweets of different lengths into one block we pad the short
ones with a filler token (id 0) up to the longest tweet IN THAT BATCH. Padding
to the batch maximum rather than a global maximum wastes almost no computation.

MASKING. Padding positions are not real words. They must be excluded from the
loss and from the metric, or the model earns credit for labelling empty space.

NO TRUNCATION. The longest tweet is 134 tokens and the 99th percentile is 79.
Truncating at 80 would silently delete the labels of about 1% of tweets, and
with dynamic padding the long ones cost almost nothing.

TWO LEVELS OF PADDING (new in Phase 2)
--------------------------------------
With subwords there are now two ragged dimensions:

    tweets have different numbers of WORDS
    words have different numbers of PIECES

So piece ids form a 3-D block (batch, words, pieces), padded on both axes.

    tweet:   [ @USER   සක්කිලියා   යනවා ]
    pieces:  [[47],   [4102, 88],  [219]]
    padded:  [[47, 0],[4102, 88], [219, 0]]
             shape (3 words, 2 pieces)

THE INVARIANT THAT MUST NEVER BREAK
-----------------------------------
Labels are one per WORD. Pieces are smaller than words. The piece block must
therefore have exactly n_words rows for a tweet with n_words words. If it does
not, labels shift relative to words: the model still trains, the loss still
falls, and the score is quietly terrible with nothing crashing.

tests/test_subword_alignment.py asserts this on every training tweet.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from embeddings import PAD_ID, encode_tokens

IGNORE_LABEL = -100   # positions the loss must skip
PIECE_PAD_ID = 0      # SentencePiece is trained with pad_id=0


class SOLDTokenDataset(Dataset):
    """One item = one tweet.

    Returns (word_ids, token_labels, sentence_label, piece_ids_per_word).
    piece_ids_per_word is None when no subword tokenizer is in use.
    """

    def __init__(self, df, vocab: Dict[str, int], sp=None):
        self.ids: List[List[int]] = [encode_tokens(t, vocab) for t in df["token_list"]]
        self.labels: List[List[int]] = [list(r) for r in df["rationales"]]
        self.sent: List[int] = [1 if l == "OFF" else 0 for l in df["label"]]

        self.pieces: Optional[List[List[List[int]]]] = None
        if sp is not None:
            from subword import tweet_to_piece_ids
            self.pieces = [tweet_to_piece_ids(sp, t) for t in df["token_list"]]

        for i, (a, b) in enumerate(zip(self.ids, self.labels)):
            if len(a) != len(b):
                raise ValueError(f"row {i}: {len(a)} words vs {len(b)} labels")
            if self.pieces is not None and len(self.pieces[i]) != len(a):
                raise ValueError(
                    f"row {i}: {len(self.pieces[i])} piece-lists vs {len(a)} words. "
                    "The subword alignment invariant is broken."
                )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i):
        pcs = self.pieces[i] if self.pieces is not None else None
        return self.ids[i], self.labels[i], self.sent[i], pcs


def collate(batch):
    """Pad a list of tweets into rectangular blocks.

    Always returns 7 items so every caller has one code path:
        word_ids     (B, W)      long
        labels       (B, W)      long, IGNORE_LABEL on padding
        mask         (B, W)      bool, True on real words
        lengths      (B,)        long, words per tweet
        sent         (B,)        long, sentence label
        piece_ids    (B, W, P)   long, or None
        piece_lens   (B, W)      long, or None   (0 for padded words)
    """
    B = len(batch)
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    W = int(lengths.max())

    word_ids = torch.full((B, W), PAD_ID, dtype=torch.long)
    labels = torch.full((B, W), IGNORE_LABEL, dtype=torch.long)
    mask = torch.zeros((B, W), dtype=torch.bool)

    for i, (tok, lab, _, _) in enumerate(batch):
        n = len(tok)
        word_ids[i, :n] = torch.tensor(tok, dtype=torch.long)
        labels[i, :n] = torch.tensor(lab, dtype=torch.long)
        mask[i, :n] = True

    sent = torch.tensor([x[2] for x in batch], dtype=torch.long)

    if batch[0][3] is None:
        return word_ids, labels, mask, lengths, sent, None, None

    P = max((len(p) for x in batch for p in x[3]), default=1)
    P = max(P, 1)
    piece_ids = torch.full((B, W, P), PIECE_PAD_ID, dtype=torch.long)
    piece_lens = torch.zeros((B, W), dtype=torch.long)

    for i, (_, _, _, pcs) in enumerate(batch):
        for j, p in enumerate(pcs):
            k = len(p)
            if k:
                piece_ids[i, j, :k] = torch.tensor(p, dtype=torch.long)
                piece_lens[i, j] = k

    return word_ids, labels, mask, lengths, sent, piece_ids, piece_lens


def make_loader(df, vocab, batch_size: int, shuffle: bool, generator=None,
                sp=None) -> DataLoader:
    """sp is a loaded SentencePiece processor, or None for word-level only."""
    return DataLoader(
        SOLDTokenDataset(df, vocab, sp=sp),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        generator=generator,
    )


def unpad_predictions(preds, mask) -> List[List[int]]:
    """Strip padding so the metric only ever sees real tokens."""
    out = []
    for row, m in zip(preds, mask):
        n = int(m.sum())
        out.append([int(v) for v in row[:n]])
    return out