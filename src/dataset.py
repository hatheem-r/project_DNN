"""
Turning tweets into tensors.

A neural network eats fixed-shape blocks of numbers. Tweets have different
lengths. This file bridges the two.

PADDING. To put 32 tweets of different lengths into one block, we pad the
short ones with a filler token (id 0) up to the length of the longest tweet
IN THAT BATCH. Padding to the batch maximum rather than a global maximum
means we waste almost no computation.

MASKING. Padding positions are not real words. They must be excluded from
the loss and from the metric, or the model gets credit for correctly
labelling empty space. The mask records which positions are real.

NO TRUNCATION. We do not cut long tweets. The longest is 134 tokens and the
99th percentile is 79. Truncating at 80 would silently delete the labels of
about 1% of tweets, and with dynamic padding the long ones cost us almost
nothing anyway.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from embeddings import PAD_ID, encode_tokens

IGNORE_LABEL = -100  # positions the loss must skip


class SOLDTokenDataset(Dataset):
    """One item = one tweet: word ids, token labels, sentence label."""

    def __init__(self, df, vocab: Dict[str, int]):
        self.ids: List[List[int]] = [encode_tokens(t, vocab) for t in df["token_list"]]
        self.labels: List[List[int]] = [list(r) for r in df["rationales"]]
        self.sent: List[int] = [1 if l == "OFF" else 0 for l in df["label"]]
        for i, (a, b) in enumerate(zip(self.ids, self.labels)):
            if len(a) != len(b):
                raise ValueError(f"row {i}: {len(a)} tokens vs {len(b)} labels")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i], self.sent[i]


def collate(batch):
    """Pad a list of tweets into one rectangular block."""
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    n_max = int(lengths.max())

    ids = torch.full((len(batch), n_max), PAD_ID, dtype=torch.long)
    labels = torch.full((len(batch), n_max), IGNORE_LABEL, dtype=torch.long)
    mask = torch.zeros((len(batch), n_max), dtype=torch.bool)

    for i, (tok, lab, _) in enumerate(batch):
        n = len(tok)
        ids[i, :n] = torch.tensor(tok, dtype=torch.long)
        labels[i, :n] = torch.tensor(lab, dtype=torch.long)
        mask[i, :n] = True

    sent = torch.tensor([x[2] for x in batch], dtype=torch.long)
    return ids, labels, mask, lengths, sent


def make_loader(df, vocab, batch_size: int, shuffle: bool, generator=None) -> DataLoader:
    return DataLoader(
        SOLDTokenDataset(df, vocab),
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