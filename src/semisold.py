"""
SemiSOLD: 145,000 extra Sinhala tweets with PRE-COMPUTED teacher scores.

WHY THIS DOES NOT BREAK THE NO-PLM RULE
---------------------------------------
The SOLD authors trained eleven classifiers on SOLD in 2022, ran them over
these 145,000 unlabeled tweets, and saved each model's confidence for the
offensive class as columns in a public file.

We read those saved numbers. No pretrained language model is ever loaded, run,
or backpropagated through inside our network. We consume a published artifact,
exactly as we consume the gold labels.

THE CONSTRAINT THAT SHAPES THE DESIGN
-------------------------------------
Those scores are SENTENCE level. SemiSOLD has no token annotations of any kind.
The SOLD authors say so themselves: their co-learning approach "does not readily
apply" to token classification because the label space grows exponentially with
sequence length, so they never trained supervised token models on augmented data.

So distillation cannot teach our token head directly. It teaches the sentence
head (09_pieces_234.py --piece2), and because both heads sit on one shared
BiLSTM, the sentence-level signal reaches the shared representation.

UNCERTAINTY FILTERING - USE THE AUTHORS' OWN FINDING
----------------------------------------------------
They filtered by disagreement among the top models and found a standard
deviation threshold of 0.1 best (8,474 instances with their teachers). At 0.15
(47,746 instances) most of their models scored lower. We default to 0.1.

OVERLAP WITH SOLD
-----------------
The public release contains nearly every SOLD training and test tweet, with
different spacing and punctuation. drop_overlap() removes them (10,087 rows);
09_pieces_234.py --exclude-sold applies it before filtering.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from embeddings import PAD_ID, encode_tokens

DATASET_NAME = "sinhala-nlp/SemiSOLD"

# The eleven classifiers listed in the SOLD paper. We auto-detect whichever are
# actually present rather than assuming the schema.
TEACHERS = ["xlmr", "xlmt", "mbert", "sinbert", "lstm_ft", "cnn_ft",
            "lstm_cbow", "cnn_cbow", "lstm_sl", "cnn_sl", "svm"]
# The strongest three, used for both the soft target and the disagreement measure.
TOP_TEACHERS = ["xlmr", "xlmt", "sinbert"]


def load_semisold() -> pd.DataFrame:
    from datasets import Dataset as HFDataset, load_dataset
    return HFDataset.to_pandas(load_dataset(DATASET_NAME, split="train"))


def match_key(text: str) -> str:
    """A tweet with user mentions, URLs, punctuation, symbols and whitespace
    removed. SOLD stores tokenised text and SemiSOLD raw text, so the same tweet
    differs in spacing and punctuation between the two. The zero-width joiner is
    kept because it is part of Sinhala letters."""
    s = re.sub(r"@USER|URL|https?\S+", "", str(text))
    return "".join(c for c in s
                   if c == "‍" or not (c.isspace() or unicodedata.category(c)[0] in "PSZC"))


def drop_overlap(df: pd.DataFrame, sold_texts: Iterable[str],
                 text_col: str = "text") -> pd.DataFrame:
    """Remove SemiSOLD rows that are SOLD tweets. The public SemiSOLD release
    contains nearly all SOLD training and test tweets."""
    keys = {match_key(t) for t in sold_texts}
    keys.discard("")
    hit = df[text_col].map(match_key).isin(keys)
    print(f"  removed {int(hit.sum()):,} SemiSOLD rows that match a SOLD tweet; "
          f"{int((~hit).sum()):,} remain")
    return df[~hit].reset_index(drop=True)


def describe(df: pd.DataFrame) -> None:
    print(f"rows: {len(df):,}")
    print("columns:", list(df.columns))
    found = [c for c in TEACHERS if c in df.columns]
    print(f"teacher score columns found: {len(found)} of {len(TEACHERS)}  {found}")
    missing = [c for c in TEACHERS if c not in df.columns]
    if missing:
        print(f"  not present: {missing}")


def prepare(
    df: pd.DataFrame,
    uncertainty_threshold: float = 0.1,
    teachers: Optional[Sequence[str]] = None,
    text_col: Optional[str] = None,
) -> pd.DataFrame:
    """Filter by teacher disagreement and compute one soft target per tweet.

    Returns a frame with token_list and soft_target, ready for the loader.
    """
    teachers = [c for c in (teachers or TOP_TEACHERS) if c in df.columns]
    if not teachers:
        teachers = [c for c in TEACHERS if c in df.columns][:3]
    if not teachers:
        raise ValueError(f"No teacher score columns found. Columns: {list(df.columns)}")

    scores = df[teachers].astype(float)
    out = df.copy()
    out["soft_target"] = scores.mean(axis=1)
    out["uncertainty"] = scores.std(axis=1)

    kept = out[out["uncertainty"] <= uncertainty_threshold].copy()

    if text_col is None:
        text_col = "tokens" if "tokens" in kept.columns else "text"
    kept["token_list"] = kept[text_col].astype(str).str.split()
    kept = kept[kept["token_list"].apply(len) > 0].reset_index(drop=True)

    print(f"  teachers used: {teachers}")
    print(f"  {len(df):,} rows -> {len(kept):,} kept at uncertainty <= {uncertainty_threshold}")
    print(f"  soft target mean {kept['soft_target'].mean():.3f}, "
          f"{(kept['soft_target'] > 0.5).mean():.1%} lean offensive")
    return kept


class SemiSOLDDataset(Dataset):
    """Unlabeled tweets with one sentence-level soft target each."""

    def __init__(self, df, vocab, sp=None):
        self.ids: List[List[int]] = [encode_tokens(t, vocab) for t in df["token_list"]]
        self.soft = df["soft_target"].astype(float).tolist()
        self.pieces = None
        if sp is not None:
            from subword import tweet_to_piece_ids
            self.pieces = [tweet_to_piece_ids(sp, t) for t in df["token_list"]]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pcs = self.pieces[i] if self.pieces is not None else None
        return self.ids[i], self.soft[i], pcs


def collate_semisold(batch):
    B = len(batch)
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    W = int(lengths.max())

    ids = torch.full((B, W), PAD_ID, dtype=torch.long)
    mask = torch.zeros((B, W), dtype=torch.bool)
    for i, (tok, _, _) in enumerate(batch):
        n = len(tok)
        ids[i, :n] = torch.tensor(tok, dtype=torch.long)
        mask[i, :n] = True
    soft = torch.tensor([x[1] for x in batch], dtype=torch.float)

    if batch[0][2] is None:
        return ids, mask, lengths, None, None, soft

    P = max(max((len(p) for p in x[2]), default=1) for x in batch)
    pid = torch.zeros((B, W, P), dtype=torch.long)
    plen = torch.zeros((B, W), dtype=torch.long)
    for i, (_, _, pcs) in enumerate(batch):
        for j, p in enumerate(pcs):
            if p:
                pid[i, j, :len(p)] = torch.tensor(p, dtype=torch.long)
                plen[i, j] = len(p)
    return ids, mask, lengths, pid, plen, soft


def make_semisold_loader(df, vocab, batch_size, sp=None, generator=None, shuffle=True):
    return DataLoader(
        SemiSOLDDataset(df, vocab, sp=sp),
        batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_semisold, generator=generator,
    )
