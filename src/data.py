"""
SOLD data loading and normalisation.

Single source of truth for how the dataset enters the project.
Nothing else in the codebase should call load_dataset() directly.

Actual dataset columns (verified against the HuggingFace data viewer,
NOT the GitHub README, which contains two errors):

    post_id     int64    Twitter ID
    text        string   raw post text
    tokens      string   tokenised text, tokens separated by a space
    rationales  string   stringified list, e.g. "[0, 1, 0]"
    label       string   "OFF" or "NOT"

TWO GOTCHAS, both discovered from the real data:

 1. The column is spelled `rationales`. The GitHub README says `rationals`.
    We accept either, plus a couple of other plausible spellings.

 2. NOT-offensive tweets carry an EMPTY list `[]`, not a list of zeros.
    Only OFF tweets have a full-length 0/1 vector. So `[]` must be
    expanded to [0] * n_tokens before training or evaluation. Failing to
    do this silently drops every negative example.
"""

from __future__ import annotations

import ast
import json
from typing import List

import numpy as np
import pandas as pd

DATASET_NAME = "sinhala-nlp/SOLD"

# Accepted spellings for the token-label column, in priority order.
RATIONALE_COLUMN_CANDIDATES = ("rationales", "rationals", "rationale", "rational")


# --------------------------------------------------------------------------
# column discovery
# --------------------------------------------------------------------------

def find_rationale_column(df: pd.DataFrame) -> str:
    for name in RATIONALE_COLUMN_CANDIDATES:
        if name in df.columns:
            return name
    raise KeyError(
        f"No rationale column found. Columns present: {list(df.columns)}. "
        f"Tried: {RATIONALE_COLUMN_CANDIDATES}"
    )


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def parse_rationales(value) -> List[int]:
    """Coerce one rationale cell into a plain list of 0/1 ints.

    Returns [] for the empty case; expansion to zeros happens later, in
    expand_rationales(), so that the empty case stays visible for auditing.
    """
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]

    if isinstance(value, str):
        s = value.strip()
        if s in ("", "[]", "()"):
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(s)
                if isinstance(parsed, (list, tuple)):
                    return [int(v) for v in parsed]
            except Exception:
                pass
        cleaned = s.strip("[]()").replace(",", " ")
        return [int(float(p)) for p in cleaned.split() if p != ""]

    raise TypeError(f"Unrecognised rationale type: {type(value)!r}")


def split_tokens(value) -> List[str]:
    """`tokens` is documented as space-separated. Handle the list case too."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(t) for t in value]
    return str(value).split()


def expand_rationales(raw: List[int], n_tokens: int) -> List[int]:
    """Turn an empty rationale list into an all-zero vector of the right length.

    Non-empty lists are returned unchanged; length problems are reported by
    the `length_mismatch` flag rather than silently patched here.
    """
    if len(raw) == 0:
        return [0] * n_tokens
    return raw


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_sold(split: str) -> pd.DataFrame:
    """Load one official SOLD split with normalised, audited columns.

    Adds:
        token_list       List[str]
        rationales_raw   List[int]  exactly as stored (may be empty)
        rationales       List[int]  expanded to len(token_list)
        n_tokens         int
        raw_empty        bool  rationale cell was []
        length_mismatch  bool  non-empty rationale whose length != n_tokens
        n_offensive_tokens int
    """
    from datasets import Dataset, load_dataset  # imported lazily

    df = Dataset.to_pandas(load_dataset(DATASET_NAME, split=split))

    col = find_rationale_column(df)
    if col != "rationales":
        df = df.rename(columns={col: "rationales"})
    df.attrs["source_rationale_column"] = col

    df["token_list"] = df["tokens"].apply(split_tokens)
    df["n_tokens"] = df["token_list"].apply(len)

    df["rationales_raw"] = df["rationales"].apply(parse_rationales)
    df["raw_empty"] = df["rationales_raw"].apply(lambda r: len(r) == 0)
    df["length_mismatch"] = [
        (len(r) > 0) and (len(r) != n)
        for r, n in zip(df["rationales_raw"], df["n_tokens"])
    ]
    df["rationales"] = [
        expand_rationales(r, n) for r, n in zip(df["rationales_raw"], df["n_tokens"])
    ]
    df["n_offensive_tokens"] = df["rationales"].apply(sum)
    return df


# --------------------------------------------------------------------------
# train / validation split
# --------------------------------------------------------------------------
# SOLD ships only train and test. The validation split is ours, carved out of
# train. EVERY component that needs to choose a hyperparameter must use THIS
# function with THIS seed, so the lexicon baseline, the BiLSTM, and every later
# model are all tuned and compared on identical data.
#
# The test split is never touched until a configuration is frozen.

VAL_SEED = 42
VAL_FRACTION = 0.2


def train_val_split(train_df, val_fraction: float = VAL_FRACTION, seed: int = VAL_SEED):
    """Split the official train split into train-part and validation.

    Deterministic: the same seed always produces the same split. Record the
    seed in the paper.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_df))
    n_val = int(len(train_df) * val_fraction)
    val = train_df.iloc[idx[:n_val]].reset_index(drop=True)
    tr = train_df.iloc[idx[n_val:]].reset_index(drop=True)
    return tr, val


def load_splits(drop_length_mismatch: bool = False):
    """Load train and test.

    drop_length_mismatch: only enable after you have counted these rows in
    the exploration report and recorded the decision in the README.
    """
    train = load_sold("train")
    test = load_sold("test")
    if drop_length_mismatch:
        train = train[~train.length_mismatch].reset_index(drop=True)
        test = test[~test.length_mismatch].reset_index(drop=True)
    return train, test