"""
Vocabulary and pretrained word embeddings.

WHAT AN EMBEDDING IS
--------------------
A neural network cannot read text. It only does arithmetic. So every word
must become a list of numbers, called a vector. An embedding is that vector.

Words with similar meaning get similar vectors, because the vectors were
learned by reading billions of words of Sinhala text. We download them
already trained (fastText) instead of learning them from our 7,500 tweets,
which is far too little data to learn good vectors from scratch.

WHY fastText AND NOT word2vec/GloVe
-----------------------------------
fastText builds a word's vector out of its character chunks. Sinhala is
agglutinative - grammar is packed into word endings - so the same root shows
up in many surface forms. Character chunks let related forms share
information. Published work finds fastText beats word2vec and GloVe for
Sinhala for exactly this reason, and the SOLD paper's own results agree:
fastText scored 0.60 at token level against CBOW's 0.58.

HOW THE SOLD AUTHORS HANDLED UNKNOWN WORDS
------------------------------------------
From offensive_nn/offensive_nn_model.py, get_emb_matrix():

    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    # then overwrite the rows of words that WERE found

So the whole matrix starts as random numbers drawn to match the mean and
standard deviation of the real vectors, and found words overwrite their row.
Words not found keep a random vector that at least has the right scale.
We copy this exactly.
"""

from __future__ import annotations

import gzip
import io
import os
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

def build_vocab(
    token_lists: Iterable[Sequence[str]],
    min_freq: int = 1,
    max_size: Optional[int] = None,
) -> Tuple[Dict[str, int], Counter]:
    """Map each word to an integer id.

    Built from TRAINING data only. If we built it from test as well, the
    model would be getting information about the exam in advance.

    Returns (word -> id, raw frequency counter).
    """
    counts = Counter()
    for toks in token_lists:
        counts.update(toks)

    words = [w for w, c in counts.most_common() if c >= min_freq]
    if max_size is not None:
        words = words[: max_size - 2]  # leave room for PAD and UNK

    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for w in words:
        vocab[w] = len(vocab)
    return vocab, counts


def encode_tokens(tokens: Sequence[str], vocab: Dict[str, int]) -> List[int]:
    """Words to ids. Anything not in the vocabulary becomes UNK."""
    return [vocab.get(t, UNK_ID) for t in tokens]


# --------------------------------------------------------------------------
# reading a .vec file
# --------------------------------------------------------------------------

def _open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="ignore")
    return open(path, encoding="utf-8", errors="ignore")


def load_vectors_for_vocab(
    path: str,
    vocab: Dict[str, int],
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Stream a fastText .vec file and keep ONLY the words we need.

    The full Sinhala file has hundreds of thousands of words and is several
    GB. Loading all of it wastes memory. We only need the ~33,000 words in
    our vocabulary, so we read line by line and discard the rest.

    Returns (word -> vector, embedding dimension).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Embedding file not found: {path}\n"
            "Download the Sinhala fastText vectors first - see notebooks/03_embeddings.py"
        )

    wanted = set(vocab)
    found: Dict[str, np.ndarray] = {}
    dim = 0
    n_lines = 0

    with _open_maybe_gzip(path) as fh:
        header = fh.readline().split()
        if len(header) == 2:                    # standard "n_words dim" header
            dim = int(header[1])
        else:                                   # no header, first line is a vector
            parts = header
            dim = len(parts) - 1
            if parts[0] in wanted:
                found[parts[0]] = np.asarray(parts[1:], dtype=np.float32)

        for line in fh:
            n_lines += 1
            sp = line.rstrip().split(" ")
            if len(sp) != dim + 1:
                continue                        # malformed or word contains a space
            if sp[0] in wanted:
                found[sp[0]] = np.asarray(sp[1:], dtype=np.float32)
            if len(found) == len(wanted):
                break

    if verbose:
        print(f"  scanned {n_lines:,} lines, matched {len(found):,} of {len(wanted):,} vocab words")
    if not found:
        raise ValueError("No vocabulary word matched the embedding file. Wrong file or wrong language?")
    return found, dim


# --------------------------------------------------------------------------
# embedding matrix
# --------------------------------------------------------------------------

def build_embedding_matrix(
    vocab: Dict[str, int],
    vectors: Dict[str, np.ndarray],
    dim: int,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Row i of the returned matrix is the vector for word id i.

    Unknown words get a random vector drawn from the same mean and standard
    deviation as the real vectors. This copies get_emb_matrix() in the SOLD
    codebase. PAD is forced to all zeros so padding contributes nothing.
    """
    rng = np.random.default_rng(seed)
    all_vecs = np.stack(list(vectors.values()))
    mean, std = float(all_vecs.mean()), float(all_vecs.std())

    matrix = rng.normal(mean, std, size=(len(vocab), dim)).astype(np.float32)
    matrix[PAD_ID] = 0.0

    hit = 0
    for word, idx in vocab.items():
        vec = vectors.get(word)
        if vec is not None:
            matrix[idx] = vec
            hit += 1

    stats = {
        "vocab_size": len(vocab),
        "dim": dim,
        "found": hit,
        "missing": len(vocab) - hit,
        "coverage": hit / len(vocab),
        "init_mean": mean,
        "init_std": std,
    }
    return matrix, stats


# --------------------------------------------------------------------------
# coverage analysis
# --------------------------------------------------------------------------

def coverage_report(
    token_lists: Iterable[Sequence[str]],
    vocab: Dict[str, int],
    vectors: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """How often, at run time, do we actually hit a real vector?

    Two different questions, and they give very different answers:
      TYPE coverage  - what share of DISTINCT words are covered
      TOKEN coverage - what share of word OCCURRENCES are covered
    Token coverage is higher because common words are usually covered.
    Token coverage is what the model actually experiences.
    """
    types = set()
    n_tokens = 0
    tok_in_vocab = 0
    tok_has_vector = 0

    for toks in token_lists:
        for t in toks:
            types.add(t)
            n_tokens += 1
            if t in vocab:
                tok_in_vocab += 1
            if t in vectors:
                tok_has_vector += 1

    typ_in_vocab = sum(1 for t in types if t in vocab)
    typ_has_vector = sum(1 for t in types if t in vectors)

    return {
        "n_types": len(types),
        "n_tokens": n_tokens,
        "type_in_vocab": typ_in_vocab / len(types),
        "type_has_vector": typ_has_vector / len(types),
        "token_in_vocab": tok_in_vocab / n_tokens,
        "token_has_vector": tok_has_vector / n_tokens,
    }