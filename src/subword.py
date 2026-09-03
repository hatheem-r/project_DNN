"""
Subword tokenization for Sinhala.

WHY WE NEED THIS
----------------
Measured in Step 3:
  - 48.7% of the DISTINCT words in the test set never appeared in training
  - 14.3% of test word OCCURRENCES never appeared in training
  - fastText's nearest neighbours in Sinhala are morphological variants, not
    synonyms:  සක්කිලි -> සක්කිලිය, සක්කිලියා, සක්කිලියෙක්, සක්කිලියන්ගේ

Our word-level model treats those five as five unrelated ID numbers. Sinhala is
agglutinative: grammar is packed into word endings, so one root generates many
surface forms. A model that cannot see inside a word is blind to that.

WHAT SUBWORDS DO
----------------
Break each word into smaller pieces that are shared across forms:

    සක්කිලියා  ->  ["සක්කිලි", "යා"]
    සක්කිලියෙක් ->  ["සක්කිලි", "යෙක්"]

Now a form the model has never seen still shares a piece with forms it knows.

TOOL
----
SentencePiece (Kudo & Richardson, EMNLP 2018). It learns the pieces from raw
text with no linguistic rules and no word list. Two algorithms:

  unigram  (default) - probabilistic; keeps pieces that best explain the corpus
  bpe                - greedy; repeatedly merges the most frequent pair

We compare both. For morphologically rich languages unigram often wins, but
that is a hypothesis to test, not an assumption.

LEAKAGE RULE
------------
The tokenizer is trained on TRAIN TEXT ONLY. It never sees validation or test.
Training it on all the text would leak information about the exam - the model
would get pieces tuned to words it is not supposed to know exist.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import sentencepiece as spm

PIECE_PAD = "<pad>"
PIECE_UNK = "<unk>"


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train_sentencepiece(
    texts: Iterable[str],
    vocab_size: int,
    model_prefix: str,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
) -> str:
    """Learn a subword vocabulary from raw text. Returns the .model path.

    character_coverage 0.9995 is the SentencePiece recommendation for
    non-Latin scripts: cover 99.95% of characters seen, treat the rest as
    unknown. Sinhala has a large character set, so full coverage would waste
    vocabulary slots on characters appearing once.
    """
    os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)
    corpus = f"{model_prefix}.corpus.txt"
    with open(corpus, "w", encoding="utf-8") as fh:
        for t in texts:
            fh.write(t.strip() + "\n")

    spm.SentencePieceTrainer.train(
        input=corpus,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=0, unk_id=1, bos_id=-1, eos_id=-1,
        pad_piece=PIECE_PAD, unk_piece=PIECE_UNK,
        # If the corpus cannot support the requested size, silently use the
        # largest it can instead of raising. Small corpora hit this; we then
        # report the size actually achieved.
        hard_vocab_limit=False,
        # SentencePiece prints hundreds of lines of training log. Silence it.
        minloglevel=2,
    )
    os.remove(corpus)
    return f"{model_prefix}.model"


def load_sentencepiece(model_path: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


# --------------------------------------------------------------------------
# applying it
# --------------------------------------------------------------------------

def word_to_pieces(sp, word: str) -> List[str]:
    """Split ONE word into pieces.

    We encode word by word rather than the whole sentence, because our labels
    are one per word. Encoding the sentence would let SentencePiece merge
    across word boundaries and the alignment would break.
    """
    return sp.encode(word, out_type=str)


def word_to_piece_ids(sp, word: str) -> List[int]:
    return sp.encode(word, out_type=int)


def tweet_to_piece_ids(sp, tokens: Sequence[str]) -> List[List[int]]:
    """One list of piece ids per word. Length always equals len(tokens).

    This invariant is what keeps labels aligned. Never return a flat list.
    """
    out = [word_to_piece_ids(sp, w) or [sp.unk_id()] for w in tokens]
    assert len(out) == len(tokens)
    return out


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------

def fragmentation(sp, token_lists: Iterable[Sequence[str]]) -> Dict[str, float]:
    """How many pieces does the average word become?

    Too high means words are shredded into meaningless fragments and the model
    must reassemble them. Too low means we are back at word level and gained
    nothing.
    """
    n_words = n_pieces = 0
    dist = Counter()
    for toks in token_lists:
        for w in toks:
            k = max(len(word_to_pieces(sp, w)), 1)
            n_words += 1
            n_pieces += k
            dist[min(k, 6)] += 1
    return {
        "words": n_words,
        "pieces": n_pieces,
        "pieces_per_word": n_pieces / max(n_words, 1),
        "pct_1_piece": dist[1] / max(n_words, 1),
        "pct_2_pieces": dist[2] / max(n_words, 1),
        "pct_3plus": sum(dist[k] for k in range(3, 7)) / max(n_words, 1),
    }


def unseen_word_rescue(
    sp,
    train_tokens: Iterable[Sequence[str]],
    eval_tokens: Iterable[Sequence[str]],
) -> Dict[str, float]:
    """THE NUMBER THAT JUSTIFIES THIS WHOLE COMPONENT.

    Take the evaluation words that never appeared in training - words the
    word-level model has nothing at all for. What fraction of THEIR PIECES did
    appear in training?

    High means subwords rescue unseen words, because the pieces are familiar
    even though the whole word is not. That is the mechanism, measured.
    """
    train_words = set()
    train_pieces = set()
    for toks in train_tokens:
        for w in toks:
            train_words.add(w)
            train_pieces.update(word_to_pieces(sp, w))

    unseen_types = set()
    n_tok = n_unseen_tok = 0
    piece_hits = piece_total = 0
    fully_covered = 0

    for toks in eval_tokens:
        for w in toks:
            n_tok += 1
            if w in train_words:
                continue
            n_unseen_tok += 1
            unseen_types.add(w)
            pcs = word_to_pieces(sp, w)
            hits = sum(1 for p in pcs if p in train_pieces)
            piece_hits += hits
            piece_total += len(pcs)
            if pcs and hits == len(pcs):
                fully_covered += 1

    return {
        "unseen_types": len(unseen_types),
        "unseen_token_rate": n_unseen_tok / max(n_tok, 1),
        "piece_coverage_of_unseen": piece_hits / max(piece_total, 1),
        "unseen_fully_covered": fully_covered / max(n_unseen_tok, 1),
    }


def shared_root_check(sp, words: Sequence[str]) -> List[Tuple[str, List[str]]]:
    """Show how a group of related word forms gets split.

    If subwords work for Sinhala, morphological variants of one root should
    share their FIRST piece.
    """
    return [(w, word_to_pieces(sp, w)) for w in words]


def build_piece_vocab(sp) -> Dict[str, int]:
    return {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
