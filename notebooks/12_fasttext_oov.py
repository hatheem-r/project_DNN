"""
fastText subword vectors for words without a .vec vector.

The baseline gives every vocabulary word missing from cc.si.300.vec a random
vector (copied from SOLD's get_emb_matrix), and maps test words unseen in
training to UNK. The full fastText model (cc.si.300.bin) can instead build a
vector for any string from its character n-grams. This script writes two
matrices that use it:

    artifacts/embedding_matrix_ftfill.npy
        Same vocabulary as artifacts/embedding_matrix.npy (the 6,000-tweet
        training part). Rows that hold a real .vec vector are kept as they are;
        the random rows are replaced by the fastText n-gram vector. Unseen test
        words still map to UNK.

    artifacts/embedding_matrix_ftopen.npy
        The same rows, plus one row for every validation and test word not in
        the training vocabulary, again from fastText. This is what running
        fastText on each word at inference time gives, so no test word is UNK.
        The vectors are frozen and computed from the word string alone; no
        labels are used. Train with 07_subword_model.py --open-vocab.

Needs artifacts/embedding_matrix.npy (notebooks/03_embeddings.py) and the
uncompressed binary model:

    wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.si.300.bin.gz
    gunzip cc.si.300.bin.gz

Run:
    python notebooks/12_fasttext_oov.py --bin embeddings/cc.si.300.bin
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from collections import Counter

import numpy as np

from data import load_sold, train_val_split
from embeddings import build_vocab, extend_vocab, PAD_ID, UNK_ID

BASE = "artifacts/embedding_matrix.npy"
FILL = "artifacts/embedding_matrix_ftfill.npy"
OPEN = "artifacts/embedding_matrix_ftopen.npy"


def rule(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


ap = argparse.ArgumentParser()
ap.add_argument("--bin", required=True, help="path to the uncompressed cc.si.300.bin")
args = ap.parse_args()

import fasttext

# ==========================================================================
rule("1. LOAD")
train_full = load_sold("train")
test = load_sold("test")
train_part, val = train_val_split(train_full)
vocab, _ = build_vocab(train_part["token_list"], min_freq=1)
base = np.load(BASE)
assert base.shape[0] == len(vocab), "embedding_matrix.npy does not match the vocabulary"
ft = fasttext.load_model(args.bin)
dim = ft.get_dimension()
assert dim == base.shape[1], f"dimension mismatch: .bin {dim}, matrix {base.shape[1]}"
print(f"training-part vocabulary {len(vocab):,}   matrix {base.shape}")
print(f"fastText model: {len(ft.words):,} words, dimension {dim}")

# ==========================================================================
rule("2. WHICH ROWS ARE RANDOM")
# A row holds a real vector if it matches the .bin vector for that word; the
# .vec file stores the same vectors to four decimal places.
real, random_rows = [], []
for w, i in vocab.items():
    if i in (PAD_ID, UNK_ID):
        continue
    if np.allclose(base[i], ft.get_word_vector(w), atol=1e-3):
        real.append(w)
    else:
        random_rows.append(w)
n_words = len(vocab) - 2
in_ft_vocab = sum(1 for w in random_rows if ft.get_word_id(w) >= 0)
print(f"real .vec vector   {len(real):,} ({len(real) / n_words:.1%})")
print(f"random vector      {len(random_rows):,} ({len(random_rows) / n_words:.1%})")
print(f"  of the random rows, words in fastText's own vocabulary: {in_ft_vocab:,}")
print(f"  the rest get a vector built only from character n-grams")
# 03_embeddings.py reports 80.8% coverage. A much lower share means the .bin
# and .vec vectors do not match, and real rows would be overwritten.
assert len(real) / n_words > 0.7, "too few rows match the .bin model - wrong file?"

# ==========================================================================
rule("3. MATRICES")
fill = base.copy()
for w in random_rows:
    fill[vocab[w]] = ft.get_word_vector(w)
np.save(FILL, fill)
print(f"{FILL}   {fill.shape}   ({len(random_rows):,} rows replaced)")

open_vocab = extend_vocab(vocab, [*val["token_list"], *test["token_list"]])
new_words = [w for w, i in sorted(open_vocab.items(), key=lambda kv: kv[1]) if i >= len(vocab)]
extra = np.stack([ft.get_word_vector(w) for w in new_words]).astype(np.float32)
opened = np.concatenate([fill, extra])
assert opened.shape[0] == len(open_vocab)
np.save(OPEN, opened)
print(f"{OPEN}   {opened.shape}   ({len(new_words):,} validation/test words added)")

# ==========================================================================
rule("4. TEST WORD OCCURRENCES BY SOURCE OF THEIR VECTOR")
real_set, rand_set = set(real), set(random_rows)
c = Counter()
for toks in test["token_list"]:
    for t in toks:
        c["real" if t in real_set else "random" if t in rand_set else "unseen"] += 1
n = sum(c.values())
print(f"test word occurrences: {n:,}")
print(f"  real .vec vector                 {c['real'] / n:6.1%}   all three matrices")
print(f"  random in baseline               {c['random'] / n:6.1%}   fastText in ftfill and ftopen")
print(f"  unseen in training (UNK)         {c['unseen'] / n:6.1%}   fastText in ftopen only")

# ==========================================================================
rule("5. SANITY CHECK - NEIGHBOURS OF REPLACED WORDS")
print("Nearest real-vector words to the most frequent replaced words. Related")
print("forms here mean the n-gram vectors carry information.\n")
freq = Counter(t for toks in train_part["token_list"] for t in toks)
V = np.stack([base[vocab[w]] for w in real])
V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
for w in sorted(random_rows, key=lambda x: -freq[x])[:8]:
    v = fill[vocab[w]]
    sims = V @ (v / (np.linalg.norm(v) + 1e-9))
    top = np.argsort(-sims)[:5]
    print(f"  {w:<18} -> {', '.join(real[i] for i in top)}")
