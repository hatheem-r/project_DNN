"""
Phase 1 / Step 3 - vocabulary and pretrained embeddings.

BEFORE RUNNING: download the Sinhala fastText vectors.

  Go to https://fasttext.cc/docs/en/crawl-vectors.html and find Sinhala.
  The text-format file is cc.si.300.vec.gz (a few hundred MB compressed).

    mkdir -p embeddings && cd embeddings
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.si.300.vec.gz

  Do NOT unzip it - the loader reads .gz directly.
  Add embeddings/ to .gitignore. Never commit a multi-hundred-MB file.

  If that URL has moved, get the current link from the fasttext.cc page
  above, or use the HuggingFace mirror facebook/fasttext-si-vectors.

Run:  python notebooks/03_embeddings.py > results/step3_report.txt
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from data import load_sold, train_val_split
from embeddings import (
    build_vocab, load_vectors_for_vocab, build_embedding_matrix,
    coverage_report, PAD_TOKEN, UNK_TOKEN,
)

VEC_PATH = os.environ.get("SOLD_VECTORS", "embeddings/cc.si.300.vec.gz")
MIN_FREQS = [1, 2, 3, 5]


def rule(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# ==========================================================================
rule("0. DATA")
train_full = load_sold("train")
test = load_sold("test")
train_part, val = train_val_split(train_full)
print(f"train-part {len(train_part):,}   validation {len(val):,}   test {len(test):,}")
print("Vocabulary is built from TRAIN-PART only. Building it from validation")
print("or test would leak information about data the model must not see.")


# ==========================================================================
rule("1. VOCABULARY SIZE vs MINIMUM FREQUENCY")
print("""min_freq drops rare words from the vocabulary. Every dropped word becomes
<UNK> at run time. Dropping too many loses information; keeping everything
makes a huge embedding matrix full of words seen once.
""")
print(f"{'min_freq':>9} {'vocab':>9} {'train tok cov':>14} {'val tok cov':>13}")
print("-" * 49)
vocabs = {}
for mf in MIN_FREQS:
    v, counts = build_vocab(train_part["token_list"], min_freq=mf)
    vocabs[mf] = v
    tr_cov = sum(1 for toks in train_part["token_list"] for t in toks if t in v)
    tr_tot = sum(len(t) for t in train_part["token_list"])
    va_cov = sum(1 for toks in val["token_list"] for t in toks if t in v)
    va_tot = sum(len(t) for t in val["token_list"])
    print(f"{mf:>9} {len(v):>9,} {tr_cov/tr_tot:>13.2%} {va_cov/va_tot:>12.2%}")

print("""
Note the gap between the two columns. Training coverage is near perfect by
construction. Validation coverage is what matters, because it is what the
model faces on data it has not memorised.""")

MIN_FREQ = 1
vocab = vocabs[MIN_FREQ]
print(f"\nUsing min_freq={MIN_FREQ} for the baseline (matches keeping all training")
print("words). This is a hyperparameter - tune it on validation in Step 5, not now.")
print(f"vocab size {len(vocab):,}  (ids 0={PAD_TOKEN}, 1={UNK_TOKEN})")


# ==========================================================================
rule("2. LOAD fastText VECTORS")
print(f"file: {VEC_PATH}")
if not os.path.exists(VEC_PATH):
    print("\nFILE NOT FOUND. See the download instructions at the top of this file.")
    print("Set SOLD_VECTORS=/path/to/cc.si.300.vec.gz if it lives elsewhere.")
    sys.exit(1)

vectors, dim = load_vectors_for_vocab(VEC_PATH, vocab)
print(f"  dimension: {dim}")


# ==========================================================================
rule("3. EMBEDDING MATRIX")
matrix, stats = build_embedding_matrix(vocab, vectors, dim)
print(f"  shape          {matrix.shape}")
print(f"  words found    {stats['found']:,} / {stats['vocab_size']:,}  ({stats['coverage']:.2%})")
print(f"  words missing  {stats['missing']:,}")
print(f"  random init    mean {stats['init_mean']:.4f}, std {stats['init_std']:.4f}")
print(f"  PAD row is all zeros: {bool(np.all(matrix[0] == 0))}")
print("""
Missing words get a random vector with the same mean and standard deviation
as the real ones. This copies get_emb_matrix() in the SOLD codebase exactly.""")


# ==========================================================================
rule("4. COVERAGE ON EACH SPLIT   <-- THE NUMBERS FOR THE PAPER")
print("""'type' = distinct words.  'token' = word occurrences.
in_vocab   = the word has its own row in our vocabulary
has_vector = that row holds a REAL fastText vector, not a random one
""")
print(f"{'split':<12} {'types':>8} {'tokens':>9} {'type vec':>10} {'token vec':>11}")
print("-" * 54)
rows = {}
for name, df in [("train-part", train_part), ("validation", val), ("test", test)]:
    r = coverage_report(df["token_list"], vocab, vectors)
    rows[name] = r
    print(f"{name:<12} {r['n_types']:>8,} {r['n_tokens']:>9,} "
          f"{r['type_has_vector']:>9.2%} {r['token_has_vector']:>10.2%}")

t = rows["test"]
print(f"""
READ THIS CAREFULLY.

On test, {t['token_in_vocab']:.1%} of word occurrences are in our vocabulary at all,
and {t['token_has_vector']:.1%} have a real fastText vector.

So roughly {1 - t['token_in_vocab']:.1%} of the words the model sees at test time are
words it has never seen in training. It has no idea what they mean.
Almost half of all DISTINCT test words are new.

This is the whole argument for the subword component in Phase 2. If the
model could look INSIDE a word - at its pieces - it could still make sense
of a word form it has never met, because it would recognise the root.

Write these numbers into the README and into the paper.""")


# ==========================================================================
rule("5. ARE THE OFFENSIVE WORDS COVERED?")
print("Overall coverage is not enough. What matters is whether the words we")
print("actually need to detect are covered.\n")

from collections import Counter
tot, pos = Counter(), Counter()
for toks, rats in zip(train_part["token_list"], train_part["rationales"]):
    for tk, r in zip(toks, rats):
        tot[tk] += 1
        if r == 1:
            pos[tk] += 1

off_words = [w for w in pos if tot[w] >= 3 and pos[w] / tot[w] > 0.5]
covered = [w for w in off_words if w in vectors]
print(f"  reliably offensive words (seen 3+, offensive >50%): {len(off_words):,}")
print(f"  of those, having a real fastText vector: {len(covered):,} ({len(covered)/max(len(off_words),1):.1%})")

all_cov = stats["coverage"]
print(f"  overall vocabulary coverage for comparison: {all_cov:.1%}")
print("""
If offensive words are covered LESS than average, that is a real finding
and belongs in the paper: swear words and slang are under-represented in
the Common Crawl text fastText was trained on. It would also strengthen
the subword argument further.""")


# ==========================================================================
rule("6. SANITY CHECK - DO THE VECTORS MEAN ANYTHING?")
print("Nearest neighbours by cosine similarity. If the vectors are loaded")
print("correctly, neighbours should look related. If they look random, the")
print("file is wrong or misaligned.\n")

words_with_vecs = [w for w in vocab if w in vectors]
freq = Counter(t for toks in train_part["token_list"] for t in toks)
probes = [w for w, _ in freq.most_common(400) if w in vectors and len(w) > 2][:5]
probes += [w for w in off_words if w in vectors][:3]

V = np.stack([vectors[w] for w in words_with_vecs])
V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
index = {w: i for i, w in enumerate(words_with_vecs)}

for p in probes:
    sims = V @ V[index[p]]
    top = np.argsort(-sims)[1:6]
    print(f"  {p:<18} -> {', '.join(words_with_vecs[i] for i in top)}")


# ==========================================================================
rule("7. SAVE FOR STEP 4")
os.makedirs("artifacts", exist_ok=True)
np.save("artifacts/embedding_matrix.npy", matrix)
with open("artifacts/vocab.txt", "w", encoding="utf-8") as fh:
    for w, i in sorted(vocab.items(), key=lambda kv: kv[1]):
        fh.write(f"{w}\t{i}\n")
print("  artifacts/embedding_matrix.npy")
print("  artifacts/vocab.txt")
print("\nAdd artifacts/ and embeddings/ to .gitignore - they are large and")
print("regenerable. The CODE is what gets committed, not the outputs.")


# ==========================================================================
rule("8. SUMMARY FOR THE README")
print(f"""
embedding file        {os.path.basename(VEC_PATH)}
dimension             {dim}
vocabulary            {len(vocab):,} words, min_freq={MIN_FREQ}, from train-part only
matrix                {matrix.shape[0]:,} x {matrix.shape[1]}
vector coverage       {stats['coverage']:.1%} of vocabulary rows are real vectors
OOV handling          random N(mean, std) of real vectors, copied from SOLD
test token coverage   {t['token_has_vector']:.1%} have a real vector
test unseen tokens    {1 - t['token_in_vocab']:.1%} never appeared in training
test unseen types     {1 - t['type_in_vocab']:.1%} of distinct words are new
offensive word cov    {len(covered)/max(len(off_words),1):.1%}
""")
