"""
Phase 2 / Piece 1 / Step 6 - train and evaluate the subword tokenizer.

No neural network here. We only decide HOW to split Sinhala words, and measure
whether splitting is likely to help, before building anything on top of it.

Run:  python notebooks/06_subword_tokenizer.py > results/step6_subword.txt

Writes:  artifacts/sp_<type>_<size>.model
         artifacts/subword_summary.json

--------------------------------------------------------------------------
WHY WE NEED THIS (measured in Step 3)
--------------------------------------------------------------------------
  48.7% of DISTINCT test words never appeared in training
  14.3% of test word OCCURRENCES never appeared in training
  fastText's Sinhala neighbours are morphological variants, not synonyms:
      සක්කිලි -> සක්කිලිය, සක්කිලියා, සක්කිලියෙක්, සක්කිලියන්ගේ

Our word-level model treats those as unrelated ID numbers. Sinhala is
agglutinative, so one root generates many surface forms. Subwords let the model
see the shared root.

--------------------------------------------------------------------------
TWO MEASUREMENTS WE TRIED AND DISCARDED - recorded so nobody repeats them
--------------------------------------------------------------------------
1. "What fraction of an unseen word's pieces were seen in training?"
   SATURATED at 99.7-99.9% for every setting, so it discriminated nothing.
   Cause: pieces bottom out at single characters, and Sinhala's character
   inventory is small, so any piece has been seen somewhere in 141,646 words.
   It was measuring "have we seen these characters", which is trivially yes.

2. "Do morphological variants share a root piece?" tested on the most COMMON
   training words. Those are exactly the words SentencePiece keeps whole,
   because a frequent string earns its own vocabulary slot. And keeping them
   whole is correct - frequent words already have real fastText vectors.

Both are replaced below by measurements taken on the UNSEEN test words, which
are the only words the subword channel actually needs to rescue.
"""
import sys, os, json, unicodedata, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from collections import Counter

from data import load_sold, train_val_split
from subword import (
    train_sentencepiece, load_sentencepiece, word_to_pieces,
    fragmentation, shared_root_check,
)

VOCAB_SIZES = [1000, 2000, 4000, 8000, 16000, 24000, 32000]
MODEL_TYPES = ["unigram", "bpe"]
OUT_DIR = "artifacts"
MIN_ROOT_LEN = 3        # a piece must be this long to count as a root
MAX_FRAGMENTATION = 2.5  # above this we are shredding, not decomposing
SP_MARK = "\u2581"       # SentencePiece word-start marker


def rule(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


def strip_mark(p):
    return p.lstrip(SP_MARK)


# --------------------------------------------------------------------------
# Sinhala grapheme handling
# --------------------------------------------------------------------------
# Sinhala writes a syllable as a base consonant plus COMBINING marks: vowel
# signs, and the hal kirima that kills the inherent vowel. Those marks are
# separate Unicode codepoints but are NOT separate letters.
#
#   සෙ  is ONE written letter = ස (base) + ෙ (combining vowel sign)
#
# If a tokenizer piece STARTS with a combining mark, the split cut a written
# letter in half. That piece cannot correspond to any morpheme.
#
# This also means MIN_ROOT_LEN counted in codepoints is looser than it looks:
# three codepoints of Sinhala can be a single visible letter.

def is_combining(ch: str) -> bool:
    return unicodedata.category(ch) in ("Mn", "Mc")


def breaks_grapheme(piece: str) -> bool:
    """True if this piece begins mid-letter."""
    p = strip_mark(piece)
    return bool(p) and is_combining(p[0])


def grapheme_len(s_: str) -> int:
    """Count VISIBLE letters, not codepoints: a base plus its marks is one."""
    n = 0
    for ch in strip_mark(s_):
        if not is_combining(ch):
            n += 1
    return n


# ==========================================================================
rule("0. DATA")
train_full = load_sold("train")
test = load_sold("test")
train_part, val = train_val_split(train_full)
print(f"train-part {len(train_part):,}  validation {len(val):,}  test {len(test):,}")
print("\nThe tokenizer is trained on TRAIN-PART TEXT ONLY. Training it on")
print("validation or test would leak information about words the model is not")
print("supposed to know exist.")

train_texts = [" ".join(t) for t in train_part["token_list"]]
train_word_counts = Counter(w for toks in train_part["token_list"] for w in toks)
train_words = set(train_word_counts)
test_types = set(w for toks in test["token_list"] for w in toks)
unseen = sorted(test_types - train_words)

test_tok_total = sum(len(t) for t in test["token_list"])
unseen_tok = sum(1 for toks in test["token_list"] for w in toks if w not in train_words)

print(f"\ntraining corpus   {len(train_texts):,} tweets, "
      f"{sum(len(t) for t in train_part['token_list']):,} words")
print(f"train types       {len(train_words):,}")
print(f"test types        {len(test_types):,}")
print(f"UNSEEN test types {len(unseen):,}  ({len(unseen)/len(test_types):.1%} of test types)")
print(f"unseen occurrences {unseen_tok:,}  ({unseen_tok/test_tok_total:.1%} of test tokens)")
print("\nThose unseen words are what this whole component exists to rescue.")

if not unseen:
    print("\nNO UNSEEN TEST WORDS. Sections 3-5 measure nothing and would divide")
    print("by zero. This cannot happen on real SOLD data (7,676 unseen types).")
    print("If you see this, the splits are wrong - check src/data.py.")
    sys.exit(1)


# ==========================================================================
rule("1. TRAIN TOKENIZERS")
print("SentencePiece learns the pieces from raw text - no rules, no dictionary.")
print("  unigram : probabilistic, keeps pieces that best explain the corpus")
print("  bpe     : greedy, repeatedly merges the most frequent pair\n")

os.makedirs(OUT_DIR, exist_ok=True)
models = {}
actual_size = {}   # requested size may exceed what the corpus can support
for mt in MODEL_TYPES:
    for vs in VOCAB_SIZES:
        prefix = f"{OUT_DIR}/sp_{mt}_{vs}"
        if not os.path.exists(prefix + ".model"):
            print(f"  training {mt} vocab={vs} ...")
            train_sentencepiece(train_texts, vs, prefix, model_type=mt)
        sp = load_sentencepiece(prefix + ".model")
        models[(mt, vs)] = sp
        got = sp.get_piece_size()
        actual_size[(mt, vs)] = got
        note = "" if got >= vs else f"   (corpus supports only {got:,})"
        print(f"  {mt:<8} requested={vs:<6} actual={got:,}{note}")

print("""
If 'actual' is capped below 'requested', the corpus does not contain enough
distinct subword candidates for that size. Two requested sizes that cap to the
same actual size are THE SAME MODEL and are de-duplicated in section 7.""")


# ==========================================================================
rule("2. FRAGMENTATION - how badly are ALL words shredded?")
print("""pieces_per_word is the cost side of the trade-off.
  Near 1.0  -> barely splitting; no gain over word level
  Above 2.5 -> shredding; the BiLSTM must reassemble meaning from fragments,
               and piece sequences get long
""")
print(f"{'type':<9} {'vocab':>7} {'pcs/word':>9} {'1 piece':>9} {'2 pieces':>9} {'3+':>8}")
print("-" * 56)
frag = {}
for mt in MODEL_TYPES:
    for vs in VOCAB_SIZES:
        f = fragmentation(models[(mt, vs)], train_part["token_list"])
        frag[(mt, vs)] = f
        print(f"{mt:<9} {vs:>7,} {f['pieces_per_word']:>9.2f} "
              f"{f['pct_1_piece']:>8.1%} {f['pct_2_pieces']:>8.1%} {f['pct_3plus']:>7.1%}")


# ==========================================================================
rule("3. ARE UNSEEN WORDS ACTUALLY SPLIT?")
print("""If an unseen word comes out as ONE piece, the subword channel hands the
model a single unfamiliar symbol - no better than the word channel. Unseen words
must be split into parts that occur elsewhere for us to gain anything.
""")
print(f"{'type':<9} {'vocab':>7} {'pcs/unseen wd':>14} {'kept whole':>12} {'2+ pieces':>11}")
print("-" * 58)
split_stats = {}
for mt in MODEL_TYPES:
    for vs in VOCAB_SIZES:
        sp = models[(mt, vs)]
        n_pieces = whole = 0
        for w in unseen:
            k = max(len(word_to_pieces(sp, w)), 1)
            n_pieces += k
            if k == 1:
                whole += 1
        split_stats[(mt, vs)] = {
            "pcs_per_unseen": n_pieces / len(unseen),
            "kept_whole": whole / len(unseen),
        }
        print(f"{mt:<9} {vs:>7,} {n_pieces/len(unseen):>14.2f} "
              f"{whole/len(unseen):>11.1%} {1-whole/len(unseen):>10.1%}")


# ==========================================================================
rule("4. DOES AN UNSEEN WORD'S ROOT APPEAR IN TRAINING WORDS?   <-- THE METRIC")
print(f"""For each unseen test word, take its pieces of at least {MIN_ROOT_LEN} characters -
the stem, not a grammatical ending. Then ask: does that stem also occur as a
piece inside some word we DID see in training?

If yes, the model can transfer what it learned about the training word to this
brand-new form. That is the entire mechanism of this component.

Pieces shorter than {MIN_ROOT_LEN} characters are ignored, because single characters
connect everything to everything and mean nothing. That is exactly what made our
first attempt at this measurement saturate at 99.9%.
""")
print(f"{'type':<9} {'vocab':>7} {'root shared':>12} {'no usable root':>15}")
print("-" * 47)
root_stats = {}
for mt in MODEL_TYPES:
    for vs in VOCAB_SIZES:
        sp = models[(mt, vs)]
        train_pieces = set()
        for w in train_words:
            for p in word_to_pieces(sp, w):
                train_pieces.add(strip_mark(p))

        shared = no_root = 0
        for w in unseen:
            roots = [strip_mark(p) for p in word_to_pieces(sp, w)]
            roots = [p for p in roots if len(p) >= MIN_ROOT_LEN]
            if not roots:
                no_root += 1
                continue
            if any(r in train_pieces for r in roots):
                shared += 1
        root_stats[(mt, vs)] = shared / len(unseen)
        print(f"{mt:<9} {vs:>7,} {shared/len(unseen):>11.1%} {no_root/len(unseen):>14.1%}")

print(f"""
'root shared'    = the unseen word contains a >={MIN_ROOT_LEN}-char piece that also occurs
                   inside training words. MAXIMISE THIS.
'no usable root' = the word shattered into pieces all shorter than {MIN_ROOT_LEN} chars.
                   That is shredding, not decomposition.""")


# ==========================================================================
rule("4b. GRAPHEME INTEGRITY - are we cutting Sinhala letters in half?")
print("""Sinhala writes a syllable as a base consonant plus combining marks - vowel
signs, and the hal kirima. Those are separate Unicode codepoints but NOT
separate letters:

    සෙ  is ONE written letter =  ස (base) + ෙ (combining vowel sign)

If a piece STARTS with a combining mark, the split cut a written letter in half.
Such a piece cannot correspond to any morpheme. This is a Sinhala-specific
failure mode that a pieces-per-word count does not reveal.
""")
print(f"{'type':<9} {'vocab':>7} {'broken pieces':>14} {'unseen wds hit':>15} {'root<2 letters':>15}")
print("-" * 64)
graph_stats = {}
for mt in MODEL_TYPES:
    for vs in VOCAB_SIZES:
        sp = models[(mt, vs)]
        n_pieces = broken = 0
        words_hit = short_root = 0
        for w in unseen:
            pcs = word_to_pieces(sp, w)
            n_pieces += len(pcs)
            b = sum(1 for p in pcs if breaks_grapheme(p))
            broken += b
            if b:
                words_hit += 1
            if all(grapheme_len(p) < 2 for p in pcs):
                short_root += 1
        graph_stats[(mt, vs)] = {
            "broken_piece_rate": broken / max(n_pieces, 1),
            "unseen_words_affected": words_hit / len(unseen),
            "all_pieces_under_2_letters": short_root / len(unseen),
        }
        print(f"{mt:<9} {vs:>7,} {broken/max(n_pieces,1):>13.1%} "
              f"{words_hit/len(unseen):>14.1%} {short_root/len(unseen):>14.1%}")

print("""
'broken pieces'  = pieces that begin with a combining mark, i.e. mid-letter cuts
'unseen wds hit' = unseen words containing at least one such piece
'root<2 letters' = every piece is under 2 VISIBLE letters. Pure shredding.

MINIMISE all three. A setting with high root-sharing but many broken graphemes
is sharing meaningless fragments, not roots.""")


# ==========================================================================
rule("5. WHAT UNSEEN WORDS LOOK LIKE WHEN SPLIT")
print("Real unseen Sinhala words. Judge whether these look like stem + ending.\n")
test_counts = Counter(w for toks in test["token_list"] for w in toks)
freq_unseen = sorted(((w, test_counts[w]) for w in unseen), key=lambda kv: -kv[1])[:8]

for mt in MODEL_TYPES:
    for vs in [1000, 2000, 8000]:
        sp = models[(mt, vs)]
        print(f"--- {mt}, vocab {vs:,} ---")
        for w, c in freq_unseen:
            pcs = [strip_mark(p) for p in word_to_pieces(sp, w)]
            print(f"  {w:<22} x{c:<4} -> {' | '.join(pcs)}")
        print()


# ==========================================================================
rule("6. OFFENSIVE WORDS - are they split sensibly?")
tot, pos = Counter(), Counter()
for toks, rats in zip(train_part["token_list"], train_part["rationales"]):
    for tk, r in zip(toks, rats):
        tot[tk] += 1
        if r == 1:
            pos[tk] += 1
off_words = sorted([w for w in pos if tot[w] >= 5 and pos[w] / tot[w] > 0.7],
                   key=lambda w: -pos[w])[:12]

for vs in [1000, 4000]:
    sp = models[("unigram", vs)]
    print(f"--- unigram, vocab {vs:,} ---")
    for w, pcs in shared_root_check(sp, off_words):
        pcs = [strip_mark(p) for p in pcs]
        flag = "  <-- kept whole" if len(pcs) == 1 else ""
        print(f"  {w:<20} -> {' | '.join(pcs)}{flag}")
    print()

print("""Frequent offensive words kept whole is FINE - they already have real
fastText vectors and plenty of training examples. What matters is that their
STEM exists as a piece, so unseen variants of the same slur can reach it.""")


# ==========================================================================
rule("7. THE TRADE-OFF AND THE SHORTLIST")
print(f"{'type':<9} {'requested':>10} {'actual':>8} {'root shared':>12} {'broken':>8} {'pcs/all wd':>11} {'pcs/unseen':>11} {'ok?':>5}")
print("-" * 82)

# De-duplicate: two requested sizes that capped to the same actual size are the
# same tokenizer. Keep the smallest request that produced each actual size.
seen_actual = {}
for mt in MODEL_TYPES:
    for vs in VOCAB_SIZES:
        key = (mt, actual_size[(mt, vs)])
        if key not in seen_actual:
            seen_actual[key] = vs

rows = []
for mt in MODEL_TYPES:
    for vs in VOCAB_SIZES:
        dup = seen_actual[(mt, actual_size[(mt, vs)])] != vs
        rows.append((root_stats[(mt, vs)], mt, vs, actual_size[(mt, vs)],
                     frag[(mt, vs)]["pieces_per_word"],
                     split_stats[(mt, vs)]["pcs_per_unseen"],
                     graph_stats[(mt, vs)]["broken_piece_rate"], dup))

shortlist = []
for r, mt, vs, act, fpw, fpu, brk, dup in sorted(rows, reverse=True):
    if dup:
        continue
    ok = fpw <= MAX_FRAGMENTATION
    if ok:
        shortlist.append((mt, vs, act, r, fpw, brk))
    print(f"{mt:<9} {vs:>10,} {act:>8,} {r:>11.1%} {brk:>7.1%} {fpw:>11.2f} "
          f"{fpu:>11.2f} {'yes' if ok else 'NO':>5}")

n_dup = sum(1 for row in rows if row[-1])
if n_dup:
    print(f"\n({n_dup} duplicate rows hidden - requested sizes that capped to an")
    print(" actual size already listed above.)")

print(f"""
'ok?' = pieces per word under {MAX_FRAGMENTATION}, so we decompose rather than shred.

THERE ARE TWO COMPETING CRITERIA, NOT ONE.

  'root shared' rewards LARGE vocabularies: longer pieces are more likely to
     match something inside a training word.
  'broken'      often favours a DIFFERENT algorithm: a tokenizer can score well
     on root-sharing while cutting Sinhala letters in half, and those pieces
     cannot correspond to any morpheme.

Do not rank on one column. Compare the algorithms at similar vocabulary sizes.

ALSO CHECK FOR A DEGENERATE WIN. If the best setting keeps 85% of words whole,
the subword channel is nearly a copy of the word channel for words we already
handle, and every gain must come from the unseen words alone. That may still be
worth it, but it is a different mechanism from real decomposition - and it
bounds how much this component can possibly buy us.

DESIGN THE SHORTLIST TO SPAN THE RANGE, NOT TO CLUSTER AT ONE END.
Three near-identical large settings teach us almost nothing. Pick settings that
answer different questions, then let VALIDATION F1 decide.""")

if shortlist:
    print("\nAUTO-RANKED (by root sharing only - read the caveats above):")
    for i, (mt, vs, act, r, fpw, brk) in enumerate(shortlist[:3], 1):
        print(f"  {i}. {mt}, {act:,} pieces  "
              f"(root shared {r:.1%}, {fpw:.2f} pcs/word, {brk:.1%} broken)")

    max_actual = max(actual_size.values())
    top_act = shortlist[0][2]
    capped = any(actual_size[(mt, vs)] < vs for mt in MODEL_TYPES for vs in VOCAB_SIZES)
    if top_act >= max_actual and not capped:
        print(f"""
  WARNING: the best setting is at the largest ACTUAL size ({max_actual:,}) and no
  tokenizer hit a corpus cap. The curve may not have turned over - add larger
  sizes and re-run.""")
    elif capped:
        print(f"""
  The largest unigram sizes capped at the corpus limit, so that end of the
  curve is fully explored. No need to sweep higher for unigram.""")

    best_frag = shortlist[0][4]
    if best_frag < 1.3:
        print(f"""
  NOTE: the top setting averages only {best_frag:.2f} pieces per word, meaning most
  words are kept whole. Include a MORE FRAGMENTING setting in the experiments
  as a contrast, or you will not learn whether decomposition helps at all.""")
else:
    print("\nNo setting stayed under the fragmentation cap. Try larger vocabularies.")

summary = {
    f"{mt}_{vs}": {
        "requested_vocab": vs,
        "actual_pieces": actual_size[(mt, vs)],
        "root_shared": root_stats[(mt, vs)],
        "pieces_per_word": frag[(mt, vs)]["pieces_per_word"],
        "pieces_per_unseen_word": split_stats[(mt, vs)]["pcs_per_unseen"],
        "unseen_kept_whole": split_stats[(mt, vs)]["kept_whole"],
        "pct_1_piece": frag[(mt, vs)]["pct_1_piece"],
        **graph_stats[(mt, vs)],
    }
    for mt in MODEL_TYPES for vs in VOCAB_SIZES
}
summary["_meta"] = {
    "unseen_types": len(unseen),
    "unseen_token_rate": unseen_tok / test_tok_total,
    "min_root_len_codepoints": MIN_ROOT_LEN,
    "max_fragmentation": MAX_FRAGMENTATION,
    "note": "MIN_ROOT_LEN counts codepoints, not visible letters. Sinhala "
            "graphemes take 2-3 codepoints, so 3 codepoints is about 2 letters.",
}
with open(f"{OUT_DIR}/subword_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"\nsaved {OUT_DIR}/subword_summary.json")