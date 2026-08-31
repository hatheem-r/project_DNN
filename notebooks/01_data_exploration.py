"""
Phase 1 / Step 1 - SOLD data exploration and verification.

Run:  python notebooks/01_data_exploration.py
(or paste cell by cell into a Colab notebook)

Prints a report answering every question you must settle before writing
any model code. Read the OUTPUT, don't just run it.
"""

import sys
import os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from data import load_sold

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ==========================================================================
# 1. LOAD
# ==========================================================================
rule("1. LOAD")

train = load_sold("train")
test = load_sold("test")

print(f"train rows: {len(train):,}   (expected 7,500)")
print(f"test  rows: {len(test):,}   (expected 2,500)")
print(f"total     : {len(train) + len(test):,}   (expected 10,000)")
print("\ncolumns:", list(train.columns))
print("\ndtypes:")
print(train.dtypes)


# ==========================================================================
# 2. RAW COLUMN TYPES  - what did rationales/tokens actually arrive as?
# ==========================================================================
rule("2. RAW COLUMN TYPES")

print("rationale column found as:", train.attrs.get("source_rationale_column"))
print("(GitHub README says 'rationals'; the real dataset uses 'rationales')\n")
raw = train.iloc[0]
print("tokens          type:", type(raw["tokens"]).__name__)
print("rationales_raw  type:", type(raw["rationales_raw"]).__name__, "(after parsing)")
print("label           type:", type(raw["label"]).__name__)
print("\nfirst row, raw text field:")
print(repr(raw["text"])[:300])


# ==========================================================================
# 3. SENTENCE LABELS  - what are the actual label values?
# ==========================================================================
rule("3. SENTENCE-LEVEL LABELS")

for name, df in [("train", train), ("test", test)]:
    counts = df["label"].value_counts()
    print(f"\n{name}:")
    for k, v in counts.items():
        print(f"  {k!r:<20} {v:>6,}  ({v / len(df):.1%})")

print("\nWhole dataset should be 4,191 offensive / 5,809 not offensive.")
combined = pd.concat([train["label"], test["label"]]).value_counts()
print(dict(combined))


# ==========================================================================
# 4. ALIGNMENT  - does every token have exactly one rationale?
# ==========================================================================
rule("4. EMPTY RATIONALES AND LENGTH ALIGNMENT   <-- CRITICAL")

print("In SOLD, NOT-offensive tweets store [] rather than a vector of zeros.")
print("load_sold() expands [] to [0]*n_tokens. Counts below are BEFORE expansion.\n")

for name, df in [("train", train), ("test", test)]:
    print(f"{name}:")
    print(f"  rows with empty rationale []      : {df['raw_empty'].sum():,} "
          f"({df['raw_empty'].mean():.1%})")
    print(f"  rows with non-empty but WRONG len : {df['length_mismatch'].sum():,}   <-- real problem if > 0")
    bad = df[df["length_mismatch"]]
    for _, r in bad.head(5).iterrows():
        print(f"     post_id={r['post_id']} tokens={r['n_tokens']} rationales={len(r['rationales_raw'])}")
    ok = all(len(r) == n for r, n in zip(df["rationales"], df["n_tokens"]))
    print(f"  after expansion, all lengths match: {ok}")
    print()

print("Cross-tab of empty-rationale against sentence label:")
for name, df in [("train", train), ("test", test)]:
    print(f"\n  {name}:")
    print(pd.crosstab(df["label"], df["raw_empty"]).rename(
        columns={True: "empty []", False: "has vector"}).to_string())

print("""
EXPECTED: every NOT row is empty. Any OFF row that is ALSO empty is an
annotation anomaly - the tweet was judged offensive but no tokens were
highlighted. Count them, note the number in the README, and decide whether
to keep them (they become all-negative training rows) or drop them.
DECISION REQUIRED - write it down either way.
""")


# ==========================================================================
# 5. RATIONALE VALUES  - are they strictly 0/1?
# ==========================================================================
rule("5. RATIONALE VALUE SET")

vals = Counter()
for r in pd.concat([train["rationales_raw"], test["rationales_raw"]]):
    vals.update(r)
print("distinct values found:", dict(vals))
print("(expect only 0 and 1)")


# ==========================================================================
# 6. THE SCHEME CHECK  - do non-offensive tweets have all-zero rationales?
# ==========================================================================
rule("6. OFFENSIVE TWEETS SHOULD HAVE AT LEAST ONE OFFENSIVE TOKEN")

for name, df in [("train", train), ("test", test)]:
    labels = sorted(df["label"].unique())
    print(f"\n{name}:")
    for lab in labels:
        sub = df[df["label"] == lab]
        with_pos = (sub["n_offensive_tokens"] > 0).sum()
        print(f"  label {lab!r:<20} rows={len(sub):>5,}  "
              f"rows with >=1 offensive token: {with_pos:>5,} ({with_pos/len(sub):.1%})")

print("\nNOT rows must be 0.0% by construction (they store []).")
print("OFF rows below 100% are the anomaly from section 4. A small number is")
print("normal annotation noise; a large number means something is wrong.")


# ==========================================================================
# 7. CLASS IMBALANCE  - the number that drives the whole project
# ==========================================================================
rule("7. TOKEN-LEVEL CLASS IMBALANCE")

for name, df in [("train", train), ("test", test)]:
    total_tokens = df["n_tokens"].sum()
    pos_tokens = df["n_offensive_tokens"].sum()
    print(f"{name}: {pos_tokens:,} offensive / {total_tokens:,} tokens "
          f"= {pos_tokens / total_tokens:.2%} positive")

print("\nPaper's 'All OFF' baseline has precision 0.03, so expect ~3%.")
print("If you see this, your rationale parsing is correct.")

# imbalance inside offensive tweets only
off_only = train[train["n_offensive_tokens"] > 0]
if len(off_only):
    print(f"\nWithin tweets that have >=1 offensive token (train, n={len(off_only):,}):")
    print(f"  {off_only['n_offensive_tokens'].sum() / off_only['n_tokens'].sum():.2%} of tokens are offensive")
    print(f"  median offensive tokens per such tweet: "
          f"{off_only['n_offensive_tokens'].median():.0f}")


# ==========================================================================
# 8. LENGTH DISTRIBUTION  - sets max_len for the model
# ==========================================================================
rule("8. TOKEN COUNT DISTRIBUTION")

for name, df in [("train", train), ("test", test)]:
    n = df["n_tokens"]
    print(f"\n{name}: min={n.min()}  median={n.median():.0f}  mean={n.mean():.1f}  max={n.max()}")
    for p in [50, 75, 90, 95, 99]:
        print(f"  p{p}: {np.percentile(n, p):.0f}")

cover = {L: (train["n_tokens"] <= L).mean() for L in (16, 24, 32, 48, 64)}
print("\ncoverage if max_len set to:")
for L, c in cover.items():
    print(f"  {L:>3}: {c:.2%} of train tweets fit without truncation")
print("\nPick the smallest max_len covering >=99% and record it in configs/baseline.yaml.")


# ==========================================================================
# 9. VOCABULARY  - first look, feeds the OOV analysis in Step 3
# ==========================================================================
rule("9. VOCABULARY")

train_vocab = Counter(t for toks in train["token_list"] for t in toks)
test_vocab = Counter(t for toks in test["token_list"] for t in toks)
unseen = set(test_vocab) - set(train_vocab)
unseen_tok = sum(test_vocab[t] for t in unseen)

print(f"train vocab size: {len(train_vocab):,}")
print(f"test  vocab size: {len(test_vocab):,}")
print(f"test types unseen in train: {len(unseen):,} ({len(unseen)/len(test_vocab):.1%})")
print(f"test tokens unseen in train: {unseen_tok:,} "
      f"({unseen_tok / sum(test_vocab.values()):.1%} of all test tokens)")
print("\nThis last percentage is the empirical case for subword modelling.")
print("Record it - it goes in the paper.")

print("\nmost common tokens:")
for tok, c in train_vocab.most_common(15):
    print(f"  {c:>6,}  {tok}")


# ==========================================================================
# 10. MOST-OFFENSIVE TOKENS  - sanity check that labels mean something
# ==========================================================================
rule("10. TOKENS MOST OFTEN LABELLED OFFENSIVE")

tok_total, tok_pos = Counter(), Counter()
for toks, rats in zip(train["token_list"], train["rationales"]):
    for t, r in zip(toks, rats):
        tok_total[t] += 1
        if r == 1:
            tok_pos[t] += 1

rows = [(t, tok_pos[t], tok_total[t], tok_pos[t] / tok_total[t])
        for t in tok_pos if tok_total[t] >= 20]
rows.sort(key=lambda x: (-x[3], -x[1]))

print(f"{'token':<20} {'off':>6} {'total':>7} {'rate':>7}")
for t, p, n, rate in rows[:20]:
    print(f"{t:<20} {p:>6} {n:>7} {rate:>6.0%}")

print("\nNOTE the rates. The paper shows keyword offensiveness mostly 30-50%,")
print("i.e. context decides. That is why a lexicon lookup cannot solve this")
print("and why you need a sequence model.")


# ==========================================================================
# 11. EXAMPLES  - read these as a team
# ==========================================================================
rule("11. TWENTY EXAMPLES (token / label)")

sample = pd.concat([
    train[train["n_offensive_tokens"] > 0].head(12),
    train[train["n_offensive_tokens"] == 0].head(8),
])

for i, (_, r) in enumerate(sample.iterrows(), 1):
    print(f"\n--- {i}. post_id={r['post_id']}  label={r['label']}  "
          f"({r['n_offensive_tokens']}/{r['n_tokens']} offensive)")
    for t, lab in zip(r["token_list"], r["rationales"]):
        mark = "  <== OFFENSIVE" if lab == 1 else ""
        print(f"    {lab}  {t}{mark}")


# ==========================================================================
# 12. SUMMARY FOR THE README
# ==========================================================================
rule("12. COPY THESE NUMBERS INTO YOUR README")

tt_train = train["n_tokens"].sum()
tp_train = train["n_offensive_tokens"].sum()
tt_test = test["n_tokens"].sum()
tp_test = test["n_offensive_tokens"].sum()

print(f"""
train rows                 {len(train):,}
test rows                  {len(test):,}
empty rationale rows (tr)  {train['raw_empty'].sum():,}
length-mismatch rows (tr)  {train['length_mismatch'].sum()}
length-mismatch rows (te)  {test['length_mismatch'].sum()}
OFF rows w/ no off token   {((train['label'] == 'OFF') & (train['n_offensive_tokens'] == 0)).sum():,}
train tokens               {tt_train:,}
train offensive tokens     {tp_train:,}  ({tp_train/tt_train:.2%})
test tokens                {tt_test:,}
test offensive tokens      {tp_test:,}  ({tp_test/tt_test:.2%})
train vocab                {len(train_vocab):,}
test tokens unseen in train {unseen_tok/sum(test_vocab.values()):.1%}
median tweet length        {train['n_tokens'].median():.0f} tokens
p99 tweet length           {np.percentile(train['n_tokens'], 99):.0f} tokens
""")