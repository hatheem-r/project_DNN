"""
Phase 1 / Step 2 - verify the evaluation metric, and establish the baselines
that our neural model must beat.

Run:  python notebooks/02_metric_check.py > results/step2_report.txt

--------------------------------------------------------------------------
THE RULE THIS FILE OBEYS
--------------------------------------------------------------------------
The test split is a locked exam. You may run something on it, but you may
NOT use its scores to CHOOSE anything. Choosing is done on validation.

  Baselines A, B, C below have no settings to choose. Running them straight
  on test is fine - there is no decision being made.

  Baseline D (the word list) HAS settings: how strict the list should be,
  and how many times a word must appear before we trust it. Those are
  chosen on VALIDATION, then the single winner is run once on test.

An earlier version of this file tried four word lists on test and reported
the best (0.631). That was test-set tuning and the number was void. This
version fixes it.
"""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from collections import Counter

from data import load_sold, train_val_split, VAL_SEED, VAL_FRACTION
from metrics import token_level_scores, format_scores

# hyperparameter grid for the word-list baseline, searched on validation only
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MIN_COUNTS = [1, 2, 3, 5, 10]


def rule(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


# ==========================================================================
# 0. SPLITS
# ==========================================================================
rule("0. SPLITS")

train_full = load_sold("train")
test = load_sold("test")
train_part, val = train_val_split(train_full)

print(f"official train : {len(train_full):,}")
print(f"  train-part   : {len(train_part):,}   used to build things")
print(f"  validation   : {len(val):,}   used to CHOOSE things")
print(f"official test  : {len(test):,}   locked, scored once at the end")
print(f"\nsplit seed = {VAL_SEED}, validation fraction = {VAL_FRACTION}")
print("Defined in src/data.py so every later model uses the identical split.")

print("\npositive token rate per split:")
for name, df in [("train-part", train_part), ("validation", val), ("test", test)]:
    r = df["n_offensive_tokens"].sum() / df["n_tokens"].sum()
    print(f"  {name:<11} {r:.2%}")
print("train-part and validation should be close, or the split was unlucky.")

gold_test = list(test["rationales"])
gold_val = list(val["rationales"])


# ==========================================================================
# A. FLAG EVERY TOKEN
# ==========================================================================
rule("A. FLAG EVERY TOKEN  ('All OFF' in the paper)   [no setting to choose]")

s_all = token_level_scores(gold_test, [[1] * n for n in test["n_tokens"]])
print(format_scores(s_all))
print(f"""
CHECK: precision must equal the test positive rate, because if you flag
everything then precision is just the base rate. Recall must be exactly
1.0000. The paper reports 0.03 precision for this same baseline, so our
metric agrees with theirs.
  precision = {s_all['offensive_precision']:.4f}   recall = {s_all['offensive_recall']:.4f}""")


# ==========================================================================
# B. FLAG NO TOKEN
# ==========================================================================
rule("B. FLAG NO TOKEN   [no setting to choose]")

s_none = token_level_scores(gold_test, [[0] * n for n in test["n_tokens"]])
print(format_scores(s_none))
print(f"""
CHECK: offensive F1 must be exactly 0.0000, while accuracy would be about
{s_none['not_offensive_precision']:.0%}. That is why accuracy is banned here.
Note macro F1 is still {s_none['macro_f1']:.4f} for a model that does nothing.
That is why we report the OFFENSIVE-CLASS F1, not a macro average.""")


# ==========================================================================
# C. RANDOM AT THE TRAINING POSITIVE RATE
# ==========================================================================
rule("C. RANDOM GUESSING   [rate taken from train, not test]")

rate = train_full["n_offensive_tokens"].sum() / train_full["n_tokens"].sum()
random.seed(42)
pred = [[1 if random.random() < rate else 0 for _ in range(n)] for n in test["n_tokens"]]
s_rand = token_level_scores(gold_test, pred)
print(f"  rate used {rate:.4f}   P={s_rand['offensive_precision']:.4f} "
      f"R={s_rand['offensive_recall']:.4f} F1={s_rand['offensive_f1']:.4f}")
print("\nCHECK: P and R both land near the positive rate. Any real model must")
print("be far above this.")


# ==========================================================================
# D. WORD LIST   [HAS settings -> chosen on validation]
# ==========================================================================
rule("D. WORD-LIST BASELINE   [settings chosen on VALIDATION]")

print("""A word's 'rate' is how reliably offensive it is in the training data:
how many times it was labelled offensive, divided by how many times it
appeared. A rate of 1.00 means it was offensive every single time.

  threshold  how strict the list is. A high threshold keeps only words that
             are almost always offensive: few flags, but usually correct.
             A low threshold keeps anything ever marked offensive: many
             flags, most of them wrong.
  min_count  how many times a word must appear before we trust its rate.
             A word seen once and offensive once has a rate of 1.00, which
             proves nothing.

Both are settings we CHOOSE, so both are searched on validation only.
""")


def build_lexicon(df, threshold, min_count):
    total, pos = Counter(), Counter()
    for toks, rats in zip(df["token_list"], df["rationales"]):
        for t, r in zip(toks, rats):
            total[t] += 1
            if r == 1:
                pos[t] += 1
    return {t for t in pos if total[t] >= min_count and pos[t] / total[t] > threshold}


def lexicon_score(df, lexicon, gold):
    pred = [[1 if t in lexicon else 0 for t in toks] for toks in df["token_list"]]
    return token_level_scores(gold, pred)


# ---- D1. search on validation
print("-" * 72)
print("D1. SEARCH ON VALIDATION (test is not touched)")
print("-" * 72)

results = []
for mc in MIN_COUNTS:
    for th in THRESHOLDS:
        lex = build_lexicon(train_part, th, mc)
        if not lex:
            continue
        s = lexicon_score(val, lex, gold_val)
        results.append(dict(min_count=mc, threshold=th, size=len(lex),
                            precision=s["offensive_precision"],
                            recall=s["offensive_recall"],
                            f1=s["offensive_f1"]))

results.sort(key=lambda r: -r["f1"])
print(f"{'min_cnt':>7} {'thresh':>7} {'words':>7} {'val P':>7} {'val R':>7} {'val F1':>7}")
print("-" * 47)
for r in results[:15]:
    print(f"{r['min_count']:>7} {r['threshold']:>7.1f} {r['size']:>7,} "
          f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f}")
print(f"\n{len(results)} settings tried, top 15 shown.")

best = results[0]
print(f"\nCHOSEN ON VALIDATION: min_count={best['min_count']}  "
      f"threshold={best['threshold']}  val F1={best['f1']:.4f}")

# ---- D2. refit on the full official train split
print("\n" + "-" * 72)
print("D2. REBUILD ON FULL TRAIN WITH THE CHOSEN SETTING")
print("-" * 72)
final_lex = build_lexicon(train_full, best["threshold"], best["min_count"])
print(f"word list size on full train: {len(final_lex):,}")
print("Standard practice: once the setting is fixed, use all the training data.")

# ---- D3. one shot at test
print("\n" + "-" * 72)
print("D3. RUN ONCE ON TEST")
print("-" * 72)
s_lex = lexicon_score(test, final_lex, gold_test)
print(f"  precision {s_lex['offensive_precision']:.4f}")
print(f"  recall    {s_lex['offensive_recall']:.4f}")
print(f"  F1        {s_lex['offensive_f1']:.4f}   <- REPORT THIS")
print(f"\n  validation F1 was {best['f1']:.4f}, test F1 is {s_lex['offensive_f1']:.4f} "
      f"(gap {s_lex['offensive_f1'] - best['f1']:+.4f}).")
print("  A small gap means validation predicts test well. A large gap means")
print("  the validation split is too small or unlucky.")
print("""
DO NOT go back and try more settings to push this number up. That would be
the same mistake the earlier version of this file made.""")


# ==========================================================================
# SUMMARY
# ==========================================================================
rule("SUMMARY - COPY INTO THE README")

print(f"""
Baselines on the SOLD test split, offensive-class F1:

  flag every token              {s_all['offensive_f1']:.4f}
  flag no token                 {s_none['offensive_f1']:.4f}
  random at train rate          {s_rand['offensive_f1']:.4f}
  word list (val-chosen)        {s_lex['offensive_f1']:.4f}   <- THE FLOOR TO BEAT
      setting: min_count={best['min_count']}, threshold={best['threshold']}, {len(final_lex):,} words
      test P={s_lex['offensive_precision']:.4f}  R={s_lex['offensive_recall']:.4f}

Published, for comparison:
  BiLSTM + CBOW                 0.58   (no pretrained model)
  BiLSTM + fastText             0.60   (no pretrained model)
  SinBERT                       0.62
  XLM-T                         0.70
  XLM-R                         0.72
  XLM-R + TSD transfer          0.73   (best published)

Our BiLSTM in Step 4 must clearly beat the word list, not just 0.60.
A lookup table is the dumbest possible model. If a neural network cannot
beat it, the neural network is broken or pointless.
""")