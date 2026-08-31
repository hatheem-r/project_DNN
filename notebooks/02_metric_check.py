"""
Phase 1 / Step 2 - verify the evaluation metric on real SOLD data.

Run:  python notebooks/02_metric_check.py

Evaluates three trivial baselines that need no training. Their scores are
predictable, so they tell you whether metrics.py is wired up correctly
BEFORE you write any model.
"""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from collections import Counter
from data import load_sold
from metrics import token_level_scores, format_scores


def rule(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


train = load_sold("train")
test = load_sold("test")
gold = list(test["rationales"])

# ---------------------------------------------------------------- baseline 1
rule("BASELINE 1: predict OFFENSIVE for every token  ('All OFF' in the paper)")
pred = [[1] * n for n in test["n_tokens"]]
s = token_level_scores(gold, pred)
print(format_scores(s))
print(f"""
CHECK: offensive precision should equal the positive token rate (~0.038),
because if you flag everything, your precision is just the base rate.
Recall must be exactly 1.0. The paper reports 0.03 precision for this
baseline, which is where our ~3-4% imbalance figure comes from.
  precision = {s['offensive_precision']:.4f}   recall = {s['offensive_recall']:.4f}""")

# ---------------------------------------------------------------- baseline 2
rule("BASELINE 2: predict NOT OFFENSIVE for every token")
pred = [[0] * n for n in test["n_tokens"]]
s = token_level_scores(gold, pred)
print(format_scores(s))
print("""
CHECK: offensive F1 must be exactly 0.0000, but 'accuracy' would be ~96%.
This is the single clearest reason we never report accuracy.
Note the macro F1 is still around 0.49 - proof that macro F1 over BOTH
classes flatters a useless model, and why we compare on offensive_f1.""")

# ---------------------------------------------------------------- baseline 3
rule("BASELINE 3: lexicon lookup - flag any token seen as offensive in train")
tok_total, tok_pos = Counter(), Counter()
for toks, rats in zip(train["token_list"], train["rationales"]):
    for t, r in zip(toks, rats):
        tok_total[t] += 1
        if r == 1:
            tok_pos[t] += 1

for thresh in (0.0, 0.3, 0.5, 0.7):
    lex = {t for t in tok_pos if tok_total[t] >= 3 and tok_pos[t] / tok_total[t] > thresh}
    pred = [[1 if t in lex else 0 for t in toks] for toks in test["token_list"]]
    s = token_level_scores(gold, pred)
    print(f"  rate > {thresh:.1f}  lexicon size {len(lex):>5,}   "
          f"P={s['offensive_precision']:.3f} R={s['offensive_recall']:.3f} "
          f"F1={s['offensive_f1']:.3f}")

print("""
CHECK: this is the number your neural model must beat to justify existing.
A word list is the dumbest possible 'model'. If your BiLSTM cannot clearly
beat it, the BiLSTM is broken. Write the best F1 here into the README.""")

# ---------------------------------------------------------------- baseline 4
rule("BASELINE 4: random, at the training positive rate")
rate = train["n_offensive_tokens"].sum() / train["n_tokens"].sum()
random.seed(42)
pred = [[1 if random.random() < rate else 0 for _ in range(n)] for n in test["n_tokens"]]
s = token_level_scores(gold, pred)
print(f"  positive rate used: {rate:.4f}")
print(f"  P={s['offensive_precision']:.4f} R={s['offensive_recall']:.4f} "
      f"F1={s['offensive_f1']:.4f}")
print("""
CHECK: precision and recall should both land near the positive rate.
Random guessing gives F1 ~0.04. Any real model must be far above this.""")

rule("SUMMARY - put these in the README")
print("""
Trivial baselines on the SOLD test split, offensive-class F1:
  all offensive     -> see BASELINE 1
  all not offensive -> 0.0000
  lexicon lookup    -> see BASELINE 3 (this is the floor to beat)
  random            -> see BASELINE 4
Published BiLSTM + fastText -> 0.60   (our Step 4/5 target)
Published XLM-R             -> 0.72
""")