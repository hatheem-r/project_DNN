"""Unit tests for src/metrics.py. Run: python tests/test_metrics.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from metrics import token_level_scores, align_predictions, flatten, aggregate_seeds

def approx(a, b, tol=1e-9): assert abs(a-b) < tol, (a, b)

# 1. perfect prediction
s = token_level_scores([[0,1,0],[0,0]], [[0,1,0],[0,0]])
approx(s["offensive_f1"], 1.0); approx(s["macro_f1"], 1.0)
print("1 perfect prediction               OK")

# 2. predicts all negative -> offensive F1 must be 0, and NOT F1 high
s = token_level_scores([[0,1,0,0],[0,0,0,0]], [[0,0,0,0],[0,0,0,0]])
approx(s["offensive_f1"], 0.0); approx(s["offensive_recall"], 0.0)
assert s["not_offensive_f1"] > 0.9
print("2 all-negative collapse detected   OK")

# 3. hand-computed case
# gold 1,1,0,0  pred 1,0,1,0  -> TP=1 FP=1 FN=1 -> P=.5 R=.5 F1=.5
s = token_level_scores([[1,1,0,0]], [[1,0,1,0]])
approx(s["offensive_precision"], 0.5); approx(s["offensive_recall"], 0.5)
approx(s["offensive_f1"], 0.5)
print("3 hand-computed P/R/F1             OK")

# 4. precision and recall are NOT symmetric (catches swapped arguments)
# gold has 1 positive, pred has 3 -> P=1/3, R=1/1
s = token_level_scores([[1,0,0,0]], [[1,1,1,0]])
approx(s["offensive_precision"], 1/3); approx(s["offensive_recall"], 1.0)
assert s["offensive_precision"] != s["offensive_recall"]
print("4 orientation not swapped          OK")

# 5. pooling: sentences are concatenated, not averaged per sentence
pooled = token_level_scores([[1,0],[1,0,0,0,0,0,0,0]], [[1,0],[0,0,0,0,0,0,0,0]])
# TP=1 FN=1 FP=0 -> P=1.0 R=0.5 F1=2/3.  A per-sentence mean would give 0.5.
approx(pooled["offensive_precision"], 1.0); approx(pooled["offensive_recall"], 0.5)
approx(pooled["offensive_f1"], 2/3)
print("5 pooled not per-sentence          OK")

# 6. macro F1 differs from offensive F1 (proves they are distinct numbers)
s = token_level_scores([[1,1,0,0,0,0,0,0,0,0]], [[1,0,1,1,0,0,0,0,0,0]])
# offensive F1 = 0.40, NOT F1 = 0.80, macro = 0.60. Reporting macro would
# flatter the model by 0.20 here. This is why we compare on offensive_f1.
approx(s["offensive_f1"], 0.4); approx(s["not_offensive_f1"], 0.8)
approx(s["macro_f1"], 0.6)
print("6 macro F1 != offensive F1         OK")

# 7. short predictions padded with NOT, long truncated
assert align_predictions([1], 4) == [1,0,0,0]
assert align_predictions([1,1,1,1,1], 3) == [1,1,1]
yt, yp = flatten([[1,1,1]], [[1]])
assert yp == [1,0,0]
print("7 padding / truncation             OK")

# 8. no positives anywhere -> zero_division must not crash
s = token_level_scores([[0,0]], [[0,0]])
approx(s["offensive_f1"], 0.0)
print("8 empty positive class safe        OK")

# 9. seed aggregation averages F1 values, not P and R
runs = [token_level_scores([[1,1,0,0]], [[1,0,0,0]]),
        token_level_scores([[1,1,0,0]], [[1,1,0,1]])]
agg = aggregate_seeds(runs)
expected = (runs[0]["offensive_f1"] + runs[1]["offensive_f1"]) / 2
approx(agg["offensive_f1"]["mean"], expected)
assert agg["offensive_f1"]["std"] > 0
print("9 seed aggregation                 OK")

print("\nall 9 tests passed")