"""
Token-level evaluation for SOLD.

THE ONLY PLACE F1 IS COMPUTED. Nothing else defines its own metric.

--------------------------------------------------------------------------
SOURCE OF TRUTH
--------------------------------------------------------------------------
Reimplemented from the official SOLD repository:
    experiments/token_level/print_stat.py   (function print_information)
    experiments/token_level/sinhala_mudes.py (how it is called)
    https://github.com/Sinhala-NLP/SOLD

Three facts established by reading that code:

1. EVALUATION IS POOLED, NOT PER-SENTENCE.
   sinhala_mudes.py builds one long DataFrame with a row per token across
   every test tweet, then evaluates that flat list in a single call. So all
   2,500 test tweets contribute their tokens to one pool. There is no
   per-sentence averaging.

2. THE HEADLINE NUMBER IS THE F1 OF THE OFFENSIVE CLASS.
   print_information reports a per-class block (precision/recall/F1 for
   each label) and separately a macro F1 over both classes. The paper's
   prose says "Macro F1" but the reported values are the OFF-class ones:
   for XLM-R the paper gives P=0.68, R=0.76, F1=0.72, and the harmonic
   mean of 0.68 and 0.76 is 0.718. A macro F1 over both classes would be
   about 0.85, because the NOT class is easy and dominant. So the table is
   per-class OFF. We report BOTH here and compare on offensive_f1.

3. THEIR PRECISION AND RECALL ARE SWAPPED.
   print_information's signature is (df, pred_column, real_column) and it
   assigns predictions = df[pred_column], real_values = df[real_column].
   sinhala_mudes.py calls print_information(test_data, "labels",
   "predictions") - so "labels" (the GOLD column) is bound to predictions,
   and "predictions" (the MODEL column) is bound to real_values. Those are
   then passed to sklearn as (y_true=model_output, y_pred=gold), which is
   backwards. Swapping the arguments turns recall into precision and
   precision into recall. F1 is symmetric, so F1 and macro F1 are correct.

   CONSEQUENCE: match the published F1, and do not expect to match the
   published precision and recall in the order printed - they are likely
   reversed. We compute P and R in the CORRECT orientation here.

4. EMPTY RATIONALES BECOME ALL ZEROS, AND THOSE ROWS ARE KEPT.
   sinhala_mudes.py: `if len(labels) == 0: ... label 0 for every token`.
   No row is dropped. This confirms our loader's behaviour and our README
   decision to keep the 209 train / 74 test OFF-with-no-offensive-token rows.

5. SHORT PREDICTIONS ARE PADDED WITH THE NEGATIVE CLASS.
   If a model returns fewer predictions than the tweet has tokens, they pad
   with "NOT". align_predictions() below does the same.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from sklearn.metrics import f1_score, precision_score, recall_score

POSITIVE = 1  # offensive
NEGATIVE = 0  # not offensive


# --------------------------------------------------------------------------
# flattening helpers
# --------------------------------------------------------------------------

def align_predictions(pred: Sequence[int], n_tokens: int) -> List[int]:
    """Make one prediction list exactly n_tokens long.

    Too short -> pad with the negative class (what the SOLD code does).
    Too long  -> truncate.
    """
    pred = list(pred)
    if len(pred) < n_tokens:
        pred = pred + [NEGATIVE] * (n_tokens - len(pred))
    return pred[:n_tokens]


def flatten(
    gold_seqs: Iterable[Sequence[int]],
    pred_seqs: Iterable[Sequence[int]],
) -> tuple[List[int], List[int]]:
    """Pool every token from every sentence into two flat lists.

    Padding positions must already be removed: pass real token labels only,
    one list per sentence, not a padded tensor.
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    for gold, pred in zip(gold_seqs, pred_seqs):
        gold = list(gold)
        pred = align_predictions(pred, len(gold))
        y_true.extend(gold)
        y_pred.extend(pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"flatten produced {len(y_true)} gold vs {len(y_pred)} pred")
    return y_true, y_pred


# --------------------------------------------------------------------------
# the metric
# --------------------------------------------------------------------------

def token_level_scores(
    gold_seqs: Iterable[Sequence[int]],
    pred_seqs: Iterable[Sequence[int]],
) -> Dict[str, float]:
    """Compute the full token-level score block.

    Returns, all in the CORRECT (y_true, y_pred) orientation:
        offensive_precision, offensive_recall, offensive_f1   <- HEADLINE
        not_offensive_precision, not_offensive_recall, not_offensive_f1
        macro_f1, weighted_f1
        support_offensive, support_total

    Compare published numbers against offensive_f1.
    """
    y_true, y_pred = flatten(gold_seqs, pred_seqs)

    kw = dict(labels=[NEGATIVE, POSITIVE], zero_division=0)

    return {
        "offensive_precision": float(precision_score(y_true, y_pred, pos_label=POSITIVE, **kw)),
        "offensive_recall": float(recall_score(y_true, y_pred, pos_label=POSITIVE, **kw)),
        "offensive_f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE, **kw)),
        "not_offensive_precision": float(precision_score(y_true, y_pred, pos_label=NEGATIVE, **kw)),
        "not_offensive_recall": float(recall_score(y_true, y_pred, pos_label=NEGATIVE, **kw)),
        "not_offensive_f1": float(f1_score(y_true, y_pred, pos_label=NEGATIVE, **kw)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", **kw)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", **kw)),
        "support_offensive": int(sum(1 for v in y_true if v == POSITIVE)),
        "support_total": len(y_true),
    }


def format_scores(scores: Dict[str, float]) -> str:
    return (
        f"OFFENSIVE  P={scores['offensive_precision']:.4f}  "
        f"R={scores['offensive_recall']:.4f}  "
        f"F1={scores['offensive_f1']:.4f}   <- headline\n"
        f"NOT        P={scores['not_offensive_precision']:.4f}  "
        f"R={scores['not_offensive_recall']:.4f}  "
        f"F1={scores['not_offensive_f1']:.4f}\n"
        f"macro F1={scores['macro_f1']:.4f}   weighted F1={scores['weighted_f1']:.4f}\n"
        f"offensive tokens: {scores['support_offensive']:,} / {scores['support_total']:,}"
    )


# --------------------------------------------------------------------------
# seed aggregation
# --------------------------------------------------------------------------

def aggregate_seeds(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Mean and std across seeds.

    IMPORTANT: compute F1 per seed FIRST (one token_level_scores call per
    seed), then average the F1 values here. Never average precision and
    recall across seeds and then combine them - that gives a different
    number, and it is the likely reason the paper's BiLSTM rows do not
    match the harmonic mean of their own P and R.
    """
    import statistics

    keys = [k for k in runs[0] if not k.startswith("support")]
    out = {}
    for k in keys:
        vals = [r[k] for r in runs]
        out[k] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return out