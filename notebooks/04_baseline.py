"""
Phase 1 / Step 4 - reproduce the BiLSTM + fastText token-level baseline.

TARGET: offensive-class F1 around 0.60, matching the published number.
This is the exit criterion for Phase 1. Nothing new is invented here.

Requires Step 3 to have been run (artifacts/embedding_matrix.npy exists).

    pip install torch pytorch-crf
    python notebooks/04_baseline.py > results/step4_report.txt

Options:
    --no-crf        train without the CRF layer (do this FIRST)
    --seeds 1 2 3   which seeds to run
    --weighted      use inverse-frequency class weights
    --unfreeze      let the embedding layer train
    --epochs 30
"""
import sys, os, argparse, csv, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from data import load_sold, train_val_split
from embeddings import build_vocab
from dataset import make_loader
from model import BiLSTMTagger
from train import train_one_seed, evaluate, get_device, compute_class_weights
from metrics import aggregate_seeds, format_scores

MATRIX_PATH = "artifacts/embedding_matrix.npy"
RESULTS_CSV = "results/results.csv"


def rule(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


p = argparse.ArgumentParser()
p.add_argument("--no-crf", action="store_true")
p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
p.add_argument("--weighted", action="store_true")
p.add_argument("--unfreeze", action="store_true")
p.add_argument("--hidden", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--dropout", type=float, default=0.5)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--patience", type=int, default=5)
p.add_argument("--tag", type=str, default="baseline")
args = p.parse_args()

use_crf = not args.no_crf

# ==========================================================================
rule("0. SETUP")
device = get_device()
print(f"device: {device}")

train_full = load_sold("train")
test = load_sold("test")
train_part, val = train_val_split(train_full)
print(f"train-part {len(train_part):,}  validation {len(val):,}  test {len(test):,}")

vocab, _ = build_vocab(train_part["token_list"], min_freq=1)
print(f"vocabulary {len(vocab):,}")

if not os.path.exists(MATRIX_PATH):
    print(f"\nMISSING {MATRIX_PATH}. Run notebooks/03_embeddings.py first.")
    sys.exit(1)
matrix = np.load(MATRIX_PATH)
print(f"embedding matrix {matrix.shape}")
if matrix.shape[0] != len(vocab):
    print(f"\nMISMATCH: matrix has {matrix.shape[0]} rows but vocab has {len(vocab)}.")
    print("Re-run notebooks/03_embeddings.py so the two agree.")
    sys.exit(1)

class_weights = compute_class_weights(train_part, device) if args.weighted else None
print(f"CRF: {use_crf}   class weights: "
      f"{None if class_weights is None else [round(float(w),3) for w in class_weights]}")
print(f"embeddings frozen: {not args.unfreeze}")


def model_fn():
    return BiLSTMTagger(
        matrix, hidden_size=args.hidden, dropout=args.dropout,
        freeze_embeddings=not args.unfreeze, use_crf=use_crf,
    )


print("\nparameters:")
for k, v in model_fn().count_parameters().items():
    print(f"  {k:<26} {v:>12,}")
print("""
Compare with XLM-R-large at about 560,000,000 parameters. The efficiency
claim in the paper rests on these numbers, so record them.""")


# ==========================================================================
rule("1. TRAIN OVER SEEDS")
test_loader = make_loader(test, vocab, args.batch_size, shuffle=False)

val_runs, test_runs, times = [], [], []
for seed in args.seeds:
    print(f"\n--- seed {seed} ---")
    model, hist, best_val, secs = train_one_seed(
        model_fn, train_part, val, vocab, seed,
        batch_size=args.batch_size, lr=args.lr, max_epochs=args.epochs,
        patience=args.patience, class_weights=class_weights, device=device,
    )
    test_scores = evaluate(model, test_loader, device)
    val_runs.append(best_val)
    test_runs.append(test_scores)
    times.append(secs)
    print(f"    best val F1 {best_val['offensive_f1']:.4f}   "
          f"TEST F1 {test_scores['offensive_f1']:.4f}   ({secs:.0f}s, {len(hist)} epochs)")

    os.makedirs("results", exist_ok=True)
    new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["tag", "seed", "crf", "weighted", "frozen_emb", "hidden",
                        "lr", "dropout", "epochs_run", "seconds",
                        "val_f1", "test_p", "test_r", "test_f1"])
        w.writerow([args.tag, seed, use_crf, args.weighted, not args.unfreeze,
                    args.hidden, args.lr, args.dropout, len(hist), round(secs, 1),
                    round(best_val["offensive_f1"], 4),
                    round(test_scores["offensive_precision"], 4),
                    round(test_scores["offensive_recall"], 4),
                    round(test_scores["offensive_f1"], 4)])


# ==========================================================================
rule("2. RESULTS OVER SEEDS")
agg_val = aggregate_seeds(val_runs)
agg_test = aggregate_seeds(test_runs)

print("F1 is computed per seed and then averaged. We never average precision")
print("and recall and then combine them - that gives a different number.\n")
print(f"{'metric':<24} {'mean':>8} {'std':>8}")
print("-" * 42)
for k in ["offensive_precision", "offensive_recall", "offensive_f1"]:
    print(f"test  {k:<18} {agg_test[k]['mean']:>8.4f} {agg_test[k]['std']:>8.4f}")
print(f"val   offensive_f1        {agg_val['offensive_f1']['mean']:>8.4f} "
      f"{agg_val['offensive_f1']['std']:>8.4f}")
per_seed = ", ".join(f"{r['offensive_f1']:.4f}" for r in test_runs)
print(f"\nper-seed test F1: {per_seed}")
print(f"mean training time: {statistics.mean(times):.0f}s per seed")


# ==========================================================================
rule("3. DID WE HIT THE TARGET?")
f1 = agg_test["offensive_f1"]["mean"]
print(f"""
our BiLSTM + fastText        {f1:.4f}
published BiLSTM + fastText  0.60     <- the Phase 1 target
our word-list baseline       0.6521   <- the real floor
published SinBERT            0.62
published XLM-R              0.72
""")

if 0.57 <= f1 <= 0.63:
    print("PASS. Reproduction is in range. Phase 1 exit criterion met.")
elif f1 > 0.63:
    print("ABOVE the published number. Good, but check for leakage before")
    print("celebrating: is the vocabulary built from train only? Is test")
    print("untouched during training and model selection?")
else:
    print("BELOW range. Work through these in order before changing the model:")
    print("  1. Is the model predicting anything? Check precision/recall are")
    print("     not both 0 - that means it collapsed to all-negative. Try")
    print("     --weighted.")
    print("  2. Are labels aligned with tokens? Re-check Step 1 section 4.")
    print("  3. Is padding excluded from the loss and the metric?")
    print("  4. Try --no-crf to isolate whether the CRF is the problem.")
    print("  5. Try more epochs or a different learning rate.")

if f1 < 0.6521:
    print(f"""
NOTE: the model at {f1:.4f} does NOT beat the word list at 0.6521. That is
expected at Phase 1 - the baseline is a faithful reproduction, not our
contribution. Phase 2 is where we beat it.""")


# ==========================================================================
rule("4. FOR THE README")
print(f"""
model                 BiLSTM + fastText{' + CRF' if use_crf else ''}
hidden size           {args.hidden}
embeddings            300d fastText, {'frozen' if not args.unfreeze else 'trainable'}
class weights         {'yes' if args.weighted else 'no'}
seeds                 {args.seeds}
test offensive F1     {agg_test['offensive_f1']['mean']:.4f} +/- {agg_test['offensive_f1']['std']:.4f}
test precision        {agg_test['offensive_precision']['mean']:.4f}
test recall           {agg_test['offensive_recall']['mean']:.4f}
parameters            {model_fn().count_parameters()['total']:,} total
training time         {statistics.mean(times):.0f}s per seed on {device}
""")


# ========================================================================
# 0. SETUP
# ========================================================================
# device: mps
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# train-part 6,000  validation 1,500  test 2,500
# vocabulary 28,456
# embedding matrix (28456, 300)
# CRF: False   class weights: None
# embeddings frozen: True

# parameters:
#   total                         8,724,450
#   trainable                       187,650
#   frozen                        8,536,800
#   embedding                     8,536,800
#   non_embedding_trainable         187,650

# Compare with XLM-R-large at about 560,000,000 parameters. The efficiency
# claim in the paper rests on these numbers, so record them.

# ========================================================================
# 1. TRAIN OVER SEEDS
# ========================================================================

# --- seed 1 ---
#     epoch  1  loss 0.1924  val P 0.7011  R 0.0432  F1 0.0814
#     epoch  2  loss 0.1131  val P 0.7836  R 0.2436  F1 0.3717
#     epoch  3  loss 0.1029  val P 0.7222  R 0.3499  F1 0.4714
#     epoch  4  loss 0.0988  val P 0.7957  R 0.2897  F1 0.4247
#     epoch  5  loss 0.0959  val P 0.8068  R 0.3045  F1 0.4422
#     epoch  6  loss 0.0927  val P 0.8111  R 0.3222  F1 0.4612
#     epoch  7  loss 0.0910  val P 0.7911  R 0.3647  F1 0.4993
#     epoch  8  loss 0.0882  val P 0.7961  R 0.3761  F1 0.5108
#     epoch  9  loss 0.0874  val P 0.8003  R 0.3803  F1 0.5156
#     epoch 10  loss 0.0877  val P 0.7662  R 0.4596  F1 0.5746
#     best val F1 0.5746   TEST F1 0.5773   (577s, 10 epochs)

# ========================================================================
# 2. RESULTS OVER SEEDS
# ========================================================================
# F1 is computed per seed and then averaged. We never average precision
# and recall and then combine them - that gives a different number.

# metric                       mean      std
# ------------------------------------------
# test  offensive_precision   0.7630   0.0000
# test  offensive_recall     0.4643   0.0000
# test  offensive_f1         0.5773   0.0000
# val   offensive_f1          0.5746   0.0000

# per-seed test F1: 0.5773
# mean training time: 577s per seed

# ========================================================================
# 3. DID WE HIT THE TARGET?
# ========================================================================

# our BiLSTM + fastText        0.5773
# published BiLSTM + fastText  0.60     <- the Phase 1 target
# our word-list baseline       0.6521   <- the real floor
# published SinBERT            0.62
# published XLM-R              0.72

# PASS. Reproduction is in range. Phase 1 exit criterion met.

# NOTE: the model at 0.5773 does NOT beat the word list at 0.6521. That is
# expected at Phase 1 - the baseline is a faithful reproduction, not our
# contribution. Phase 2 is where we beat it.

# ========================================================================
# 4. FOR THE README
# ========================================================================

# model                 BiLSTM + fastText
# hidden size           64
# embeddings            300d fastText, frozen
# class weights         no
# seeds                 [1]
# test offensive F1     0.5773 +/- 0.0000
# test precision        0.7630
# test recall           0.4643
# parameters            8,724,450 total
# training time         577s per seed on mps


# -------------------------------------------------------------------------------------------------------------------------------------------

# ========================================================================
# 0. SETUP
# ========================================================================
# device: mps
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# train-part 6,000  validation 1,500  test 2,500
# vocabulary 28,456
# embedding matrix (28456, 300)
# CRF: True   class weights: None
# embeddings frozen: True

# parameters:
#   total                         8,724,458
#   trainable                       187,658
#   frozen                        8,536,800
#   embedding                     8,536,800
#   non_embedding_trainable         187,658

# Compare with XLM-R-large at about 560,000,000 parameters. The efficiency
# claim in the paper rests on these numbers, so record them.

# ========================================================================
# 1. TRAIN OVER SEEDS
# ========================================================================

# --- seed 1 ---
#     epoch  1  loss 4.1824  val P 0.7764  R 0.1820  F1 0.2949
#     epoch  2  loss 2.3784  val P 0.7768  R 0.2613  F1 0.3911
#     epoch  3  loss 2.2325  val P 0.7281  R 0.3527  F1 0.4752
#     epoch  4  loss 2.1270  val P 0.7930  R 0.2712  F1 0.4042
#     epoch  5  loss 2.0609  val P 0.8211  R 0.2762  F1 0.4134
#     epoch  6  loss 1.9916  val P 0.8272  R 0.2847  F1 0.4236
#     epoch  7  loss 1.9611  val P 0.7983  R 0.3421  F1 0.4789
#     epoch  8  loss 1.9046  val P 0.8020  R 0.3414  F1 0.4789
#     epoch  9  loss 1.8866  val P 0.8047  R 0.3619  F1 0.4993
#     epoch 10  loss 1.8851  val P 0.7801  R 0.4044  F1 0.5326
#     best val F1 0.5326   TEST F1 0.5180   (559s, 10 epochs)

# ========================================================================
# 2. RESULTS OVER SEEDS
# ========================================================================
# F1 is computed per seed and then averaged. We never average precision
# and recall and then combine them - that gives a different number.

# metric                       mean      std
# ------------------------------------------
# test  offensive_precision   0.7806   0.0000
# test  offensive_recall     0.3876   0.0000
# test  offensive_f1         0.5180   0.0000
# val   offensive_f1          0.5326   0.0000

# per-seed test F1: 0.5180
# mean training time: 559s per seed

# ========================================================================
# 3. DID WE HIT THE TARGET?
# ========================================================================

# our BiLSTM + fastText        0.5180
# published BiLSTM + fastText  0.60     <- the Phase 1 target
# our word-list baseline       0.6521   <- the real floor
# published SinBERT            0.62
# published XLM-R              0.72

# BELOW range. Work through these in order before changing the model:
#   1. Is the model predicting anything? Check precision/recall are
#      not both 0 - that means it collapsed to all-negative. Try
#      --weighted.
#   2. Are labels aligned with tokens? Re-check Step 1 section 4.
#   3. Is padding excluded from the loss and the metric?
#   4. Try --no-crf to isolate whether the CRF is the problem.
#   5. Try more epochs or a different learning rate.

# NOTE: the model at 0.5180 does NOT beat the word list at 0.6521. That is
# expected at Phase 1 - the baseline is a faithful reproduction, not our
# contribution. Phase 2 is where we beat it.

# ========================================================================
# 4. FOR THE README
# ========================================================================

# model                 BiLSTM + fastText + CRF
# hidden size           64
# embeddings            300d fastText, frozen
# class weights         no
# seeds                 [1]
# test offensive F1     0.5180 +/- 0.0000
# test precision        0.7806
# test recall           0.3876
# parameters            8,724,458 total
# training time         559s per seed on mps



# -------------------------------------------------------------------------------------------------------------------------------------------

