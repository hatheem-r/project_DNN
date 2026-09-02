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


# 1. Simplest possible version. Confirms the pipeline works.
python notebooks/04_baseline.py --no-crf --seeds 1 --epochs 10 --tag smoke

# 2. Add the CRF. Score should go up.
python notebooks/04_baseline.py --seeds 1 --epochs 10 --tag crf

# 3. Full run, five seeds.
python notebooks/04_baseline.py > results/step4_report.txt

python notebooks/04_baseline.py --epochs 60 --patience 12 --tag tuned



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
#     epoch 11  loss 1.8471  val P 0.8163  R 0.3555  F1 0.4953
#     epoch 12  loss 1.8243  val P 0.8165  R 0.3499  F1 0.4898
#     epoch 13  loss 1.8015  val P 0.7810  R 0.4143  F1 0.5414
#     epoch 14  loss 1.7897  val P 0.7932  R 0.3938  F1 0.5263
#     epoch 15  loss 1.7741  val P 0.8051  R 0.3803  F1 0.5166
#     epoch 16  loss 1.7525  val P 0.7827  R 0.4108  F1 0.5388
#     epoch 17  loss 1.7473  val P 0.7967  R 0.4079  F1 0.5396
#     epoch 18  loss 1.7170  val P 0.7878  R 0.4391  F1 0.5639
#     epoch 19  loss 1.6919  val P 0.7723  R 0.4299  F1 0.5523
#     epoch 20  loss 1.6993  val P 0.7508  R 0.4823  F1 0.5873
#     epoch 21  loss 1.6782  val P 0.7899  R 0.4207  F1 0.5490
#     epoch 22  loss 1.6487  val P 0.7949  R 0.4008  F1 0.5330
#     epoch 23  loss 1.6519  val P 0.7872  R 0.4271  F1 0.5537
#     epoch 24  loss 1.6348  val P 0.7749  R 0.4632  F1 0.5798
#     epoch 25  loss 1.6236  val P 0.7934  R 0.4242  F1 0.5528
#     epoch 26  loss 1.6131  val P 0.7614  R 0.4745  F1 0.5846
#     epoch 27  loss 1.6015  val P 0.7631  R 0.4632  F1 0.5765
#     epoch 28  loss 1.5909  val P 0.7926  R 0.3924  F1 0.5249
#     epoch 29  loss 1.5590  val P 0.7578  R 0.4632  F1 0.5749
#     epoch 30  loss 1.5701  val P 0.7239  R 0.5198  F1 0.6051
#     epoch 31  loss 1.5239  val P 0.7406  R 0.5035  F1 0.5995
#     epoch 32  loss 1.5310  val P 0.7692  R 0.4603  F1 0.5760
#     epoch 33  loss 1.5113  val P 0.7544  R 0.4894  F1 0.5936
#     epoch 34  loss 1.4945  val P 0.7821  R 0.4525  F1 0.5734
#     epoch 35  loss 1.4779  val P 0.8068  R 0.3874  F1 0.5234
#     epoch 36  loss 1.4666  val P 0.7531  R 0.4795  F1 0.5859
#     epoch 37  loss 1.4567  val P 0.7122  R 0.5135  F1 0.5967
#     epoch 38  loss 1.4380  val P 0.7660  R 0.4568  F1 0.5723
#     epoch 39  loss 1.4289  val P 0.7392  R 0.4837  F1 0.5848
#     epoch 40  loss 1.4014  val P 0.7607  R 0.4773  F1 0.5866
#     epoch 41  loss 1.4155  val P 0.7835  R 0.4511  F1 0.5726
#     epoch 42  loss 1.3730  val P 0.7463  R 0.4979  F1 0.5973
#     early stop at epoch 42 (best F1 0.6051)
#     best val F1 0.6051   TEST F1 0.6023   (2301s, 42 epochs)

# --- seed 2 ---
#     epoch  1  loss 4.1723  val P 0.7472  R 0.2387  F1 0.3618
#     epoch  2  loss 2.3860  val P 0.7569  R 0.3088  F1 0.4386
#     epoch  3  loss 2.2423  val P 0.7516  R 0.3343  F1 0.4627
#     epoch  4  loss 2.1435  val P 0.8004  R 0.2727  F1 0.4068
#     epoch  5  loss 2.0530  val P 0.7989  R 0.3208  F1 0.4578
#     epoch  6  loss 2.0083  val P 0.8062  R 0.3329  F1 0.4712
#     epoch  7  loss 1.9597  val P 0.7964  R 0.3768  F1 0.5115
# ^[[C^[[C^[[C^[[C^[[C^[[C^[[C^[[C^[[C^[[C^[[C^[[C^[[C^[[C^[[C^[[C^[[C    epoch  8  loss 1.9200  val P 0.7566  R 0.4448  F1 0.5602
#     epoch  9  loss 1.9047  val P 0.7918  R 0.3987  F1 0.5304
#     epoch 10  loss 1.8724  val P 0.7901  R 0.4051  F1 0.5356
#     epoch 11  loss 1.8333  val P 0.8353  R 0.3484  F1 0.4918
#     epoch 12  loss 1.8223  val P 0.8045  R 0.3789  F1 0.5152
#     epoch 13  loss 1.8082  val P 0.8086  R 0.3711  F1 0.5087
#     epoch 14  loss 1.7777  val P 0.8561  R 0.3244  F1 0.4705
#     epoch 15  loss 1.7619  val P 0.7924  R 0.4001  F1 0.5318
#     epoch 16  loss 1.7446  val P 0.8000  R 0.3909  F1 0.5252
#     epoch 17  loss 1.7252  val P 0.8364  R 0.3513  F1 0.4948
#     epoch 18  loss 1.7165  val P 0.7968  R 0.4193  F1 0.5494
#     epoch 19  loss 1.6846  val P 0.8190  R 0.3782  F1 0.5174
#     epoch 20  loss 1.6772  val P 0.7732  R 0.4490  F1 0.5681
#     epoch 21  loss 1.6692  val P 0.7861  R 0.4320  F1 0.5576
#     epoch 22  loss 1.6762  val P 0.7767  R 0.4582  F1 0.5764
#     epoch 23  loss 1.6416  val P 0.7974  R 0.4348  F1 0.5628
#     epoch 24  loss 1.6136  val P 0.8039  R 0.4327  F1 0.5626
#     epoch 25  loss 1.6111  val P 0.7838  R 0.4391  F1 0.5629
#     epoch 26  loss 1.6025  val P 0.8056  R 0.4051  F1 0.5391
#     epoch 27  loss 1.5891  val P 0.7902  R 0.4242  F1 0.5521
#     epoch 28  loss 1.6017  val P 0.7845  R 0.4433  F1 0.5665
#     epoch 29  loss 1.5667  val P 0.7915  R 0.4356  F1 0.5619
#     epoch 30  loss 1.5446  val P 0.7421  R 0.4993  F1 0.5970
#     epoch 31  loss 1.5288  val P 0.7733  R 0.4710  F1 0.5854
#     epoch 32  loss 1.5193  val P 0.7850  R 0.4603  F1 0.5804
#     epoch 33  loss 1.5025  val P 0.7822  R 0.4681  F1 0.5857
#     epoch 34  loss 1.4949  val P 0.7592  R 0.4844  F1 0.5914
#     epoch 35  loss 1.4663  val P 0.7699  R 0.4738  F1 0.5866
#     epoch 36  loss 1.4686  val P 0.7773  R 0.4646  F1 0.5816
#     epoch 37  loss 1.4528  val P 0.7680  R 0.4759  F1 0.5877
#     epoch 38  loss 1.4475  val P 0.8199  R 0.4093  F1 0.5461
#     epoch 39  loss 1.4105  val P 0.7725  R 0.4568  F1 0.5741
#     epoch 40  loss 1.3934  val P 0.7481  R 0.4943  F1 0.5953
#     epoch 41  loss 1.3906  val P 0.7731  R 0.4561  F1 0.5737
#     epoch 42  loss 1.3803  val P 0.7933  R 0.4348  F1 0.5618
#     early stop at epoch 42 (best F1 0.5970)
#     best val F1 0.5970   TEST F1 0.5828   (2238s, 42 epochs)

# --- seed 3 ---
#     epoch  1  loss 4.3022  val P 0.7600  R 0.2288  F1 0.3517
#     epoch  2  loss 2.3986  val P 0.7530  R 0.3067  F1 0.4358
#     epoch  3  loss 2.2507  val P 0.7810  R 0.2854  F1 0.4180
#     epoch  4  loss 2.1203  val P 0.7937  R 0.3024  F1 0.4379
#     epoch  5  loss 2.0619  val P 0.7729  R 0.3711  F1 0.5014
#     epoch  6  loss 1.9884  val P 0.7809  R 0.3584  F1 0.4913
#     epoch  7  loss 1.9575  val P 0.8237  R 0.3244  F1 0.4654
#     epoch  8  loss 1.9389  val P 0.7814  R 0.3697  F1 0.5019
#     epoch  9  loss 1.8758  val P 0.8087  R 0.3562  F1 0.4946
#     epoch 10  loss 1.8603  val P 0.8192  R 0.3626  F1 0.5027
#     epoch 11  loss 1.8374  val P 0.8200  R 0.3711  F1 0.5110
#     epoch 12  loss 1.8124  val P 0.8280  R 0.3307  F1 0.4727
#     epoch 13  loss 1.8012  val P 0.7646  R 0.4348  F1 0.5544
#     epoch 14  loss 1.7941  val P 0.8430  R 0.3194  F1 0.4633
#     epoch 15  loss 1.7788  val P 0.8107  R 0.3548  F1 0.4936
#     epoch 16  loss 1.7544  val P 0.8380  R 0.3371  F1 0.4808
#     epoch 17  loss 1.7356  val P 0.7902  R 0.4214  F1 0.5497
#     epoch 18  loss 1.7206  val P 0.8047  R 0.3853  F1 0.5211
#     epoch 19  loss 1.7001  val P 0.8191  R 0.3881  F1 0.5267
#     epoch 20  loss 1.6977  val P 0.7845  R 0.4023  F1 0.5318
#     epoch 21  loss 1.6810  val P 0.8248  R 0.3768  F1 0.5173
#     epoch 22  loss 1.6618  val P 0.8027  R 0.4150  F1 0.5472
#     epoch 23  loss 1.6186  val P 0.8106  R 0.4001  F1 0.5358
#     epoch 24  loss 1.6353  val P 0.7927  R 0.4143  F1 0.5442
#     epoch 25  loss 1.6194  val P 0.7773  R 0.4178  F1 0.5435
#     early stop at epoch 25 (best F1 0.5544)
#     best val F1 0.5544   TEST F1 0.5545   (1355s, 25 epochs)

# --- seed 4 ---
#     epoch  1  loss 4.1857  val P 0.7043  R 0.2564  F1 0.3759
#     epoch  2  loss 2.3543  val P 0.7514  R 0.2953  F1 0.4240
#     epoch  3  loss 2.2180  val P 0.7461  R 0.3081  F1 0.4361
#     epoch  4  loss 2.1142  val P 0.7619  R 0.3286  F1 0.4592
#     epoch  5  loss 2.0482  val P 0.7861  R 0.3435  F1 0.4781
#     epoch  6  loss 2.0011  val P 0.8187  R 0.2911  F1 0.4295
#     epoch  7  loss 1.9337  val P 0.7920  R 0.3506  F1 0.4860
#     epoch  8  loss 1.9270  val P 0.8355  R 0.3201  F1 0.4629
#     epoch  9  loss 1.8773  val P 0.8621  R 0.2967  F1 0.4415
#     epoch 10  loss 1.8536  val P 0.8044  R 0.3640  F1 0.5012
#     epoch 11  loss 1.8223  val P 0.7833  R 0.3916  F1 0.5222
#     epoch 12  loss 1.8061  val P 0.8231  R 0.3591  F1 0.5000
#     epoch 13  loss 1.7942  val P 0.7832  R 0.4093  F1 0.5377
#     epoch 14  loss 1.7712  val P 0.8370  R 0.3272  F1 0.4705
#     epoch 15  loss 1.7543  val P 0.8203  R 0.3492  F1 0.4898
#     epoch 16  loss 1.7344  val P 0.8211  R 0.3640  F1 0.5044
#     epoch 17  loss 1.7156  val P 0.8044  R 0.4136  F1 0.5463
#     epoch 18  loss 1.7034  val P 0.8164  R 0.3746  F1 0.5136
#     epoch 19  loss 1.7007  val P 0.8171  R 0.3924  F1 0.5301
#     epoch 20  loss 1.6843  val P 0.8028  R 0.4008  F1 0.5347
#     epoch 21  loss 1.6623  val P 0.7986  R 0.3931  F1 0.5268
#     epoch 22  loss 1.6589  val P 0.7984  R 0.4178  F1 0.5486
#     epoch 23  loss 1.6403  val P 0.8072  R 0.3973  F1 0.5325
#     epoch 24  loss 1.6175  val P 0.8014  R 0.4171  F1 0.5487
#     epoch 25  loss 1.5965  val P 0.7753  R 0.4348  F1 0.5572
#     epoch 26  loss 1.5975  val P 0.8046  R 0.3994  F1 0.5338
#     epoch 27  loss 1.5742  val P 0.7804  R 0.4405  F1 0.5632
#     epoch 28  loss 1.5585  val P 0.7705  R 0.4469  F1 0.5657
#     epoch 29  loss 1.5572  val P 0.7831  R 0.4398  F1 0.5633
#     epoch 30  loss 1.5401  val P 0.7681  R 0.4504  F1 0.5679
#     epoch 31  loss 1.5210  val P 0.7992  R 0.4285  F1 0.5579
#     epoch 32  loss 1.5074  val P 0.8176  R 0.3874  F1 0.5257
#     epoch 33  loss 1.4987  val P 0.8011  R 0.4193  F1 0.5504
#     epoch 34  loss 1.4990  val P 0.7799  R 0.4568  F1 0.5762
#     epoch 35  loss 1.4748  val P 0.7661  R 0.4710  F1 0.5833
#     epoch 36  loss 1.4396  val P 0.7414  R 0.5198  F1 0.6112
#     epoch 37  loss 1.4407  val P 0.7914  R 0.4433  F1 0.5683
#     epoch 38  loss 1.4325  val P 0.7557  R 0.5170  F1 0.6140
#     epoch 39  loss 1.4088  val P 0.7841  R 0.4603  F1 0.5801
#     epoch 40  loss 1.3956  val P 0.7662  R 0.4759  F1 0.5872
#     epoch 41  loss 1.3735  val P 0.7161  R 0.5269  F1 0.6071
#     epoch 42  loss 1.3723  val P 0.7281  R 0.5368  F1 0.6180
#     epoch 43  loss 1.3349  val P 0.7533  R 0.4887  F1 0.5928
#     epoch 44  loss 1.3589  val P 0.7697  R 0.4851  F1 0.5951
#     epoch 45  loss 1.3285  val P 0.7595  R 0.4965  F1 0.6004
#     epoch 46  loss 1.3061  val P 0.7648  R 0.4858  F1 0.5942
#     epoch 47  loss 1.3052  val P 0.7211  R 0.5220  F1 0.6056
#     epoch 48  loss 1.2925  val P 0.7715  R 0.4901  F1 0.5994
#     epoch 49  loss 1.2557  val P 0.7170  R 0.5418  F1 0.6172
#     epoch 50  loss 1.2490  val P 0.7078  R 0.5198  F1 0.5994
#     epoch 51  loss 1.2509  val P 0.7674  R 0.4837  F1 0.5934
#     epoch 52  loss 1.2227  val P 0.7582  R 0.4773  F1 0.5858
#     epoch 53  loss 1.2215  val P 0.7399  R 0.4915  F1 0.5906
#     epoch 54  loss 1.2153  val P 0.7571  R 0.4901  F1 0.5950
#     early stop at epoch 54 (best F1 0.6180)
#     best val F1 0.6180   TEST F1 0.6019   (3061s, 54 epochs)

# --- seed 5 ---
#     epoch  1  loss 4.0442  val P 0.7896  R 0.1834  F1 0.2977
#     epoch  2  loss 2.3846  val P 0.7606  R 0.3038  F1 0.4342
#     epoch  3  loss 2.2229  val P 0.7789  R 0.2819  F1 0.4139
#     epoch  4  loss 2.1491  val P 0.7730  R 0.3088  F1 0.4413
#     epoch  5  loss 2.0630  val P 0.7635  R 0.3612  F1 0.4904
#     epoch  6  loss 2.0088  val P 0.7721  R 0.3527  F1 0.4842
#     epoch  7  loss 1.9555  val P 0.7771  R 0.3704  F1 0.5017
#     epoch  8  loss 1.9375  val P 0.8239  R 0.3081  F1 0.4485
#     epoch  9  loss 1.9078  val P 0.7841  R 0.3626  F1 0.4959
#     epoch 10  loss 1.8527  val P 0.7860  R 0.3824  F1 0.5145
#     epoch 11  loss 1.8497  val P 0.8136  R 0.3463  F1 0.4858
#     epoch 12  loss 1.8319  val P 0.7929  R 0.3796  F1 0.5134
#     epoch 13  loss 1.8019  val P 0.7988  R 0.3909  F1 0.5250
#     epoch 14  loss 1.7767  val P 0.8239  R 0.3711  F1 0.5117
#     epoch 15  loss 1.7754  val P 0.7964  R 0.4129  F1 0.5438
#     epoch 16  loss 1.7508  val P 0.7459  R 0.4802  F1 0.5842
#     epoch 17  loss 1.7407  val P 0.7965  R 0.3909  F1 0.5245
#     epoch 18  loss 1.7160  val P 0.8147  R 0.3768  F1 0.5153
#     epoch 19  loss 1.7137  val P 0.7809  R 0.4341  F1 0.5580
#     epoch 20  loss 1.6978  val P 0.8382  R 0.3633  F1 0.5069
#     epoch 21  loss 1.6695  val P 0.7642  R 0.4681  F1 0.5806
#     epoch 22  loss 1.6687  val P 0.7749  R 0.4681  F1 0.5837
#     epoch 23  loss 1.6368  val P 0.7629  R 0.4490  F1 0.5653
#     epoch 24  loss 1.6299  val P 0.7829  R 0.4469  F1 0.5690
#     epoch 25  loss 1.6074  val P 0.7605  R 0.4724  F1 0.5828
#     epoch 26  loss 1.6048  val P 0.8022  R 0.4108  F1 0.5433
#     epoch 27  loss 1.5971  val P 0.7950  R 0.4256  F1 0.5544
#     epoch 28  loss 1.5820  val P 0.7731  R 0.4561  F1 0.5737
#     early stop at epoch 28 (best F1 0.5842)
#     best val F1 0.5842   TEST F1 0.5823   (1683s, 28 epochs)

# ========================================================================
# 2. RESULTS OVER SEEDS
# ========================================================================
# F1 is computed per seed and then averaged. We never average precision
# and recall and then combine them - that gives a different number.

# metric                       mean      std
# ------------------------------------------
# test  offensive_precision   0.7299   0.0291
# test  offensive_recall     0.4899   0.0386
# test  offensive_f1         0.5847   0.0195
# val   offensive_f1          0.5917   0.0242

# per-seed test F1: 0.6023, 0.5828, 0.5545, 0.6019, 0.5823
# mean training time: 2128s per seed

# ========================================================================
# 3. DID WE HIT THE TARGET?
# ========================================================================

# our BiLSTM + fastText        0.5847
# published BiLSTM + fastText  0.60     <- the Phase 1 target
# our word-list baseline       0.6521   <- the real floor
# published SinBERT            0.62
# published XLM-R              0.72

# PASS. Reproduction is in range. Phase 1 exit criterion met.

# NOTE: the model at 0.5847 does NOT beat the word list at 0.6521. That is
# expected at Phase 1 - the baseline is a faithful reproduction, not our
# contribution. Phase 2 is where we beat it.

# ========================================================================
# 4. FOR THE README
# ========================================================================

# model                 BiLSTM + fastText + CRF
# hidden size           64
# embeddings            300d fastText, frozen
# class weights         no
# seeds                 [1, 2, 3, 4, 5]
# test offensive F1     0.5847 +/- 0.0195
# test precision        0.7299
# test recall           0.4899
# parameters            8,724,458 total
# training time         2128s per seed on mps