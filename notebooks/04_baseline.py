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
For comparison, XLM-R-large has about 559 million parameters.""")


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
rule("3. COMPARISON WITH PUBLISHED RESULTS")
f1 = agg_test["offensive_f1"]["mean"]
print(f"""
our BiLSTM + fastText        {f1:.4f}
published BiLSTM + fastText  0.60
our lexicon baseline         0.6521
published SinBERT            0.62
published XLM-R              0.72
""")

if 0.57 <= f1 <= 0.63:
    print("Within 0.03 of the published 0.60.")
elif f1 > 0.63:
    print("Above the published number by more than 0.03.")
else:
    print("Below the published number by more than 0.03.")


# ==========================================================================
rule("4. SUMMARY")
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
