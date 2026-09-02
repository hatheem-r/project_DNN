"""
Phase 1 / Step 5b - refit on the FULL training split. Closes Phase 1.

WHY
---
Steps 4 and 5 trained on 6,000 tweets because 1,500 were held out for
validation. The published baseline almost certainly used all 7,500. That is
20% less data and is the most likely remaining source of our 0.015 gap.

The hyperparameters are now chosen, so we rebuild on the full split. This is
the same protocol we used for the word-list baseline in Step 2: choose on
validation, refit on everything, score test once.

FIXED EPOCH BUDGET
------------------
With no validation split left there is nothing to early stop on. So we train
for a fixed number of epochs taken from the validation runs. Best epochs in
Step 5 were 30, 30, 13, 42, 16 - mean 26, median 30. We use 30.

This is legitimate because the budget came from VALIDATION, not from test.

NOTE ON THE VOCABULARY
----------------------
The vocabulary and embedding matrix are rebuilt from the full 7,500 tweets,
so they are larger than the artifacts saved in Step 3. That is correct: the
vocabulary must come from whatever data the model trains on.

Run:  python notebooks/05_full_train_refit.py > results/step5b_refit.txt
"""
import sys, os, csv, argparse, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from data import load_sold
from embeddings import build_vocab, load_vectors_for_vocab, build_embedding_matrix
from dataset import make_loader
from model import BiLSTMTagger
from train import set_seed, evaluate, get_device
from metrics import aggregate_seeds

VEC_PATH = os.environ.get("SOLD_VECTORS", "embeddings/cc.si.300.vec.gz")
RESULTS_CSV = "results/results.csv"


def rule(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=30, help="fixed budget from validation")
p.add_argument("--batch-size", type=int, default=32, help="32 = frozen Phase 1 config")
p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
p.add_argument("--hidden", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--dropout", type=float, default=0.5)
p.add_argument("--no-crf", action="store_true")
p.add_argument("--tag", type=str, default="fulltrain_refit")
args = p.parse_args()
use_crf = not args.no_crf


# ==========================================================================
rule("0. SETUP")
device = get_device()
print(f"device: {device}")

train_full = load_sold("train")
test = load_sold("test")
print(f"train {len(train_full):,} (FULL official split)   test {len(test):,}")
print("No validation split. Fixed epoch budget instead of early stopping.")

vocab, _ = build_vocab(train_full["token_list"], min_freq=1)
print(f"\nvocabulary rebuilt from full train: {len(vocab):,} words")
print("(larger than the 28,456 from train-part - correct, we have more data now)")

if not os.path.exists(VEC_PATH):
    print(f"\nMISSING {VEC_PATH}. See notebooks/03_embeddings.py for the download.")
    sys.exit(1)

print(f"\nloading vectors from {VEC_PATH}")
vectors, dim = load_vectors_for_vocab(VEC_PATH, vocab)
matrix, stats = build_embedding_matrix(vocab, vectors, dim)
print(f"  matrix {matrix.shape}, real vectors for {stats['coverage']:.1%} of rows")

print(f"\nconfig: epochs={args.epochs} (fixed)  batch={args.batch_size}  "
      f"crf={use_crf}  hidden={args.hidden}  lr={args.lr}")
if args.batch_size != 32:
    print("  WARNING: batch size differs from the frozen Phase 1 config (32).")
    print("  Two things change at once (data amount AND batch size), so a score")
    print("  difference cannot be attributed to either. Log it as its own row.")


def model_fn():
    return BiLSTMTagger(matrix, hidden_size=args.hidden, dropout=args.dropout,
                        freeze_embeddings=True, use_crf=use_crf)


params = model_fn().count_parameters()
print("\nparameters:")
for k, v in params.items():
    print(f"  {k:<26} {v:>12,}")


# ==========================================================================
rule("1. TRAIN")
test_loader = make_loader(test, vocab, args.batch_size, shuffle=False)
runs, times = [], []

for seed in args.seeds:
    g = set_seed(seed)
    model = model_fn().to(device)
    loader = make_loader(train_full, vocab, args.batch_size, shuffle=True, generator=g)
    opt = torch.optim.Adam(
        [p_ for p_ in model.parameters() if p_.requires_grad], lr=args.lr
    )

    print(f"\n--- seed {seed} ---")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        total, nb = 0.0, 0
        for ids, labels, mask, lengths, _ in loader:
            ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
            opt.zero_grad()
            loss = model.loss(ids, labels, mask, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
            nb += 1
        if ep % 5 == 0 or ep == args.epochs:
            print(f"    epoch {ep:>2}/{args.epochs}  loss {total/max(nb,1):.4f}")

    secs = time.time() - t0
    s = evaluate(model, test_loader, device)
    runs.append(s)
    times.append(secs)
    print(f"    TEST  P {s['offensive_precision']:.4f}  "
          f"R {s['offensive_recall']:.4f}  F1 {s['offensive_f1']:.4f}   ({secs:.0f}s)")

    os.makedirs("results", exist_ok=True)
    new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["tag", "seed", "crf", "weighted", "frozen_emb", "hidden",
                        "lr", "dropout", "epochs_run", "seconds",
                        "val_f1", "test_p", "test_r", "test_f1"])
        w.writerow([args.tag, seed, use_crf, False, True, args.hidden, args.lr,
                    args.dropout, args.epochs, round(secs, 1), "",
                    round(s["offensive_precision"], 4),
                    round(s["offensive_recall"], 4),
                    round(s["offensive_f1"], 4)])


# ==========================================================================
rule("2. RESULT")
agg = aggregate_seeds(runs)
print(f"{'metric':<26} {'mean':>8} {'std':>8}")
print("-" * 44)
for k in ["offensive_precision", "offensive_recall", "offensive_f1"]:
    print(f"{k:<26} {agg[k]['mean']:>8.4f} {agg[k]['std']:>8.4f}")
print("\nper-seed test F1: " + ", ".join(f"{r['offensive_f1']:.4f}" for r in runs))
print(f"mean time: {statistics.mean(times):.0f}s per seed")

f1 = agg["offensive_f1"]["mean"]
print(f"""
COMPARISON

  full-train refit (7,500)     {f1:.4f}
  train-part only (6,000)      0.5847
  published BiLSTM + fastText  0.60
  our word list                0.6521
  published SinBERT            0.62
  published XLM-R              0.72

  change from more data:       {f1 - 0.5847:+.4f}
""")

if f1 >= 0.5847:
    print("More data helped, as expected. Report this as the final Phase 1 baseline.")
else:
    print("More data did not help. Report it honestly anyway - do not go back and")
    print("tune. A null result here is information, not a failure.")


# ==========================================================================
rule("3. FINAL PHASE 1 BASELINE - FOR THE README")
print(f"""
model                 BiLSTM + fastText{' + CRF' if use_crf else ''}
trained on            {len(train_full):,} tweets (full official train split)
epochs                {args.epochs} fixed, budget taken from validation runs
batch size            {args.batch_size}
vocabulary            {len(vocab):,} words, min_freq=1, from full train
embeddings            300d fastText, frozen, {stats['coverage']:.1%} real vectors
seeds                 {args.seeds}

test precision        {agg['offensive_precision']['mean']:.4f}
test recall           {agg['offensive_recall']['mean']:.4f}
TEST OFFENSIVE F1     {f1:.4f} +/- {agg['offensive_f1']['std']:.4f}

parameters            {params['total']:,} total, {params['trainable']:,} trainable
time                  {statistics.mean(times):.0f}s per seed on {device}

PHASE 2 MUST BEAT: {max(f1, 0.6521):.4f}
""")
