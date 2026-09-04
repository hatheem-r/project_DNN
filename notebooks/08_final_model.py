"""
Phase 2 / Step 8 - THE FINAL MODEL. Full-train refit with the subword channel.

WHAT MAKES THIS THE FINAL MODEL
-------------------------------
Every run so far trained on 6,000 tweets, holding 1,500 back to choose settings.
Those settings are now chosen and frozen, so validation has no job left. We fold
it back in and retrain on all 7,500.

Three things are rebuilt from the FULL train split, not train-part:
  - the SentencePiece tokenizer
  - the word vocabulary
  - the fastText embedding matrix
The vocabulary must come from whatever data the model actually trains on.

FIXED EPOCH BUDGET
------------------
With no validation split there is nothing to early stop on. We train for a fixed
number of epochs taken from the validation runs. bpe_1000 stopped best at epochs
30, 30, 41, 35, 37 - median 35. That budget came from VALIDATION, never test.

TWO MODELS ARE REPORTED
-----------------------
  full     word channel + subword channel   best F1
  minimal  subword channel only             no pretrained resources at all,
                                            176k trainable, -0.0012 val F1

Run:
    python notebooks/08_final_model.py --both
    python notebooks/08_final_model.py --model full
    python notebooks/08_final_model.py --model minimal
"""
import sys, os, csv, argparse, time, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from data import load_sold
from embeddings import build_vocab, load_vectors_for_vocab, build_embedding_matrix
from dataset import make_loader
from model import BiLSTMTagger
from subword import train_sentencepiece, load_sentencepiece
from train import set_seed, evaluate, get_device
from metrics import aggregate_seeds

VEC_PATH = os.environ.get("SOLD_VECTORS", "embeddings/cc.si.300.vec.gz")
RESULTS_CSV = "results/results_final.csv"
OUT_DIR = "artifacts"

# Frozen after Step 7b. Do not change without re-running the validation sweep.
SP_TYPE = "bpe"
SP_VOCAB = 1000
PIECE_DIM = 50
SUBWORD_DIM = 100
POOLING = "bilstm"
HIDDEN = 64
DROPOUT = 0.5
LR = 1e-3
BATCH = 32
USE_CRF = True
FIXED_EPOCHS = 35          # median best epoch from the bpe_1000 validation runs

BASELINE = 0.5965          # Phase 1 full-train refit, word only
BASELINE_STD = 0.0103
WORDLIST = 0.6521


def rule(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


p = argparse.ArgumentParser()
p.add_argument("--model", choices=["full", "minimal"], default="full")
p.add_argument("--both", action="store_true", help="run full then minimal")
p.add_argument("--epochs", type=int, default=FIXED_EPOCHS)
p.add_argument("--batch-size", type=int, default=BATCH)
p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
p.add_argument("--pooling", choices=["bilstm", "mean"], default=POOLING)
p.add_argument("--no-crf", action="store_true")
args = p.parse_args()
use_crf = not args.no_crf


# ==========================================================================
rule("0. SETUP")
device = get_device()
print(f"device {device}")

train_full = load_sold("train")
test = load_sold("test")
print(f"train {len(train_full):,} (FULL official split)   test {len(test):,}")
print("No validation split. Fixed epoch budget instead of early stopping.")

# ---- tokenizer, rebuilt on full train
prefix = f"{OUT_DIR}/sp_final_{SP_TYPE}_{SP_VOCAB}_fulltrain"
os.makedirs(OUT_DIR, exist_ok=True)
if not os.path.exists(prefix + ".model"):
    print(f"\ntraining {SP_TYPE} tokenizer, vocab {SP_VOCAB}, on the FULL train split")
    train_sentencepiece([" ".join(t) for t in train_full["token_list"]],
                        SP_VOCAB, prefix, model_type=SP_TYPE)
sp = load_sentencepiece(prefix + ".model")
n_pieces = sp.get_piece_size()
print(f"tokenizer: {n_pieces:,} pieces")

# ---- word vocabulary and embedding matrix, rebuilt on full train
vocab, _ = build_vocab(train_full["token_list"], min_freq=1)
print(f"word vocabulary: {len(vocab):,} (was 28,456 from train-part)")

if not os.path.exists(VEC_PATH):
    print(f"\nMISSING {VEC_PATH}. See notebooks/03_embeddings.py for the download.")
    sys.exit(1)
vectors, dim = load_vectors_for_vocab(VEC_PATH, vocab)
matrix, emb_stats = build_embedding_matrix(vocab, vectors, dim)
print(f"embedding matrix {matrix.shape}, real vectors for {emb_stats['coverage']:.1%}")

print(f"""
frozen configuration
  tokenizer       {SP_TYPE} {SP_VOCAB} ({n_pieces:,} pieces), trained on full train
  piece_dim       {PIECE_DIM}
  subword_dim     {SUBWORD_DIM}
  pooling         {args.pooling}
  hidden          {HIDDEN}
  dropout         {DROPOUT}
  lr              {LR}
  batch           {args.batch_size}
  crf             {use_crf}
  epochs          {args.epochs} FIXED, no early stopping
  seeds           {args.seeds}""")


def build(use_word_channel):
    return lambda: BiLSTMTagger(
        matrix, hidden_size=HIDDEN, dropout=DROPOUT, freeze_embeddings=True,
        use_crf=use_crf, n_pieces=n_pieces, piece_dim=PIECE_DIM,
        subword_dim=SUBWORD_DIM, subword_pooling=args.pooling,
        use_word_channel=use_word_channel,
    )


def log(tag, seed, s, secs, params):
    os.makedirs("results", exist_ok=True)
    fields = ["tag", "sp", "n_pieces", "pooling", "word_channel", "crf", "batch",
              "epochs", "seed", "seconds", "trainable", "total",
              "test_p", "test_r", "test_f1"]
    new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({
            "tag": tag, "sp": f"{SP_TYPE}_{SP_VOCAB}", "n_pieces": n_pieces,
            "pooling": args.pooling, "word_channel": tag == "full",
            "crf": use_crf, "batch": args.batch_size, "epochs": args.epochs,
            "seed": seed, "seconds": round(secs, 1),
            "trainable": params["trainable"], "total": params["total"],
            "test_p": round(s["offensive_precision"], 4),
            "test_r": round(s["offensive_recall"], 4),
            "test_f1": round(s["offensive_f1"], 4),
        })


def train_and_score(name, use_word_channel):
    model_fn = build(use_word_channel)
    params = model_fn().count_parameters()

    rule(f"MODEL: {name.upper()}   "
         f"({'word + subword' if use_word_channel else 'SUBWORD ONLY, no pretrained resources'})")
    print("parameters:")
    for k, v in params.items():
        print(f"  {k:<26} {v:>12,}")
    if not use_word_channel:
        print("\n  The fastText matrix is not used at all in this model. No 600 MB")
        print("  vector file, no frozen embedding table. Fully self-contained.")

    test_loader = make_loader(test, vocab, args.batch_size, shuffle=False, sp=sp)
    runs, times = [], []

    for seed in args.seeds:
        g = set_seed(seed)
        model = model_fn().to(device)
        loader = make_loader(train_full, vocab, args.batch_size, shuffle=True,
                             generator=g, sp=sp)
        opt = torch.optim.Adam(
            [q for q in model.parameters() if q.requires_grad], lr=LR)

        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            model.train()
            total, nb = 0.0, 0
            for ids, labels, mask, lengths, _, pid, plen in loader:
                ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
                if pid is not None:
                    pid, plen = pid.to(device), plen.to(device)
                opt.zero_grad()
                loss = model.loss(ids, labels, mask, lengths,
                                  piece_ids=pid, piece_lens=plen)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                total += loss.item(); nb += 1
            if ep % 10 == 0 or ep == args.epochs:
                print(f"    epoch {ep:>2}/{args.epochs}  loss {total/max(nb,1):.4f}")

        secs = time.time() - t0
        s = evaluate(model, test_loader, device)
        runs.append(s); times.append(secs)
        print(f"    seed {seed}  TEST  P {s['offensive_precision']:.4f}  "
              f"R {s['offensive_recall']:.4f}  F1 {s['offensive_f1']:.4f}   ({secs:.0f}s)")
        log(name, seed, s, secs, params)

    agg = aggregate_seeds(runs)
    print(f"\n  TEST F1 {agg['offensive_f1']['mean']:.4f} "
          f"+/- {agg['offensive_f1']['std']:.4f}")
    print("  per-seed: " + ", ".join(f"{r['offensive_f1']:.4f}" for r in runs))
    print(f"  mean time {statistics.mean(times):.0f}s per seed")
    return agg, params


# ==========================================================================
rule("1. TRAIN")
results = {}
todo = [("full", True), ("minimal", False)] if args.both else \
       [(args.model, args.model == "full")]
for name, wc in todo:
    results[name] = train_and_score(name, wc)


# ==========================================================================
rule("2. FINAL RESULTS")
print(f"""{'model':<34} {'F1':>8} {'std':>8} {'P':>8} {'R':>8} {'trainable':>12}
{'-' * 82}""")
print(f"{'Published BiLSTM + CBOW':<34} {0.58:>8.4f} {'':>8} {'':>8} {'':>8} {'':>12}")
print(f"{'Published BiLSTM + fastText':<34} {0.60:>8.4f} {'':>8} {'0.74':>8} {'0.48':>8} {'':>12}")
print(f"{'Our Phase 1 baseline':<34} {BASELINE:>8.4f} {BASELINE_STD:>8.4f} "
      f"{0.7452:>8.4f} {0.4979:>8.4f} {187658:>12,}")
print(f"{'Published SinBERT':<34} {0.62:>8.4f} {'':>8} {'':>8} {'':>8} {'~110M':>12}")
print(f"{'Our word list':<34} {WORDLIST:>8.4f} {'':>8} {0.6361:>8.4f} {0.6689:>8.4f} {0:>12,}")
print(f"{'Published XLM-T':<34} {0.70:>8.4f} {'':>8} {'0.64':>8} {'0.77':>8} {'~270M':>12}")
for name in ("full", "minimal"):
    if name in results:
        a, pr = results[name]
        label = "OURS: word + subword" if name == "full" else "OURS: subword only (no PLM/fastText)"
        print(f"{label:<34} {a['offensive_f1']['mean']:>8.4f} "
              f"{a['offensive_f1']['std']:>8.4f} "
              f"{a['offensive_precision']['mean']:>8.4f} "
              f"{a['offensive_recall']['mean']:>8.4f} {pr['trainable']:>12,}")
print(f"{'Published XLM-R':<34} {0.72:>8.4f} {'':>8} {'0.68':>8} {'0.76':>8} {'~560M':>12}")
print(f"{'Published XLM-R + TSD transfer':<34} {0.73:>8.4f} {'':>8} {'':>8} {'':>8} {'~560M':>12}")

if "full" in results:
    a, pr = results["full"]
    f1 = a["offensive_f1"]["mean"]
    print(f"""
CHANGE FROM THE PHASE 1 BASELINE
  {BASELINE:.4f} -> {f1:.4f}   ({f1 - BASELINE:+.4f})
  precision 0.7452 -> {a['offensive_precision']['mean']:.4f}  ({a['offensive_precision']['mean'] - 0.7452:+.4f})
  recall    0.4979 -> {a['offensive_recall']['mean']:.4f}  ({a['offensive_recall']['mean'] - 0.4979:+.4f})
""")

    dp = a["offensive_precision"]["mean"] - 0.7452
    dr = a["offensive_recall"]["mean"] - 0.4979
    if f1 < BASELINE:
        print("""  WORSE THAN THE BASELINE. Something is wrong - check that the tokenizer,
  vocabulary and embedding matrix were all rebuilt on the FULL train split,
  and that the epoch budget suits the larger dataset.""")
    elif dr > 0.05 and abs(dp) < 0.05:
        print("""  The gain is recall-driven at near-constant precision. That is the mechanism
  predicted from the measured 48.7% unseen-type rate, not a threshold shift.
  Say this in the paper - the mechanism is what makes the number credible.""")
    elif dr > 0.05 and dp < -0.05:
        print(f"""  Recall rose {dr:+.4f} but precision fell {dp:+.4f}. The model is firing more
  freely rather than seeing more. Report both columns and do NOT claim the
  unseen-morphology mechanism without further evidence.""")
    else:
        print("""  The precision/recall pattern differs from the 6,000-tweet runs. Report what
  the numbers actually show rather than the expected story.""")

    print(f"""EFFICIENCY
  ours {pr['trainable']:,} trainable vs XLM-R-large ~560,000,000
  = {560_000_000 / pr['trainable']:,.0f}x fewer trainable parameters""")

if "minimal" in results:
    a, pr = results["minimal"]
    print(f"""
MINIMAL MODEL
  {a['offensive_f1']['mean']:.4f} +/- {a['offensive_f1']['std']:.4f} with {pr['trainable']:,} trainable parameters
  and NO pretrained embeddings of any kind - no fastText, no PLM.
  = {560_000_000 / pr['trainable']:,.0f}x fewer trainable parameters than XLM-R-large""")

print(f"\nrows written to {RESULTS_CSV}")
print("""
WRITE IT UP HONESTLY. The headline is not "we nearly match XLM-R". It is:
a lightweight model with a subword channel reaches this F1 at a small fraction
of the parameters, and the gain is precision-neutral and recall-driven.
The mechanism is what makes it credible.""")
