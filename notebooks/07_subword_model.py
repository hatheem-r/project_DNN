"""
Phase 2 / Piece 1 / Step 7 - the subword model.

Adds a SECOND input channel to the frozen Phase 1 baseline. Nothing is removed.

    word "සක්කිලියා"
       |
       +----------------------------+
       |                            |
    word id lookup            split into pieces
       |                            |
    300 fastText numbers      pieces -> vectors -> pool to ONE vector
    (frozen)                        |
       |                       100 numbers (learned)
       +------------ concat --------+
                     |
              400 numbers per word
                     |
             BiLSTM(64) -> Linear -> CRF     <- unchanged from Phase 1

WHY CONCATENATE RATHER THAN REPLACE
-----------------------------------
80.8% of vocabulary words have a real fastText vector and the word channel
already works for them. For the rest it is a random vector - pure noise - and
the subword channel carries the signal instead. Keeping both gives the model a
fallback rather than throwing away what works.

BEFORE RUNNING
--------------
    python tests/test_subword_alignment.py      <- must pass first
    python notebooks/06_subword_tokenizer.py    <- builds artifacts/sp_*.model

Run:
    # fast sweep on validation (batch 128, no CRF)
    python notebooks/07_subword_model.py --sweep > results/step7_sweep.txt

    # winner, full settings, 5 seeds, test scored once
    python notebooks/07_subword_model.py --final --sp bpe_8000 > results/step7_final.txt
"""
import sys, os, csv, argparse, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from data import load_sold, train_val_split
from embeddings import build_vocab
from dataset import make_loader
from model import BiLSTMTagger
from subword import load_sentencepiece
from train import train_one_seed, evaluate, get_device
from metrics import aggregate_seeds

MATRIX = "artifacts/embedding_matrix.npy"
RESULTS_CSV = "results/results.csv"
BASELINE_F1 = 0.5965       # Phase 1, full-train refit
BASELINE_STD = 0.0103
WORDLIST_F1 = 0.6521

# The four settings chosen in Step 6. Three vary size with the algorithm held
# constant; the fourth is size-matched against bpe_8000 to isolate the algorithm
# and test whether breaking Sinhala graphemes actually costs F1.
SWEEP = [
    ("bpe_24000",     "84.7% words whole - near word level"),
    ("bpe_8000",      "73.5% words whole - middle"),
    ("bpe_2000",      "55.3% words whole - real decomposition"),
    ("unigram_8000",  "76.0% words whole - CONTROL vs bpe_8000, grapheme effect"),
]


def rule(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


p = argparse.ArgumentParser()
p.add_argument("--sweep", action="store_true", help="fast validation sweep")
p.add_argument("--final", action="store_true", help="full run, scores test")
p.add_argument("--sp", type=str, default=None, help="e.g. bpe_8000")
p.add_argument("--pooling", type=str, default="bilstm", choices=["bilstm", "mean"])
p.add_argument("--piece-dim", type=int, default=50)
p.add_argument("--subword-dim", type=int, default=100)
p.add_argument("--no-word-channel", action="store_true", help="ablation: subword only")
p.add_argument("--seeds", type=int, nargs="+", default=None)
p.add_argument("--epochs", type=int, default=None)
p.add_argument("--patience", type=int, default=None)
p.add_argument("--batch-size", type=int, default=None)
p.add_argument("--no-crf", action="store_true")
args = p.parse_args()

if not (args.sweep or args.final):
    print("Pass --sweep or --final. See the docstring.")
    sys.exit(1)

# Sweep = cheap settings (Step 0 found CRF costs 6-8x). Final = frozen Phase 1
# settings so the number is comparable to 0.5965.
if args.sweep:
    SEEDS = args.seeds or [1, 2, 3]
    BATCH = args.batch_size or 128
    USE_CRF = False
    EPOCHS = args.epochs or 60
    PATIENCE = args.patience or 12
else:
    SEEDS = args.seeds or [1, 2, 3, 4, 5]
    BATCH = args.batch_size or 32
    USE_CRF = not args.no_crf
    EPOCHS = args.epochs or 60
    PATIENCE = args.patience or 12


# ==========================================================================
rule("0. SETUP")
device = get_device()
print(f"device {device}   mode {'SWEEP (validation only)' if args.sweep else 'FINAL (scores test)'}")
print(f"seeds {SEEDS}   batch {BATCH}   crf {USE_CRF}   epochs<={EPOCHS} patience {PATIENCE}")
if args.sweep:
    print("\nSweep uses batch 128 and no CRF for speed (Step 0: CRF costs 6-8x).")
    print("Rankings only. Every reported number comes from a --final run.")

train_full = load_sold("train")
test = load_sold("test")
train_part, val = train_val_split(train_full)
vocab, _ = build_vocab(train_part["token_list"], min_freq=1)
matrix = np.load(MATRIX)
print(f"\ntrain-part {len(train_part):,}  val {len(val):,}  test {len(test):,}")
print(f"vocab {len(vocab):,}   embedding matrix {matrix.shape}")
if matrix.shape[0] != len(vocab):
    print("MISMATCH between matrix rows and vocab. Re-run notebooks/03_embeddings.py.")
    sys.exit(1)


def build(sp, n_pieces):
    return lambda: BiLSTMTagger(
        matrix, hidden_size=64, dropout=0.5, freeze_embeddings=True,
        use_crf=USE_CRF, n_pieces=n_pieces, piece_dim=args.piece_dim,
        subword_dim=args.subword_dim, subword_pooling=args.pooling,
        use_word_channel=not args.no_word_channel,
    )


def log_row(tag, seed, val_f1, s, secs, epochs):
    os.makedirs("results", exist_ok=True)
    new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        wtr = csv.writer(fh)
        if new:
            wtr.writerow(["tag", "seed", "crf", "weighted", "frozen_emb", "hidden",
                          "lr", "dropout", "epochs_run", "seconds",
                          "val_f1", "test_p", "test_r", "test_f1"])
        wtr.writerow([tag, seed, USE_CRF, False, True, 64, 1e-3, 0.5, epochs,
                      round(secs, 1), round(val_f1, 4),
                      round(s["offensive_precision"], 4) if s else "",
                      round(s["offensive_recall"], 4) if s else "",
                      round(s["offensive_f1"], 4) if s else ""])


def run(sp_name, note=""):
    """Train one configuration over all seeds. Returns (val_agg, test_agg)."""
    path = f"artifacts/sp_{sp_name}.model"
    if not os.path.exists(path):
        print(f"  MISSING {path} - run notebooks/06_subword_tokenizer.py first")
        return None, None
    sp = load_sentencepiece(path)
    n_pieces = sp.get_piece_size()

    model_fn = build(sp, n_pieces)
    params = model_fn().count_parameters()
    print(f"\n--- {sp_name}  ({n_pieces:,} pieces)  {note}")
    print(f"    trainable {params['trainable']:,}  "
          f"(subword channel {params['subword_channel']:,}, "
          f"lstm input {params['lstm_input_dim']})")

    val_runs, test_runs, times = [], [], []
    for seed in SEEDS:
        model, hist, best_val, secs = train_one_seed(
            model_fn, train_part, val, vocab, seed,
            batch_size=BATCH, max_epochs=EPOCHS, patience=PATIENCE,
            device=device, sp=sp, verbose=False,
        )
        val_runs.append(best_val)
        times.append(secs)
        line = f"    seed {seed}  val F1 {best_val['offensive_f1']:.4f}"

        test_s = None
        if args.final:
            test_loader = make_loader(test, vocab, BATCH, shuffle=False, sp=sp)
            test_s = evaluate(model, test_loader, device)
            test_runs.append(test_s)
            line += f"   TEST F1 {test_s['offensive_f1']:.4f}"
        print(line + f"   ({secs:.0f}s, {len(hist)} epochs)")
        log_row(f"subword_{sp_name}_{args.pooling}", seed,
                best_val["offensive_f1"], test_s, secs, len(hist))

    va = aggregate_seeds(val_runs)
    ta = aggregate_seeds(test_runs) if test_runs else None
    print(f"    VAL  F1 {va['offensive_f1']['mean']:.4f} +/- {va['offensive_f1']['std']:.4f}"
          f"   ({statistics.mean(times):.0f}s/seed)")
    if ta:
        print(f"    TEST F1 {ta['offensive_f1']['mean']:.4f} +/- {ta['offensive_f1']['std']:.4f}"
              f"   P {ta['offensive_precision']['mean']:.4f}"
              f"  R {ta['offensive_recall']['mean']:.4f}")
    return va, ta


# ==========================================================================
if args.sweep:
    rule("1. SWEEP - four tokenizer settings on VALIDATION")
    print("Test is NOT touched. These numbers choose a setting, nothing more.")
    results = {}
    for name, note in SWEEP:
        va, _ = run(name, note)
        if va:
            results[name] = va["offensive_f1"]

    rule("2. RANKING")
    if not results:
        print("Nothing ran. Run notebooks/06_subword_tokenizer.py first.")
        sys.exit(1)
    print(f"{'setting':<16} {'val F1':>9} {'std':>8}")
    print("-" * 35)
    for name, m in sorted(results.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"{name:<16} {m['mean']:>9.4f} {m['std']:>8.4f}")

    best = max(results.items(), key=lambda kv: kv[1]["mean"])
    print(f"\nBEST ON VALIDATION: {best[0]}  ({best[1]['mean']:.4f})")

    if "bpe_8000" in results and "unigram_8000" in results:
        d = results["bpe_8000"]["mean"] - results["unigram_8000"]["mean"]
        pooled = (results["bpe_8000"]["std"] + results["unigram_8000"]["std"]) / 2
        print(f"""
GRAPHEME CONTROL (bpe_8000 vs unigram_8000, same vocabulary size)
  difference {d:+.4f}, pooled std {pooled:.4f}""")
        if abs(d) <= max(pooled, 1e-6):
            print("  Inside the noise. Step 6's grapheme finding does not translate")
            print("  into an F1 difference here. Report it as a tokenizer-quality")
            print("  observation, NOT as an accuracy claim.")
        elif d > 0:
            print("  BPE wins. The grapheme finding translates into accuracy.")
            print("  This is a real result and belongs in the paper.")
        else:
            print("  Unigram wins despite breaking more graphemes. Report honestly -")
            print("  root-sharing evidently matters more than grapheme integrity here.")

    print(f"""
NEXT: run the winner with the frozen Phase 1 settings and score test once.

    python notebooks/07_subword_model.py --final --sp {best[0]}
""")

# ==========================================================================
else:
    if not args.sp:
        print("--final needs --sp, e.g. --sp bpe_8000")
        sys.exit(1)
    rule(f"1. FINAL RUN - {args.sp}")
    print("Frozen Phase 1 settings (batch 32, CRF on) so the number is directly")
    print(f"comparable to the baseline {BASELINE_F1:.4f}.")
    va, ta = run(args.sp)
    if ta is None:
        sys.exit(1)

    rule("2. RESULT")
    f1 = ta["offensive_f1"]["mean"]
    std = ta["offensive_f1"]["std"]
    delta = f1 - BASELINE_F1
    noise = max(std, BASELINE_STD)

    print(f"""
  Phase 1 baseline (word only)  {BASELINE_F1:.4f} +/- {BASELINE_STD:.4f}
  + subword channel             {f1:.4f} +/- {std:.4f}
  change                        {delta:+.4f}

  our word list                 {WORDLIST_F1:.4f}   <- still the floor to beat
  published XLM-R               0.7200

  precision {ta['offensive_precision']['mean']:.4f}   recall {ta['offensive_recall']['mean']:.4f}
  Phase 1 was precision 0.7452, recall 0.4979.""")

    if delta > 2 * noise:
        print(f"\n  CLEAR GAIN. {delta:+.4f} is more than twice the noise ({noise:.4f}).")
    elif delta > noise:
        print(f"\n  MODEST GAIN. {delta:+.4f} exceeds the noise ({noise:.4f}) but not by much.")
        print("  Report with the standard deviation and do not overstate it.")
    elif delta > -noise:
        print(f"\n  NO MEASURABLE EFFECT. {delta:+.4f} is inside the noise ({noise:.4f}).")
        print("  Report it as such. A careful null result on a low-resource")
        print("  language is a legitimate finding, and hiding it is not an option.")
    else:
        print(f"\n  IT HURT. {delta:+.4f}. Report honestly, then investigate:")
        print("    - did recall rise while precision collapsed?")
        print("    - is the subword channel too large relative to 6,000 tweets?")
        print("    - try --pooling mean, or a smaller --subword-dim")

    print(f"""
  ABLATION ROWS STILL OWED
    --pooling mean            does piece ORDER matter?
    --no-word-channel         is fastText still needed?
  Both at 5 seeds, same settings.
""")
