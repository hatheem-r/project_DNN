"""
Phase 2, Pieces 2, 3 and 4 - one script, three experiments.

    --piece2   joint sentence + token head        (sweeps lambda)
    --piece3   balanced loss functions            (CE / weighted / focal / dice)
    --piece4   offline distillation from SemiSOLD (needs Piece 2)
    --final    the winning combination, full-train refit, test scored once

Everything is chosen on VALIDATION. Test is scored only by --final.

Base model, frozen after Piece 1:
    bpe_1000 subword channel, piece_dim 50, subword_dim 100, BiLSTM pooling,
    hidden 64, dropout 0.5, Adam lr 1e-3, batch 32.

Run:
    python notebooks/09_pieces_234.py --piece3
    python notebooks/09_pieces_234.py --piece2
    python notebooks/09_pieces_234.py --piece4
    python notebooks/09_pieces_234.py --final --lambda 0.3 --loss focal
"""
import sys, os, csv, argparse, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from data import load_sold, train_val_split
from embeddings import build_vocab, load_vectors_for_vocab, build_embedding_matrix
from dataset import make_loader
from model import BiLSTMTagger
from subword import train_sentencepiece, load_sentencepiece
from losses import TokenLoss, inverse_frequency_weights
from train import train_one_seed, evaluate, get_device
from metrics import aggregate_seeds

VEC_PATH = os.environ.get("SOLD_VECTORS", "embeddings/cc.si.300.vec.gz")
# CSV path is per-piece so three people can run in parallel and commit
# without colliding. Set below once the mode is known.
CSV_PATH = None
SP_TYPE, SP_VOCAB = "bpe", 1000
PIECE1_VAL = 0.7083     # bpe_1000 both channels, validation
PIECE1_TEST = 0.7002    # full-train refit, both channels
BASELINE = 0.5965


def rule(t):
    print("\n" + "=" * 72); print(t); print("=" * 72)


p = argparse.ArgumentParser()
p.add_argument("--piece2", action="store_true")
p.add_argument("--piece3", action="store_true")
p.add_argument("--piece4", action="store_true")
p.add_argument("--final", action="store_true")
p.add_argument("--lambda", dest="lam", type=float, default=0.0)
p.add_argument("--loss", default="cross_entropy",
               choices=["cross_entropy", "weighted", "focal", "dice"])
p.add_argument("--gamma", type=float, default=2.0)
p.add_argument("--alpha", type=float, default=0.75)
p.add_argument("--distill-weight", type=float, default=1.0)
p.add_argument("--uncertainty", type=float, default=0.1)
p.add_argument("--minimal", action="store_true", help="subword channel only")
p.add_argument("--seeds", type=int, nargs="+", default=None)
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--patience", type=int, default=12)
p.add_argument("--batch-size", type=int, default=None)
p.add_argument("--crf", action="store_true", help="force CRF on (off by default)")
p.add_argument("--owner", type=str, default=os.environ.get("OWNER", "unknown"),
               help="your name - recorded in every row so results are traceable")
args = p.parse_args()

if not any([args.piece2, args.piece3, args.piece4, args.final]):
    print("Pass one of --piece2 --piece3 --piece4 --final"); sys.exit(1)

PIECE = ("final" if args.final else
         "piece2" if args.piece2 else
         "piece3" if args.piece3 else "piece4")
CSV_PATH = f"results/results_{PIECE}.csv"

SEEDS = args.seeds or ([1, 2, 3, 4, 5] if args.final else [1, 2, 3])
BATCH = args.batch_size or (32 if args.final else 64)
# The CRF cannot take per-class weights, and Step 7b showed it is now redundant
# (0.7064 without vs 0.7043 with, inside noise). So these experiments run
# without it. Evidence-backed, not convenience.
USE_CRF = args.crf


# ==========================================================================
rule("0. SETUP")
device = get_device()
print(f"experiment: {PIECE.upper()}   owner: {args.owner}")
print(f"device {device}   seeds {SEEDS}   batch {BATCH}   crf {USE_CRF}")
if args.owner == "unknown":
    print("\nWARNING: no --owner given. Pass --owner YourName so the rows in")
    print("the results file say who ran them.")

train_full = load_sold("train")
test = load_sold("test")
train_part, val = train_val_split(train_full)
fit_df = train_full if args.final else train_part
print(f"train-part {len(train_part):,}  val {len(val):,}  test {len(test):,}")
print(f"fitting on {len(fit_df):,} tweets ({'FULL train' if args.final else 'train-part'})")

sp_prefix = f"artifacts/sp_{'final_' if args.final else ''}{SP_TYPE}_{SP_VOCAB}" + \
            ("_fulltrain" if args.final else "")
if not os.path.exists(sp_prefix + ".model"):
    train_sentencepiece([" ".join(t) for t in fit_df["token_list"]],
                        SP_VOCAB, sp_prefix, model_type=SP_TYPE)
sp = load_sentencepiece(sp_prefix + ".model")
n_pieces = sp.get_piece_size()

vocab, _ = build_vocab(fit_df["token_list"], min_freq=1)
mpath = "artifacts/embedding_matrix.npy"
if args.final or not os.path.exists(mpath) or np.load(mpath).shape[0] != len(vocab):
    vectors, dim = load_vectors_for_vocab(VEC_PATH, vocab)
    matrix, _ = build_embedding_matrix(vocab, vectors, dim)
else:
    matrix = np.load(mpath)
print(f"tokenizer {n_pieces:,} pieces   vocab {len(vocab):,}   matrix {matrix.shape}")

class_weights = inverse_frequency_weights(fit_df, device)
print(f"inverse-frequency class weights: "
      f"[{float(class_weights[0]):.3f}, {float(class_weights[1]):.3f}]")


def build(loss_kind="cross_entropy", sentence_head=False):
    def fn():
        tl = None if USE_CRF else TokenLoss(
            loss_kind, weight=class_weights.cpu(), gamma=args.gamma, alpha=args.alpha)
        return BiLSTMTagger(
            matrix, hidden_size=64, dropout=0.5, freeze_embeddings=True,
            use_crf=USE_CRF, n_pieces=n_pieces, piece_dim=50, subword_dim=100,
            subword_pooling="bilstm", use_word_channel=not args.minimal,
            sentence_head=sentence_head, token_loss=tl,
        )
    return fn


def log(tag, seed, val_f1, s, secs, params):
    os.makedirs("results", exist_ok=True)
    fields = ["piece", "owner", "tag", "loss", "lambda", "distill", "uncertainty",
              "crf", "minimal", "batch", "seed", "seconds", "trainable", "val_f1",
              "test_p", "test_r", "test_f1"]
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({"piece": PIECE, "owner": args.owner,
                    "tag": tag, "loss": args.loss, "lambda": args.lam,
                    "distill": args.distill_weight if args.piece4 or args.final else "",
                    "uncertainty": args.uncertainty if args.piece4 or args.final else "",
                    "crf": USE_CRF, "minimal": args.minimal, "batch": BATCH,
                    "seed": seed, "seconds": round(secs, 1),
                    "trainable": params["trainable"], "val_f1": round(val_f1, 4),
                    "test_p": round(s["offensive_precision"], 4) if s else "",
                    "test_r": round(s["offensive_recall"], 4) if s else "",
                    "test_f1": round(s["offensive_f1"], 4) if s else ""})


def run(tag, loss_kind, lam, distill_loader=None, score_test=False):
    fn = build(loss_kind, sentence_head=(lam > 0 or distill_loader is not None))
    params = fn().count_parameters()
    print(f"\n--- {tag}   trainable {params['trainable']:,}"
          f"{'  (sentence head ' + format(params['sentence_head'], ',') + ')' if params['sentence_head'] else ''}")

    vals, tests, times = [], [], []
    for seed in SEEDS:
        model, hist, best_val, secs = train_one_seed(
            fn, fit_df, val, vocab, seed, batch_size=BATCH,
            max_epochs=args.epochs, patience=args.patience, device=device, sp=sp,
            verbose=False, sentence_lambda=lam, distill_loader=distill_loader,
            distill_weight=args.distill_weight,
        )
        vals.append(best_val); times.append(secs)
        line = f"    seed {seed}  val F1 {best_val['offensive_f1']:.4f}"
        ts = None
        if score_test:
            ts = evaluate(model, make_loader(test, vocab, BATCH, False, sp=sp), device)
            tests.append(ts)
            line += f"   TEST {ts['offensive_f1']:.4f}"
        print(line + f"   ({secs:.0f}s, {len(hist)} ep)")
        log(tag, seed, best_val["offensive_f1"], ts, secs, params)

    va = aggregate_seeds(vals)
    print(f"    VAL  {va['offensive_f1']['mean']:.4f} +/- {va['offensive_f1']['std']:.4f}"
          f"   P {va['offensive_precision']['mean']:.4f}"
          f"  R {va['offensive_recall']['mean']:.4f}   ({statistics.mean(times):.0f}s/seed)")
    ta = aggregate_seeds(tests) if tests else None
    if ta:
        print(f"    TEST {ta['offensive_f1']['mean']:.4f} +/- {ta['offensive_f1']['std']:.4f}"
              f"   P {ta['offensive_precision']['mean']:.4f}"
              f"  R {ta['offensive_recall']['mean']:.4f}")
    return va, ta


def verdict(name, got, ref, noise):
    d = got - ref
    noise = max(noise, 1e-4)   # zero std would make an exact tie read as "hurt"
    if d > 2 * noise:
        return f"{name}: {d:+.4f}  CLEAR GAIN"
    if d > noise:
        return f"{name}: {d:+.4f}  modest gain, report with the std"
    if d > -noise:
        return f"{name}: {d:+.4f}  NO MEASURABLE EFFECT - report the null"
    return f"{name}: {d:+.4f}  HURT - report honestly"


# ==========================================================================
if args.piece3:
    rule("PIECE 3 - BALANCED LOSS FUNCTIONS")
    print("""Only 4.14% of training tokens are offensive, so plain cross-entropy is
dominated by the easy negative class.

WARNING BEFORE YOU READ THE NUMBERS. After Piece 1 the model sits near
precision 0.74 / recall 0.68 - close to balanced. These losses all push toward
recall, so there is far less headroom than at Phase 1's 0.75/0.50. Pushing too
hard trades precision for recall at a NET LOSS. Read both columns.

The CRF is off: it computes its own sequence likelihood and cannot take
per-class weights. Step 7b showed it is now redundant, so this is
evidence-backed rather than convenient.""")
    res = {}
    for k in ["cross_entropy", "weighted", "focal", "dice"]:
        args.loss = k
        va, _ = run(f"piece3_{k}", k, 0.0)
        res[k] = va
    rule("PIECE 3 RANKING")
    print(f"{'loss':<16} {'val F1':>9} {'std':>8} {'P':>9} {'R':>9}")
    print("-" * 56)
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["offensive_f1"]["mean"]):
        print(f"{k:<16} {v['offensive_f1']['mean']:>9.4f} {v['offensive_f1']['std']:>8.4f} "
              f"{v['offensive_precision']['mean']:>9.4f} {v['offensive_recall']['mean']:>9.4f}")
    ce = res["cross_entropy"]["offensive_f1"]
    best = max(res.items(), key=lambda kv: kv[1]["offensive_f1"]["mean"])
    print("\n" + verdict(f"best ({best[0]}) vs cross_entropy",
                         best[1]["offensive_f1"]["mean"], ce["mean"],
                         max(ce["std"], best[1]["offensive_f1"]["std"])))
    print("""
If the answer is null, that is a real finding and pairs with the CRF result:
once the input representation is fixed, the downstream corrections stop
mattering. Report it, do not bury it.""")

# ==========================================================================
elif args.piece2:
    rule("PIECE 2 - JOINT SENTENCE + TOKEN HEAD")
    print("""A second output on the SAME encoder: is the whole tweet offensive?
Two views of one phenomenon, so forcing one BiLSTM to serve both regularises it.

loss = token_loss + lambda * sentence_loss

lambda 0 is the Piece 1 model exactly. This head is also the docking port for
Piece 4 - SemiSOLD's teacher scores are sentence level, so without it
distillation has nothing to attach to.""")
    res = {}
    for lam in [0.0, 0.1, 0.3, 0.5, 1.0]:
        va, _ = run(f"piece2_lam{lam}", args.loss, lam)
        res[lam] = va
    rule("PIECE 2 RANKING")
    print(f"{'lambda':>8} {'val F1':>9} {'std':>8} {'P':>9} {'R':>9}")
    print("-" * 48)
    for lam, v in sorted(res.items()):
        print(f"{lam:>8.1f} {v['offensive_f1']['mean']:>9.4f} {v['offensive_f1']['std']:>8.4f} "
              f"{v['offensive_precision']['mean']:>9.4f} {v['offensive_recall']['mean']:>9.4f}")
    base = res[0.0]["offensive_f1"]
    best = max(((l, v) for l, v in res.items() if l > 0),
               key=lambda kv: kv[1]["offensive_f1"]["mean"])
    print("\n" + verdict(f"best lambda={best[0]} vs lambda=0",
                         best[1]["offensive_f1"]["mean"], base["mean"],
                         max(base["std"], best[1]["offensive_f1"]["std"])))
    print(f"\nUse --lambda {best[0]} for Piece 4 even if the gain here is null:")
    print("the sentence head is REQUIRED for distillation to have anywhere to go.")

# ==========================================================================
elif args.piece4:
    rule("PIECE 4 - OFFLINE DISTILLATION FROM SemiSOLD")
    if args.lam <= 0:
        print("Piece 4 needs a sentence head. Pass --lambda (e.g. --lambda 0.3).")
        sys.exit(1)
    from semisold import load_semisold, describe, prepare, make_semisold_loader
    print("Loading SemiSOLD (145k tweets with saved teacher scores)...")
    raw = load_semisold()
    describe(raw)
    print()
    semi = prepare(raw, uncertainty_threshold=args.uncertainty)
    print("""
No pretrained language model is loaded, run, or backpropagated through here.
The teacher scores were computed by the dataset authors in 2022 and stored as
columns in a public file. We consume a published artifact.

They are SENTENCE level - SemiSOLD has no token labels - so distillation trains
the sentence head, and the shared encoder carries any benefit to the tokens.""")

    res = {}
    for dw in [0.0, 0.5, 1.0]:
        args.distill_weight = dw
        dl = None if dw == 0 else make_semisold_loader(semi, vocab, BATCH, sp=sp)
        va, _ = run(f"piece4_dw{dw}", args.loss, args.lam, distill_loader=dl)
        res[dw] = va
    rule("PIECE 4 RANKING")
    print(f"{'distill w':>10} {'val F1':>9} {'std':>8} {'P':>9} {'R':>9}")
    print("-" * 50)
    for dw, v in sorted(res.items()):
        print(f"{dw:>10.1f} {v['offensive_f1']['mean']:>9.4f} {v['offensive_f1']['std']:>8.4f} "
              f"{v['offensive_precision']['mean']:>9.4f} {v['offensive_recall']['mean']:>9.4f}")
    base = res[0.0]["offensive_f1"]
    best = max(((d, v) for d, v in res.items() if d > 0),
               key=lambda kv: kv[1]["offensive_f1"]["mean"])
    print("\n" + verdict(f"best weight={best[0]} vs no distillation",
                         best[1]["offensive_f1"]["mean"], base["mean"],
                         max(base["std"], best[1]["offensive_f1"]["std"])))
    print("""
The authors found lightweight models gain most from this augmentation
(BiLSTM+CBOW +2.78%, XLM-R +0.63%). If we see nothing, one likely reason is
that Piece 1 already moved us into the "already strong" regime where they
observed little benefit. Say that, with their numbers cited.""")

# ==========================================================================
else:
    rule("FINAL - WINNING COMBINATION, FULL TRAIN, TEST SCORED ONCE")
    print(f"loss {args.loss}   lambda {args.lam}   "
          f"distill weight {args.distill_weight if args.lam > 0 else 'n/a'}")
    dl = None
    if args.lam > 0 and args.distill_weight > 0:
        from semisold import load_semisold, prepare, make_semisold_loader
        semi = prepare(load_semisold(), uncertainty_threshold=args.uncertainty)
        dl = make_semisold_loader(semi, vocab, BATCH, sp=sp)
    va, ta = run("final_combined", args.loss, args.lam, distill_loader=dl, score_test=True)
    rule("RESULT")
    f1 = ta["offensive_f1"]["mean"]
    print(f"""
  Phase 1 baseline            {BASELINE:.4f}
  Piece 1 (subword) test      {PIECE1_TEST:.4f}
  this run                    {f1:.4f}   ({f1 - PIECE1_TEST:+.4f} vs Piece 1)

  precision {ta['offensive_precision']['mean']:.4f}   recall {ta['offensive_recall']['mean']:.4f}
  published XLM-R             0.7200
""")
    print(verdict("vs Piece 1", f1, PIECE1_TEST,
                  max(ta["offensive_f1"]["std"], 0.0066)))

print(f"""
========================================================================
WHAT TO COMMIT
========================================================================
  {CSV_PATH}
  results/{PIECE}_report.txt   (if you redirected stdout there)

Commit ONLY those two files. Do not commit artifacts/ or embeddings/.

    git add {CSV_PATH} results/{PIECE}_report.txt
    git commit -m "{PIECE} results ({args.owner})"
    git pull --rebase && git push
""")
