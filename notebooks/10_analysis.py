"""
Error analysis and efficiency measurement - the last two gaps in the paper.

    --errors       train one seed, dump test predictions against gold, and
                   produce an error breakdown plus example tweets for Section 7.7
    --efficiency   measure parameters, inference throughput and peak memory
                   for Section 9

Run:
    python notebooks/10_analysis.py --errors     > results/error_analysis.txt
    python notebooks/10_analysis.py --efficiency > results/efficiency.txt

--errors trains one seed (about 14 minutes on a T4) and SAVES the model, so run
it first; --efficiency then reuses that checkpoint.

WHAT THIS SCRIPT CANNOT DO FOR YOU
----------------------------------
It prints the tweets, the gold spans and the predicted spans. Deciding WHY a
miss happened - implicit offence, sarcasm, an unseen inflection, a plausible
annotation disagreement - requires reading Sinhala. Section 7.7 needs a human.
"""
import sys, os, argparse, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from collections import Counter

import numpy as np
import torch

from data import load_sold
from embeddings import build_vocab, load_vectors_for_vocab, build_embedding_matrix
from dataset import make_loader, unpad_predictions
from model import BiLSTMTagger
from subword import train_sentencepiece, load_sentencepiece, word_to_pieces
from train import set_seed, get_device
from metrics import token_level_scores

VEC = os.environ.get("SOLD_VECTORS", "embeddings/cc.si.300.vec.gz")
CKPT = "artifacts/analysis_model.pt"
SP_PREFIX = "artifacts/sp_final_bpe_1000_fulltrain"
EPOCHS, BATCH, SEED = 35, 32, 1
SP_MARK = "\u2581"


def rule(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


ap = argparse.ArgumentParser()
ap.add_argument("--errors", action="store_true")
ap.add_argument("--efficiency", action="store_true")
ap.add_argument("--minimal", action="store_true",
                help="analyse the subword-only model instead of word + subword")
ap.add_argument("--examples", type=int, default=24)
ap.add_argument("--epochs", type=int, default=EPOCHS)
args = ap.parse_args()
if not (args.errors or args.efficiency):
    print("Pass --errors or --efficiency")
    sys.exit(1)


# ==========================================================================
rule("0. SETUP")
device = get_device()
train_full = load_sold("train")
test = load_sold("test")
print(f"device {device}   train {len(train_full):,}   test {len(test):,}")
print(f"model: {'subword only' if args.minimal else 'word + subword'}")

os.makedirs("artifacts", exist_ok=True)
os.makedirs("results", exist_ok=True)

if not os.path.exists(SP_PREFIX + ".model"):
    print("training the tokenizer on the full train split")
    train_sentencepiece([" ".join(t) for t in train_full["token_list"]],
                        1000, SP_PREFIX, model_type="bpe")
sp = load_sentencepiece(SP_PREFIX + ".model")

vocab, _ = build_vocab(train_full["token_list"], min_freq=1)
if not os.path.exists(VEC):
    print(f"\nMISSING {VEC}. See notebooks/03_embeddings.py for the download.")
    sys.exit(1)
vectors, dim = load_vectors_for_vocab(VEC, vocab, verbose=False)
matrix, _ = build_embedding_matrix(vocab, vectors, dim)
print(f"tokenizer {sp.get_piece_size():,} pieces   vocab {len(vocab):,}")


def build():
    return BiLSTMTagger(
        matrix, hidden_size=64, dropout=0.5, freeze_embeddings=True, use_crf=True,
        n_pieces=sp.get_piece_size(), piece_dim=50, subword_dim=100,
        subword_pooling="bilstm", use_word_channel=not args.minimal)


def get_model():
    """Load the saved checkpoint, or train one and save it."""
    model = build().to(device)
    if os.path.exists(CKPT):
        model.load_state_dict(torch.load(CKPT, map_location=device))
        print(f"loaded {CKPT}")
        return model

    print(f"no checkpoint; training one seed for {args.epochs} epochs "
          f"(about 14 minutes on a T4)")
    g = set_seed(SEED)
    loader = make_loader(train_full, vocab, BATCH, shuffle=True, generator=g, sp=sp)
    opt = torch.optim.Adam(
        [q for q in model.parameters() if q.requires_grad], lr=1e-3)
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = n = 0
        for ids, lab, mask, lens, _, pid, plen in loader:
            ids, lab, mask = ids.to(device), lab.to(device), mask.to(device)
            if pid is not None:
                pid, plen = pid.to(device), plen.to(device)
            opt.zero_grad()
            loss = model.loss(ids, lab, mask, lens, piece_ids=pid, piece_lens=plen)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            n += 1
        if ep % 10 == 0 or ep == args.epochs:
            print(f"  epoch {ep}/{args.epochs}  loss {tot / max(n, 1):.4f}")
    torch.save(model.state_dict(), CKPT)
    print(f"saved {CKPT}")
    return model


# ==========================================================================
if args.errors:
    model = get_model()
    model.eval()

    rule("1. PREDICTIONS ON THE TEST SPLIT")
    loader = make_loader(test, vocab, BATCH, shuffle=False, sp=sp)
    gold_all, pred_all = [], []
    with torch.no_grad():
        for ids, lab, mask, lens, _, pid, plen in loader:
            ids, mask = ids.to(device), mask.to(device)
            if pid is not None:
                pid, plen = pid.to(device), plen.to(device)
            pred_all.extend(model.predict(ids, mask, lens, pid, plen))
            gold_all.extend(unpad_predictions(lab, mask.cpu()))

    s = token_level_scores(gold_all, pred_all)
    print(f"  P {s['offensive_precision']:.4f}   R {s['offensive_recall']:.4f}   "
          f"F1 {s['offensive_f1']:.4f}")
    print(f"  {s['support_offensive']:,} offensive of {s['support_total']:,} tokens")
    print("\n  This is ONE seed, so it will differ slightly from the 5-seed mean")
    print("  reported in the paper. Use the 5-seed figure for any headline claim.")

    # ------------------------------------------------------------------
    rule("2. ERRORS BY WORD FAMILIARITY   <-- TESTS OUR CENTRAL CLAIM")
    print("""Our claim is that subwords help on words never seen in training. This
table tests it directly: does the model still fail disproportionately on unseen
words, or has that gap closed?
""")
    train_words = set(w for toks in train_full["token_list"] for w in toks)
    buckets = {"seen": Counter(), "unseen": Counter()}
    for toks, g, pr in zip(test["token_list"], gold_all, pred_all):
        for w, gl, pl in zip(toks, g, pr):
            b = "seen" if w in train_words else "unseen"
            buckets[b]["n"] += 1
            if gl == 1 and pl == 1:
                buckets[b]["tp"] += 1
            elif gl == 0 and pl == 1:
                buckets[b]["fp"] += 1
            elif gl == 1 and pl == 0:
                buckets[b]["fn"] += 1

    print(f"{'bucket':<10} {'tokens':>9} {'TP':>7} {'FP':>7} {'FN':>7} "
          f"{'P':>8} {'R':>8} {'F1':>8}")
    print("-" * 68)
    scores = {}
    for b, c in buckets.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        pp = tp / (tp + fp) if tp + fp else 0.0
        rr = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pp * rr / (pp + rr) if pp + rr else 0.0
        scores[b] = f1
        print(f"{b:<10} {c['n']:>9,} {tp:>7,} {fp:>7,} {fn:>7,} "
              f"{pp:>8.4f} {rr:>8.4f} {f1:>8.4f}")

    gap = scores["seen"] - scores["unseen"]
    print(f"""
  seen minus unseen F1: {gap:+.4f}

INTERPRET CAREFULLY, AND REPORT WHICHEVER IT IS.
  A large positive gap means subwords narrowed the unseen-word problem but did
  not solve it. Say that; it is an honest limitation and a clear direction for
  future work.
  A small gap means the mechanism claim is strongly supported.
  Either way this number belongs in Section 7.7.""")

    # ------------------------------------------------------------------
    rule("3. THE MOST FREQUENT INDIVIDUAL ERRORS")
    fp_c, fn_c = Counter(), Counter()
    for toks, g, pr in zip(test["token_list"], gold_all, pred_all):
        for w, gl, pl in zip(toks, g, pr):
            if gl == 0 and pl == 1:
                fp_c[w] += 1
            if gl == 1 and pl == 0:
                fn_c[w] += 1

    print("FALSE POSITIVES - flagged but not offensive:\n")
    for w, c in fp_c.most_common(15):
        tag = "seen" if w in train_words else "UNSEEN"
        print(f"  {c:>4}x  {w:<24} ({tag})")

    print("\nFALSE NEGATIVES - offensive but missed:\n")
    for w, c in fn_c.most_common(15):
        tag = "seen" if w in train_words else "UNSEEN"
        pcs = " | ".join(p.lstrip(SP_MARK) for p in word_to_pieces(sp, w))
        print(f"  {c:>4}x  {w:<24} ({tag})  ->  {pcs}")

    print("""
Piece splits are shown for the misses so you can check whether a missed word
decomposed into a root the model should have recognised. If it did and the model
still missed it, the failure is contextual rather than lexical - which is a
different and more interesting kind of error.""")

    # ------------------------------------------------------------------
    rule("4. EXAMPLE TWEETS FOR SECTION 7.7")
    print("Legend:  [word] correct hit   <word> false positive   {word} missed\n")
    cats = {"perfect": [], "over-flagged": [], "under-flagged": [], "mixed": []}
    per_cat = max(args.examples // 4, 1)
    for i, (toks, g, pr) in enumerate(zip(test["token_list"], gold_all, pred_all)):
        if sum(g) == 0 and sum(pr) == 0:
            continue
        fp = sum(1 for a, b in zip(g, pr) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(g, pr) if a == 1 and b == 0)
        k = ("perfect" if fp == 0 and fn == 0 else
             "over-flagged" if fn == 0 else
             "under-flagged" if fp == 0 else "mixed")
        if len(cats[k]) < per_cat:
            cats[k].append((i, toks, g, pr))

    for k, items in cats.items():
        print(f"--- {k.upper()} ---")
        for i, toks, g, pr in items:
            marked = []
            for w, gl, pl in zip(toks, g, pr):
                if gl == 1 and pl == 1:
                    marked.append(f"[{w}]")
                elif gl == 0 and pl == 1:
                    marked.append(f"<{w}>")
                elif gl == 1 and pl == 0:
                    marked.append(f"{{{w}}}")
                else:
                    marked.append(w)
            print(f"  #{i}  sentence label = {test.iloc[i]['label']}")
            print(f"    {' '.join(marked)}")
        print()

    print("""NEXT STEP, AND IT NEEDS A HUMAN.

Pick five to eight examples above and write Section 7.7. For each, say WHY the
model behaved as it did: an unseen inflection of a known root, offence carried by
context rather than any single word, sarcasm, or a plausible annotation
disagreement. That judgement requires reading Sinhala and cannot be automated.

Reviewers consistently reward this section and almost no student paper has one.""")

    with open("results/error_analysis_raw.json", "w") as fh:
        json.dump({"gold": gold_all, "pred": pred_all,
                   "tokens": [list(t) for t in test["token_list"]],
                   "labels": list(test["label"])}, fh)
    print("\nraw predictions saved to results/error_analysis_raw.json")


# ==========================================================================
if args.efficiency:
    rule("EFFICIENCY MEASUREMENTS - FOR SECTION 9")
    model = get_model()
    model.eval()

    p = model.count_parameters()
    print("parameters (measured):")
    for k, v in p.items():
        print(f"  {k:<26} {v:>14,}")

    loader = make_loader(test, vocab, BATCH, shuffle=False, sp=sp)
    batches = list(loader)

    def run_all():
        n_tok = n_tweet = 0
        with torch.no_grad():
            for ids, _, mask, lens, _, pid, plen in batches:
                ids, mask = ids.to(device), mask.to(device)
                if pid is not None:
                    pid_, plen_ = pid.to(device), plen.to(device)
                else:
                    pid_ = plen_ = None
                model.predict(ids, mask, lens, pid_, plen_)
                n_tok += int(mask.sum())
                n_tweet += ids.size(0)
        return n_tok, n_tweet

    run_all()   # warm up; first pass includes kernel compilation
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    n_tok, n_tweet = run_all()
    if device.type == "cuda":
        torch.cuda.synchronize()
    el = time.time() - t0

    print(f"""
inference over the full test split, measured on {device}:
  tweets                 {n_tweet:,}
  real tokens            {n_tok:,}
  wall clock             {el:.2f}s
  tweets per second      {n_tweet / el:,.0f}
  tokens per second      {n_tok / el:,.0f}
  ms per tweet           {el / n_tweet * 1000:.3f}""")
    if device.type == "cuda":
        print(f"  peak GPU memory        {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    print(f"""
  model size at float32  ~{p['total'] * 4 / 1e6:.1f} MB
  external files needed  {'none' if args.minimal else 'cc.si.300.vec.gz (460 MB)'}

FOR THE PAPER.
Report these as MEASURED and state the hardware and batch size ({BATCH}).
Published transformer figures should be cited as parameter counts only, unless
you time XLM-R yourself on the same GPU in inference-only mode. Loading a model
to time its forward pass is not fine-tuning, so that comparison is available to
you if you want a like-for-like throughput number - but say clearly which figures
are measured and which are cited.""")
