"""
Transformer inference efficiency, for the Section 9 comparison the lecturer asked
for. Measures the same quantities as notebooks/10_analysis.py --efficiency, on
the same hardware and same test split, so the two tables sit side by side.

WHAT THIS DOES NOT DO
----------------------
It does not fine-tune anything. Each transformer is loaded with its public
pretrained weights plus a fresh, UNTRAINED classification head, then timed doing
a forward pass over the test tweets. We are timing the ARCHITECTURE's inference
cost, not reproducing its accuracy - accuracy numbers are cited from the
published papers, never measured here. Loading a public model to time it is not
fine-tuning it, so this does not touch the no-PLM rule that governs our own model.

Run:
    python notebooks/11_transformer_efficiency.py --model sinbert
    python notebooks/11_transformer_efficiency.py --model xlmt
    python notebooks/11_transformer_efficiency.py --model xlmr-base
    python notebooks/11_transformer_efficiency.py --model xlmr-large
    python notebooks/11_transformer_efficiency.py --all      # all four in one run

On CPU, time a sample rather than all 2,500 tweets - XLM-R-large over the full
split takes tens of minutes. Tweets per second is the comparable figure:
    CUDA_VISIBLE_DEVICES="" python notebooks/11_transformer_efficiency.py \\
        --all --allow-cpu --limit 250 --threads 1
"""
import sys, os, time, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from data import load_sold

MODELS = {
    "sinbert":    "NLPC-UOM/SinBERT-large",
    "xlmt":       "cardiffnlp/twitter-xlm-roberta-base",
    "xlmr-base":  "xlm-roberta-base",
    "xlmr-large": "xlm-roberta-large",
}
BATCH = 32
MAX_LEN = 128   # covers our p99 tweet length of 84 tokens with headroom


def rule(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bench_one(name, hf_id, test_texts, device, latency_tweets=0):
    rule(f"{name}  ({hf_id})")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(hf_id)
    # 2 labels to match our own task shape (offensive / not offensive per token).
    # Weights for this head are random - we are not training or evaluating
    # accuracy, only timing the forward pass through the loaded architecture.
    model = AutoModelForTokenClassification.from_pretrained(hf_id, num_labels=2)
    model.to(device).eval()
    load_s = time.time() - t0

    total = sum(p.numel() for p in model.parameters())
    size_mb = total * 4 / 1e6
    print(f"  loaded in {load_s:.1f}s")
    print(f"  parameters (measured): {total:,}")
    print(f"  size at float32: {size_mb:,.1f} MB")

    batches = []
    for i in range(0, len(test_texts), BATCH):
        chunk = test_texts[i:i + BATCH]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN)
        batches.append(enc)

    def run_all():
        n_tok = n_tweet = 0
        with torch.no_grad():
            for enc in batches:
                enc = {k: v.to(device) for k, v in enc.items()}
                model(**enc)
                n_tok += int(enc["attention_mask"].sum())
                n_tweet += enc["input_ids"].size(0)
        return n_tok, n_tweet

    if device.type == "cuda":
        run_all()  # warm up
    else:
        with torch.no_grad():  # one batch is enough to warm up on CPU
            model(**{k: v.to(device) for k, v in batches[0].items()})
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    n_tok, n_tweet = run_all()
    if device.type == "cuda":
        torch.cuda.synchronize()
    el = time.time() - t0

    result = {
        "model": name, "hf_id": hf_id, "device": str(device),
        "parameters": total, "size_mb": round(size_mb, 1),
        "load_seconds": round(load_s, 1),
        "wall_clock_s": round(el, 2),
        "tweets_per_s": round(n_tweet / el, 1),
        "tokens_per_s": round(n_tok / el, 1),
        "ms_per_tweet": round(el / n_tweet * 1000, 3),
    }
    if device.type == "cuda":
        result["peak_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    # Batch-1 latency, matching notebooks/10_analysis.py --efficiency.
    if latency_tweets:
        lat_ms = []
        with torch.no_grad():
            for k, text in enumerate(test_texts[:latency_tweets + 3]):
                enc = tok([text], return_tensors="pt", truncation=True, max_length=MAX_LEN)
                enc = {kk: v.to(device) for kk, v in enc.items()}
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                model(**enc)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if k >= 3:                       # first few are warm-up
                    lat_ms.append((time.perf_counter() - t1) * 1000)
        lat_ms.sort()
        result["latency_ms_median"] = round(lat_ms[len(lat_ms) // 2], 3)
        result["latency_ms_p95"] = round(lat_ms[int(len(lat_ms) * 0.95)], 3)

    print(f"\n  inference over {n_tweet:,} tweets, batch {BATCH}, max_len {MAX_LEN}:")
    print(f"    wall clock       {result['wall_clock_s']}s")
    print(f"    tweets/s         {result['tweets_per_s']:,}")
    print(f"    tokens/s         {result['tokens_per_s']:,}")
    print(f"    ms/tweet         {result['ms_per_tweet']}")
    if "peak_memory_mb" in result:
        print(f"    peak GPU memory  {result['peak_memory_mb']} MB")
    if "latency_ms_median" in result:
        print(f"    batch-1 latency  median {result['latency_ms_median']} ms, "
              f"95th pct {result['latency_ms_p95']} ms")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=list(MODELS), help="one model")
ap.add_argument("--all", action="store_true", help="run all four")
ap.add_argument("--allow-cpu", action="store_true",
                help="run on CPU instead of refusing; pair with --limit")
ap.add_argument("--limit", type=int, default=0,
                help="time only the first N test tweets (0 = all 2,500)")
ap.add_argument("--threads", type=int, default=0,
                help="limit PyTorch to this many CPU threads (0 = PyTorch default)")
ap.add_argument("--latency-tweets", type=int, default=0,
                help="also time this many tweets one at a time (batch-1 latency)")
args = ap.parse_args()
if not (args.model or args.all):
    print("Pass --model <name> or --all. Names:", list(MODELS)); sys.exit(1)
if args.threads:
    torch.set_num_threads(args.threads)

rule("SETUP")
device = get_device()
print(f"device: {device}   CPU threads used: {torch.get_num_threads()} "
      f"(of {os.cpu_count()} available)")
if device.type != "cuda" and not args.allow_cpu:
    print("\nWARNING: no GPU detected. Runtime -> Change runtime type -> T4 GPU.")
    print("To time on CPU deliberately, pass --allow-cpu with --limit (e.g. 250).")
    sys.exit(1)
if device.type != "cuda" and not args.limit:
    print("\nWARNING: timing all 2,500 tweets on CPU. XLM-R-large alone may take")
    print("tens of minutes. Consider --limit 250.")

test = load_sold("test")
test_texts = [" ".join(t) for t in test["token_list"]]
if args.limit:
    test_texts = test_texts[:args.limit]
print(f"test tweets timed: {len(test_texts):,}")

todo = list(MODELS.items()) if args.all else [(args.model, MODELS[args.model])]
results = []
for name, hf_id in todo:
    try:
        results.append(bench_one(name, hf_id, test_texts, device, args.latency_tweets))
    except Exception as e:
        print(f"\n  FAILED to benchmark {name}: {e}")
        print("  Report this error to the group rather than skipping silently -")
        print("  it may mean the model needs a different loading class.")

rule("SUMMARY - FOR SECTION 9")
print(f"device {device}, {torch.get_num_threads()} CPU threads, "
      f"{len(test_texts):,} tweets, batch {BATCH}")
print(f"{'model':<12} {'params':>14} {'MB':>8} {'tweets/s':>10} {'ms/tweet':>10} "
      f"{'b1 ms':>8} {'peak MB':>9}")
print("-" * 78)
for r in results:
    print(f"{r['model']:<12} {r['parameters']:>14,} {r['size_mb']:>8.1f} "
          f"{r['tweets_per_s']:>10,} {r['ms_per_tweet']:>10.3f} "
          f"{r.get('latency_ms_median', '-'):>8} {r.get('peak_memory_mb', '-'):>9}")

print("""
Compare against results/efficiency_minimal_<device>_t<threads>.json from
notebooks/10_analysis.py --efficiency --minimal, run on the same hardware with
the same thread setting. Compare tweets/s and batch-1 ms, not tokens/s: these
models count their own subword tokens, ours counts words.
""")

os.makedirs("results", exist_ok=True)
tag = f"{device.type}_t{torch.get_num_threads()}" + (f"_n{args.limit}" if args.limit else "")
with open(f"results/transformer_efficiency_{tag}.json", "w") as fh:
    json.dump(results, fh, indent=2)
print(f"saved results/transformer_efficiency_{tag}.json")
print("\nAlso commit the printed summary table above (redirect stdout to a .txt file).")
