# Subword Modelling for Token-Level Sinhala Offensive Language Detection Without Pretraining

Code and run logs for the paper. The task is token-level offensive language
identification on the Sinhala Offensive Language Dataset (SOLD): label each word
of a tweet as offensive or not. The model is a BiLSTM-CRF tagger with a subword
channel: each word is split into SentencePiece BPE pieces, which a small per-word
BiLSTM encodes into one vector per word. The subword-only variant uses no
pretrained embeddings or language models.

## Results

Offensive-class F1 on the official SOLD test split. Ours: mean and standard
deviation over five seeds, trained on all 7,500 training tweets.

| Model | P | R | F1 | Trainable params | Pretrained artefact |
|---|---|---|---|---|---|
| Published BiLSTM + fastText | 0.74 | 0.48 | 0.60 | - | fastText |
| Our reproduction | 0.745 | 0.498 | 0.597 ± 0.010 | 188K | fastText |
| Lexicon baseline (ours) | 0.636 | 0.669 | 0.652 | 0 | none |
| Published XLM-T | 0.77 | 0.64 | 0.70 | ~277M | weights |
| Ours: word + subword | 0.728 | 0.675 | 0.700 ± 0.003 | 330K | fastText |
| **Ours: subword only** | 0.743 | 0.677 | **0.708 ± 0.007** | **176K** | **none** |
| Published XLM-R-large | 0.76 | 0.68 | 0.72 | ~559M | weights |

Published precision and recall are shown in corrected order: the token-level
evaluation script released with SOLD passes gold labels and predictions in
exchanged positions (documented in `src/metrics.py`).

## Setup

Python 3.10 or later. A GPU is recommended; every experiment fits on a free
Kaggle or Colab T4.

```bash
pip install -r requirements.txt
python tests/test_metrics.py
python tests/test_subword_alignment.py
```

SOLD and SemiSOLD are downloaded from HuggingFace (`sinhala-nlp/SOLD`,
`sinhala-nlp/SemiSOLD`) on first use. Models that keep the word channel need the
Sinhala fastText vectors (about 460 MB):

```bash
mkdir -p embeddings
wget -O embeddings/cc.si.300.vec.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.si.300.vec.gz
python notebooks/03_embeddings.py        # writes artifacts/embedding_matrix.npy
```

## Repository layout

| Path | Contents |
|---|---|
| `src/` | data loading, vocabulary and embeddings, SentencePiece and character tokenizers, the tagger, losses, training loop, metrics, SemiSOLD loader |
| `notebooks/` | numbered Python scripts, one per experiment (run from the repository root) |
| `runners/` | Kaggle and Colab notebooks that clone this repository and run the scripts on a free GPU |
| `results/` | logs of every run reported in the paper, plus executed runner notebooks |
| `tests/` | metric and subword-alignment tests |

## Naming

Script, flag and log names follow the project's internal stages. Steps 1-8 are
the numbered scripts. Phase 1 is the reproduction of the word-level baseline
(Steps 1-5); Phase 2 covers the subword model and later experiments. Within
Phase 2, Piece 1 is the subword channel, Piece 2 the sentence head
(`--piece2`), Piece 3 the balanced losses (`--piece3`) and Piece 4 distillation
(`--piece4`).

## Two protocols

- **Refit**: train on all 7,500 training tweets for a fixed 35 epochs, then score
  the test split once. Used for the headline results.
- **Development protocol**: hold out 1,500 training tweets (seed 42) for
  validation, train on the remaining 6,000 with early stopping (up to 60 epochs,
  patience 12), then score validation and test. Used for comparisons between
  variants.

## Reproducing the paper

| Paper | Command | Log in `results/` |
|---|---|---|
| Table I: lexicon baseline | `python notebooks/02_metric_check.py` | `step2_report.txt` |
| Table I: our reproduction | `python notebooks/05_full_train_refit.py` | `step5b_refit.txt` |
| Table I: ours | `python notebooks/08_final_model.py --both` | `step8_final.txt` |
| Table II: tokenizer sweep | `python notebooks/06_subword_tokenizer.py`, then `python notebooks/07_subword_model.py --sweep` | `step6_subword.txt`, `step7_sweep.txt` |
| Table II: BPE-1k, BPE-2k | `python notebooks/07_subword_model.py --final --sp bpe_1000` (and `bpe_2000`) | `step7b_bpe1000.txt`, `step7_final.txt` |
| Table II: similar-capacity control | `python notebooks/07_subword_model.py --final --sp bpe_24000 --piece-dim 4` | `step7b_matched_control.txt` |
| Sec. V-C: no word channel | `python notebooks/07_subword_model.py --final --sp bpe_1000 --no-word-channel` | `step7b_nowordch.txt` |
| Sec. V-C: mean pooling | `python notebooks/08_final_model.py --model full --pooling mean` | `step8_meanpool.txt` |
| Sec. V-C: no CRF | `python notebooks/07_subword_model.py --final --sp bpe_2000 --no-crf` | `step7b_nocrf.txt` |
| Table III: characters, Transformer encoder | `runners/extra_baselines_kaggle.ipynb` (`07_subword_model.py --sp char`, `--encoder transformer`) | `extra_*.txt` |
| Table III: fastText subword vectors | `runners/fasttext_timing_kaggle.ipynb` (`12_fasttext_oov.py`, then `07_subword_model.py --matrix ... --open-vocab`) | `fasttext_oov_prep.txt`, `extra_bilstm_word_ft*.txt` |
| Table IV, Fig. 2: errors by word familiarity | `python notebooks/10_analysis.py --errors` and `--errors --word-only` | `error_analysis.txt`, `error_analysis_wordonly.txt` |
| Table V: losses | `python notebooks/09_pieces_234.py --piece3` | `piece3_report.txt` |
| Table V: sentence head | `python notebooks/09_pieces_234.py --piece2 --seeds 1 2 3 4 5` | `piece2_report.txt` |
| Table V: distillation | `python notebooks/09_pieces_234.py --piece4 --lambda 0.3 --exclude-sold --seeds 1 2 3 4 5` | `piece4_nosold_report.txt` |
| Table VI: inference cost | `runners/fasttext_timing_kaggle.ipynb` (`10_analysis.py --efficiency --minimal`, `11_transformer_efficiency.py --all`) | `efficiency_minimal_*.txt`, `transformer_efficiency_*.txt` |

The three Table V experiments can also be run together on Colab with
`runners/table5_colab.ipynb`. `notebooks/12_fasttext_oov.py` needs the binary fastText model
(`cc.si.300.bin.gz`, 3.2 GB), which `runners/fasttext_timing_kaggle.ipynb`
downloads. `results/piece4_report.txt` is an earlier distillation run that still
contained the SemiSOLD tweets overlapping with SOLD; see `src/semisold.py`.

## Data notes

- The SOLD rationale column is named `rationales`, not `rationals` as documented.
  Non-offensive tweets store an empty list, which `src/data.py` expands to one
  zero per token.
- The public SemiSOLD release contains nearly every SOLD training and test tweet.
  `09_pieces_234.py --exclude-sold` removes them (10,087 rows) before
  distillation.
