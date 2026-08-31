Two conventions to agree on in the first meeting and then never revisit. First, main is always working; each person works on feature/<name> and merges by pull request that at least one other person reads. Second, nobody writes their own copy of the metric. Everything imports from src/metrics.py. The single most common way student teams produce contradictory results is three people quietly computing F1 three different ways.


# Token-Level Sinhala Offensive Language Detection (no pretrained language models)

CS3631 group project, University of Moratuwa.

## What this project does

Most systems can only say "this tweet is offensive" or "this tweet is not offensive".
Our model does something harder. It looks at every single word in a Sinhala tweet and
says which words are the offensive ones.

This is useful because a human moderator can see the highlighted words and check the
decision in one second. They do not have to trust the machine blindly.

The task is called **token-level offensive language detection**. A "token" is one word.

## The rule we work under

We are not allowed to fine-tune large pretrained language models like XLM-R or BERT.

We treat this as the point of the project, not a problem. Big models cost a lot of money
to run. A small model that gets close is more useful in the real world. XLM-R-large has
about 560 million parameters. Our model will be far smaller.

## What we must beat

These are published numbers on the same dataset and the same test split.

| Model | Uses a big pretrained model? | Macro F1 |
|---|---|---|
| BiLSTM + CBOW | No | 0.58 |
| BiLSTM + fastText | No | **0.60  <- our target to beat** |
| SinBERT | Yes | 0.62 |
| XLM-T | Yes | 0.70 |
| XLM-R | Yes | 0.72 |
| XLM-R + transfer from TSD | Yes | 0.73 (best published) |

For comparison: in English, 36 teams tried the same kind of task at SemEval-2021 and the
best score was about 0.68. This task is hard in every language.

## The data

We use **SOLD** (Sinhala Offensive Language Dataset) from HuggingFace: `sinhala-nlp/SOLD`.

Columns in the dataset:

| Column | What it is |
|---|---|
| `post_id` | Twitter ID |
| `text` | The raw tweet |
| `tokens` | The tweet split into words, separated by spaces |
| `rationales` | 1 if that word is offensive, 0 if not. **This is what we predict.** |
| `label` | `OFF` or `NOT` for the whole tweet |

## Two problems with the data (found in Step 1)

**Problem 1. The column name in the official README is wrong.**
The GitHub README says the column is called `rationals`. The real dataset calls it
`rationales`. Our loader checks for both spellings.

**Problem 2. Non-offensive tweets store an empty list, not a list of zeros.**
A `NOT` tweet has `rationales = []`. It does not have `[0, 0, 0, 0]`.
Our loader turns `[]` into a list of zeros with the correct length.

This second one is important. If you do not fix it, you silently throw away every
non-offensive tweet, which is 58% of the data.

## Verified numbers (from `notebooks/01_data_exploration.py`)

### Basic counts

| Thing | Train | Test |
|---|---|---|
| Rows | 7,500 | 2,500 |
| `NOT` tweets | 4,324 (57.7%) | 1,485 (59.4%) |
| `OFF` tweets | 3,176 (42.3%) | 1,015 (40.6%) |

Whole dataset: **4,191 OFF and 5,809 NOT**. This matches the published paper exactly.

### Data quality checks

| Check | Train | Test | Result |
|---|---|---|---|
| Rows with empty `[]` rationale | 4,523 (60.3%) | 1,549 (62.0%) | Expected |
| Rationale list with the wrong length | 0 | 0 | Good, no corruption |
| Lengths match after we expand `[]` | Yes | Yes | Good |
| Rationale values found | only 0 and 1 | only 0 and 1 | Good |
| `NOT` tweets with an offensive token | 0 | 0 | Good, as designed |

### The annotation problem we found

Some tweets are labelled `OFF` but no words are marked offensive.

| Type | Train | Test |
|---|---|---|
| `OFF` with empty `[]` rationale | 199 | 64 |
| `OFF` with a real list that is all zeros | 10 | 10 |
| **`OFF` with no offensive token at all** | **209 (6.6%)** | **74 (7.3%)** |

**Our decision: we keep these rows.**

Reasons. The rate is almost the same in train and test, so this is normal annotator
behaviour, not broken data. These are probably tweets that are offensive because of
context, not because of one bad word. They are part of the official split that all the
published scores used, so removing them would make our numbers not comparable.

At token level these rows give only 0 labels. At sentence level they are still `OFF`,
so they still teach our sentence head something in Phase 2.

### Class imbalance (this drives the whole project)

| | Train | Test |
|---|---|---|
| Total tokens | 176,370 | 59,670 |
| Offensive tokens | 7,294 | 2,268 |
| **Percent offensive** | **4.14%** | **3.80%** |

Only about 4 words in every 100 are offensive.

This means a model that says "not offensive" for every single word is about 96% accurate
and completely useless. That is why we never use accuracy. We use Macro F1.

Inside tweets that do have offensive words, 9.46% of tokens are offensive, and the
typical tweet has 2 offensive words.

Note: train is 4.14% and test is 3.80%. So class weights must be calculated from the
train split only.

### Tweet length

| | Train | Test |
|---|---|---|
| Shortest | 5 tokens | 5 tokens |
| Median | 19 | 20 |
| Average | 23.5 | 23.9 |
| 90th percentile | 49 | 49 |
| 99th percentile | 79 | 84 |
| Longest | 123 | 134 |

**Decision: `max_len = 80`.** This covers about 99% of tweets.

Do not use a small value like 48. It cuts off 10% of tweets and deletes their labels.
The paper says tweets are under 20 words, but that is the median. The tail is much
longer because punctuation is split into separate tokens. `.` is the most common token
in the whole dataset (19,439 times).

### Vocabulary and unknown words

| Thing | Value |
|---|---|
| Different words in train | 33,004 |
| Different words in test | 15,751 |
| Test words never seen in train | 7,165 (45.5% of word types) |
| **Test tokens never seen in train** | **7,851 (13.2% of all test tokens)** |

**This is the most important number for our design.**

About 1 in every 8 words at test time is a word the model has never seen. Almost half of
all different words are new. This happens because Sinhala is agglutinative, meaning it
packs grammar into word endings, so the same word appears in many different forms.

This is the evidence for our subword component in Phase 2. If the model can look inside
words instead of treating each word as one whole unit, it can still understand a word it
has never seen before.

### Sanity check on the labels

The words most often marked offensive have offensive rates of 91% to 100%, over 20 to
198 appearances each. They are clear Sinhala swear words. This confirms the labels mean
what they should.

## What we are building (Phase 2)

Four parts, added one at a time on top of the baseline.

1. **Subword input.** Split words into small pieces so the model can handle word endings
   and unseen words. Justified by the 13.2% unknown word rate above.
2. **Joint multi-task head.** Learn the sentence label and the word labels at the same
   time, sharing one encoder, so each task helps the other.
3. **Balanced loss.** Stop the model ignoring the rare offensive words. Justified by the
   4.14% imbalance above.
4. **Offline distillation.** Learn from SemiSOLD, which has 145,000 extra tweets with
   saved scores from 11 earlier models. Those scores are sentence-level only, so they
   train our sentence head, which then helps the token head through the shared encoder.
   No pretrained model is ever inside our network.

## How we measure success

- **Macro F1 on offensive tokens.** The main number. Directly comparable to the table above.
- **Precision and recall reported separately.** The published baseline has precision 0.48
  and recall 0.74, while XLM-R has precision 0.68 and recall 0.76. So the gap is mostly a
  precision problem. We flag too many innocent words, not too few offensive ones.
- **Model size and speed.** To support the efficiency claim.
- **Five random seeds, report the mean and the standard deviation.** Same protocol as the
  original paper. Calculate F1 for each seed first, then average the F1 values. Do not
  average precision and recall and then combine them.

## Repository structure

```
.
├── configs/
│   └── baseline.yaml           all hyperparameters live here
├── src/
│   ├── data.py                 the ONLY place that loads the dataset
│   ├── metrics.py              the ONLY place that calculates F1
│   ├── models/
│   └── train.py
├── notebooks/
│   └── 01_data_exploration.py  Step 1 checks
└── results/
    ├── results.csv             every run we have ever done
    └── step1_report.txt        output of the exploration script
```

## Rules for the team

1. Nobody calls `load_dataset()` outside `src/data.py`.
2. Nobody writes their own F1 function. Everyone imports from `src/metrics.py`.
3. Never touch the test split until a configuration is frozen. Tune on a validation split
   taken from train.
4. Every run gets logged to `results/results.csv` with its seed and full config.
5. Every claim in the paper must point to a number in a table. If it does not, soften it
   or delete it.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install "datasets<3.0.0" pandas numpy pyarrow matplotlib
python notebooks/01_data_exploration.py > results/step1_report.txt
```

## Progress

- [x] **Step 0** Repository set up
- [x] **Step 1** Data loaded and verified. All checks passed.
- [ ] **Step 2** Evaluation metric, matched to the SOLD paper
- [ ] **Step 3** fastText embeddings loaded, unknown word rate confirmed
- [ ] **Step 4** BiLSTM + CRF baseline built
- [ ] **Step 5** Five seeds run, baseline scores about 0.60, config frozen

## Data source

Ranasinghe, T., Anuradha, I., Premasiri, D., Silva, K., Hettiarachchi, H., Uyangodage, L.,
& Zampieri, M. (2024). SOLD: Sinhala offensive language dataset. *Language Resources and
Evaluation*. https://doi.org/10.1007/s10579-024-09723-1

Dataset: https://huggingface.co/datasets/sinhala-nlp/SOLD
Code: https://github.com/Sinhala-NLP/SOLD

## Warning

This repository works with real offensive social media text. The dataset contains
language that many people will find upsetting.