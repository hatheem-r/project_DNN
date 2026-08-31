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

## Step 2: the evaluation metric

### Why this step came before any model

A metric is one number that says how good the model is. The danger is that
there are several ways to calculate it, and they give different answers on the
same predictions. If the SOLD authors used one way and we use another, our 0.65
might really be worse than their 0.60 and we would never know. So we copied
their method before writing any model.

### The four numbers

Say a tweet has 20 words. 3 are truly offensive. Our model flags 5 words, and 2
of those 5 are correct.

- True positives = 2. Flagged and correct.
- False positives = 3. Flagged but innocent.
- False negatives = 1. Offensive but missed.
- **Precision** = 2/5 = 0.40. Of what we flagged, how much was right.
- **Recall** = 2/3 = 0.67. Of what we should have caught, how much we caught.
- **F1** = 0.50. Both combined. It punishes being lopsided.

We never use accuracy. 96% of words are not offensive, so a model that says
"not offensive" to everything is 96% accurate and useless.

### What we found in the official SOLD code

We cloned https://github.com/Sinhala-NLP/SOLD and read
`experiments/token_level/print_stat.py` and `sinhala_mudes.py`. Three findings.

**1. The headline number is the F1 of the offensive class, not macro F1.**
The paper's text says "Macro F1" but the table numbers are per-class. Proof:
the paper gives XLM-R P=0.68, R=0.76, F1=0.72, and the harmonic mean of 0.68
and 0.76 is 0.718. A macro F1 over both classes would be about 0.85, because
the not-offensive class is easy and huge. So we compare on offensive-class F1.

**2. Their precision and recall are swapped.**
`print_information(df, pred_column, real_column)` sets
`predictions = df[pred_column]`, but it is called as
`print_information(test_data, "labels", "predictions")`. So the gold column
goes into the predictions variable and the model output goes into the real
variable. sklearn then receives `(y_true=model_output, y_pred=gold)`, which is
backwards. Swapping those turns recall into precision and precision into
recall. **F1 is symmetric so F1 is unaffected.** We match their F1 and do not
expect to match their printed P and R order. Our code computes P and R the
correct way round.

**3. Evaluation is pooled, not per-sentence.**
They build one table with a row per token across all 2,500 test tweets and run
one calculation. There is no per-tweet averaging.

Two extra confirmations from their code:

- `if len(labels) == 0: label 0 for every token`. The official code expands
  empty rationales to zeros and keeps those rows, exactly matching our Step 1
  decision.
- `experiments/token_level/` contains only `sinhala_mudes.py` and
  `sinhala_lime.py`, both transformer-based. There is no LSTM or CNN token-level
  script anywhere in the repository. **Our research gap is confirmed from the
  authors' own code**, which is stronger than saying we searched and found
  nothing.

### Our metric, written down

Headline: **F1 of the offensive class**, pooled over all test tokens.
Also reported: precision, recall, not-offensive F1, macro F1, weighted F1.
Padding positions are excluded. Short predictions are padded with the negative
class, the same as the official code.

Across seeds: calculate F1 for each seed first, then average the F1 values.
Never average precision and recall and then combine them. That gives a
different number, and it is probably why the paper's BiLSTM rows do not match
the harmonic mean of their own printed P and R.

## The three splits

SOLD ships with only train and test. **We create the validation split ourselves**
by cutting 20% off train. It is defined once in `src/data.py` as
`train_val_split()` with seed 42, so every model in the project is tuned and
compared on identical data.

| Split | Rows | Purpose | Positive token rate |
|---|---|---|---|
| train-part | 6,000 | Build things | 4.15% |
| validation | 1,500 | **Choose** things | 4.07% |
| test | 2,500 | Locked. Scored once, at the end | 3.80% |

**The rule.** You may run something on test. You may not use test scores to
*choose* anything. Choosing happens on validation.

An earlier version of our metric script tried four word lists on test and
reported the best one (0.631). That was test-set tuning and the number was
void. It has been removed.

## Baselines (verified, no neural network)

Run with `python notebooks/02_metric_check.py`.

Baselines A, B and C have no settings to choose, so running them directly on
test is legitimate. Baseline D has settings, so they were chosen on validation.

| Baseline | Precision | Recall | Offensive F1 |
|---|---|---|---|
| A. Flag every token | 0.0380 | 1.0000 | 0.0732 |
| B. Flag no token | 0.0000 | 0.0000 | **0.0000** |
| C. Random at 4.14% | 0.0403 | 0.0432 | 0.0417 |
| **D. Word list (validation-chosen)** | **0.6361** | **0.6689** | **0.6521** |

Word list setting, chosen on validation: `min_count=1`, `threshold=0.3`,
1,506 words after rebuilding on the full train split.
Validation F1 was 0.6453 and test F1 is 0.6521, a gap of only +0.0068.

Checks that passed:

- Flag-everything precision is 0.0380, exactly the test positive rate, and
  recall is exactly 1.0000. The paper reports 0.03 precision for this same
  baseline, so **our metric agrees with theirs**.
- Flag-nothing gives offensive F1 exactly 0.0000 while accuracy would be 96%.
  Its macro F1 is still 0.4903, which proves macro F1 over both classes
  flatters a useless model. This is why we report offensive-class F1.
- Random gives F1 0.0417, near the positive rate, as expected.
- All 9 unit tests in `tests/test_metrics.py` pass.
- The validation-to-test gap is tiny, so validation is a reliable guide for
  every hyperparameter choice in Phase 2.

## KEY FINDING: a word list beats three published models

| Model | Offensive F1 | Learning involved? |
|---|---|---|
| Published BiLSTM + CBOW | 0.58 | Yes |
| Published BiLSTM + fastText | 0.60 | Yes |
| Published SinBERT | 0.62 | Yes |
| **Our word list** | **0.6521** | **No. It is a lookup table.** |
| Published XLM-T | 0.70 | Yes |
| Published XLM-R | 0.72 | Yes |
| Published XLM-R + TSD transfer | 0.73 | Yes |

A list of 1,506 words with no neural network, no embeddings and no training
beats the published non-transformer baseline by 0.05 and beats SinBERT, a
transformer pretrained on Sinhala.

**Three consequences.**

**1. Our floor moved.** Beating 0.60 is no longer enough to justify a neural
model. Our BiLSTM must clearly beat **0.6521**.

**2. This is a result for the paper.** Nobody has published a lexicon baseline
on SOLD at token level. It is the kind of finding reviewers remember.

**3. It supports our main argument with a number.** Why does a trained BiLSTM
score 0.60 when a word list scores 0.65? The likely answer is that the
token-level BiLSTM in the SOLD paper was reported in passing and never properly
tuned. Previously our claim that "the authors built a careful lightweight path
for sentence level and never built one for token level" rested on the file
structure of their repository. Now it also rests on evidence.

The result is robust, not a fluke. The top 15 settings out of 45 all score
between 0.599 and 0.645 on validation, so a wide range of sensible word lists
lands in the same place.

**Where the gap lives.** The published BiLSTM has precision 0.48 and recall
0.74. Our word list has precision 0.636 and recall 0.669. XLM-R has precision
0.68 and recall 0.76. Recall is similar across all of them. **The difference is
precision.** Lightweight models flag too many innocent words. Our Phase 2
components should therefore target precision, not recall.

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

- **F1 of the offensive class**, pooled over all test tokens. The main number.
  Directly comparable to the table above. See the Step 2 section for why this is
  the offensive-class F1 and not a macro average.
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
│   ├── 01_data_exploration.py  Step 1 checks
│   └── 02_metric_check.py      Step 2 trivial baselines
├── tests/
│   └── test_metrics.py         9 unit tests, must always pass
└── results/
    ├── results.csv             every run we have ever done
    ├── step1_report.txt        output of the exploration script
    └── step2_report.txt        output of the metric check
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
python tests/test_metrics.py
python notebooks/02_metric_check.py > results/step2_report.txt
```

## Progress

- [x] **Step 0** Repository set up
- [x] **Step 1** Data loaded and verified. All checks passed.
- [x] **Step 2** Metric copied from the official code. 9 unit tests pass.
      Validation split created. Baselines run honestly.
      **Word list = 0.6521. This is the floor our model must beat.**
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