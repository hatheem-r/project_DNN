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

**Our final Phase 1 baseline: 0.5965 +/- 0.0103.** See Step 5b below.

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

## Step 3: vocabulary and word embeddings

### What an embedding is

A neural network can only do arithmetic. It cannot read Sinhala. So every word
must first become a list of numbers, called a vector. That vector is the
embedding.

The numbers are not random. They were learned by a program that read a very
large amount of Sinhala text and noticed which words appear in similar places.
Words used in similar ways end up with similar vectors. So the model starts out
already knowing something about Sinhala, instead of having to work it out from
our 6,000 training tweets.

Learning the vectors from our own data instead is possible but much worse. The
SOLD paper tried it and scored 0.55, the worst of their three options.

### What Step 3 did

1. Listed every distinct word in the 6,000 train-part tweets and gave each an
   ID number. That is the vocabulary: **28,456 words**.
2. Opened the fastText Sinhala file (**808,044 words**, each with 300 numbers)
   and pulled out the vectors for our words.
3. Built a matrix of **28,456 rows by 300 columns**. Row n is the vector for
   word n.
4. Saved it to `artifacts/` for Step 4 to load.

### Why fastText and not word2vec or GloVe

fastText builds a word's vector out of its character chunks, not out of the
whole word as one unit. Sinhala is agglutinative, so one root appears in many
different forms with different endings. Character chunks let those forms share
information. Published work finds fastText beats word2vec and GloVe for Sinhala
for this reason, and the SOLD paper agrees: fastText 0.60, CBOW 0.58,
learned-from-scratch 0.55 at token level.

### How unknown words are handled

Copied exactly from `get_emb_matrix()` in the SOLD codebase. The whole matrix
starts filled with random numbers drawn from the same mean and standard
deviation as the real vectors (mean -0.0002, std 0.0479). Then the rows of
words that WERE found get overwritten with their real vector. So a word with no
vector still gets numbers of a sensible size.

### Results

| Thing | Value |
|---|---|
| Embedding file | `cc.si.300.vec.gz` (fastText Common Crawl Sinhala) |
| Words in the file | 808,044 |
| Dimension | 300 |
| Our vocabulary | 28,456 words, `min_freq=1`, from train-part only |
| Matrix | 28,456 x 300 |
| Words with a real vector | 23,314 (81.9%) |
| Words with a random vector | 5,142 |

Coverage per split. "Type" means distinct words. "Token" means word
occurrences. Token coverage is higher because common words are usually covered.

| Split | Types | Tokens | Type coverage | Token coverage |
|---|---|---|---|---|
| train-part | 28,454 | 141,646 | 81.94% | 92.11% |
| validation | 10,574 | 34,724 | 55.47% | 81.47% |
| test | 15,751 | 59,670 | 49.83% | 81.44% |

**On test: 14.3% of word occurrences never appeared in training, and 48.7% of
distinct words never appeared in training.**

Those two numbers together say something specific. The unfamiliar words are
mostly rare ones, so they are a small share of occurrences but nearly half of
the dictionary. The model has to guess at almost half of the different words it
meets at test time.

### Vocabulary size versus minimum frequency

`min_freq` drops rare words. Every dropped word becomes `<UNK>`.

| min_freq | Vocabulary | Train token coverage | Validation token coverage |
|---|---|---|---|
| **1** | **28,456** | **100.00%** | **85.87%** |
| 2 | 9,776 | 86.81% | 80.89% |
| 3 | 5,934 | 81.39% | 77.51% |
| 5 | 3,356 | 75.29% | 72.98% |

Note how many words appear only once: dropping them cuts the vocabulary from
28,456 to 9,776. We use `min_freq=1` for the baseline. It is a hyperparameter
and will be tuned on validation in Step 5, never on test.

## KEY OBSERVATION: fastText neighbours are word endings, not synonyms

The nearest neighbours of a word, by cosine similarity in the embedding space:

```
කියලා     ->  කියලානේ, කියලාද, කියලාත්, කිව්වා, කියලත්
කරන්න     ->  කරන්නත්, කරගන්න, වෙන්න, කරන්නම, කරන්නවා
සක්කිලි   ->  සක්කිලිය, සක්කිලියා, සක්කිලියෙක්, සක්කිලියන්ගේ, අවජාතකයො
පකයෙක්    ->  පබයෙක්, පරයෙක්, අපතයෙක්, කුබියෙක්, ගෝතයෙක්
```

These are not synonyms. **They are the same word with different endings.**
`සක්කිලි` and its four neighbours are one slur in four grammatical forms.
`කරන්න`, `කරන්නත්`, `කරන්නම`, `කරන්නවා` are one verb inflected four ways.

**Why this matters.** Our word-level model treats each of those as a completely
unrelated ID number. To the BiLSTM, `සක්කිලි` is word #4,102 and `සක්කිලියා` is
word #19,887, and it has no idea they are related. It must learn each one
separately from however few examples each has. And when it meets a form like
`සක්කිලියන්ට` at test time that never appeared in training, it has nothing.

This is very likely what much of our 48.7% unseen-type figure actually is: not
new words, but **new forms of familiar words**.

We now have three independent pieces of evidence for the subword component:

1. 48.7% of distinct test words are unseen (measured, Step 3)
2. fastText neighbours are dominated by morphological variants (observed, Step 3)
3. Published work finds fastText beats word2vec and GloVe for Sinhala precisely
   because it uses subword information

Good news in the same output: `පකයෙක් -> පබයෙක්, පරයෙක්, අපතයෙක්, කුබියෙක්`
shows different slurs sharing the `-යෙක්` ending. Offensive words cluster
together in the vector space.

### A hypothesis that turned out wrong

We expected offensive words to be **under**-covered by fastText, on the theory
that Common Crawl is clean web text where slang and profanity are rare.

The opposite happened. Offensive words are covered at **96.2%** against 81.9%
overall.

But the comparison is confounded and we should not report it as it stands. The
offensive set is filtered to words seen 3 or more times, while the 81.9%
baseline includes every vocabulary word, thousands of which appear exactly once.
Rare words are the ones fastText misses, so we may be measuring a frequency
effect rather than an offensiveness effect.

**TODO before the paper:** compute coverage for all words seen 3+ times and
compare against the offensive words' 96.2%. If they are similar, the effect is
purely frequency. If offensive words are still higher, that is a genuine
finding. Owner: Person 2.

### Known limitation to carry into Step 4

5,142 vocabulary words have random vectors. If embeddings are **frozen**, which
is what the baseline does and what the paper did, those words are stuck with
meaningless numbers forever. The model can never learn them.

Three candidate fixes, to be compared as a Step 5 ablation rather than decided
now:

1. Map any word without a real vector to `<UNK>`, so they share one consistent
   representation instead of 5,142 different random ones.
2. Keep them and unfreeze the embedding layer so the model can learn them.
3. Keep them frozen and accept the noise. This is the current baseline.

### Planned Phase 2 experiment, deliberately NOT used now

fastText also ships a `.bin` model that can **generate** a vector for a word it
has never seen, by combining that word's character chunks. That would fix much
of our 14.3% unknown-token problem.

We are not using it in the baseline, for two reasons. It is almost certainly not
what the paper did, so it would break our reproduction. And more importantly it
**is** a subword mechanism, which is our Phase 2 contribution. Using it now
would fold our own contribution into the baseline and make our improvement look
smaller than it is.

## Step 4: the baseline model (BiLSTM + fastText + CRF)

### What the model is

Three parts stacked on top of each other.

**Embedding layer.** Looks up the 300-number fastText vector for each word.
Frozen, meaning the vectors never change during training. The model uses
fastText's knowledge as given, which is what the paper did.

**BiLSTM.** Reads the tweet word by word while keeping a running memory, then
does it again backwards, and combines both directions. So the representation of
word 7 knows about every word before AND after it. This matters because
offensiveness is contextual: the SOLD paper shows most "offensive" keywords are
only offensive 30-50% of the time. That is exactly why our word list stalled at
0.65 - a lookup table cannot see context.

**CRF.** Without it, every word is labelled independently and the model can
produce incoherent sequences. A CRF scores the whole label sequence instead,
learning that offensive words come in runs.

### Padding and masking

Neural networks need rectangular blocks of numbers, but tweets have different
lengths. Short tweets get padded with a filler token up to the longest tweet
**in that batch**. Those filler positions are not real words, so they are
excluded from both the loss and the metric. Otherwise the model would get
credit for correctly labelling empty space.

**No truncation.** We changed the earlier `max_len = 80` decision. With
per-batch padding a 134-token tweet costs almost nothing, while truncating at 80
would silently delete the labels of about 1% of tweets.

### Result

Two runs. The first used `--epochs 30 --patience 5`. Inspecting it showed the
models were being stopped while still improving, so the second run used
`--epochs 60 --patience 12`. That is a validation-based decision, not a
test-based one.

| Run | Precision | Recall | **F1** | Std |
|---|---|---|---|---|
| Run 1: epochs 30, patience 5 | 0.7558 | 0.4581 | 0.5701 | 0.0178 |
| **Run 2: epochs 60, patience 12** | **0.7299** | **0.4899** | **0.5847** | **0.0195** |

Run 2 per-seed test F1: 0.6023, 0.5828, 0.5545, 0.6019, 0.5823.
Five seeds, F1 computed per seed then averaged.

Longer training gained **+0.0146 F1**. It came exactly where predicted: recall
rose from 0.458 to 0.490 while precision fell slightly from 0.756 to 0.730. The
model was still learning to fire more often when the first run cut it off.

Published BiLSTM + fastText is 0.60. **We are at 0.5847**, within the acceptable
reproduction range and within 0.015 of the published number.

## KEY FINDING: our precision and recall confirm the swapped-argument bug

In Step 2 we found by reading the SOLD source that their evaluation passes
`(y_true=model_output, y_pred=gold)` to sklearn, which is backwards, and that
this swaps precision and recall while leaving F1 unchanged. We predicted their
published precision and recall were reversed.

| | Precision | Recall | F1 |
|---|---|---|---|
| Our BiLSTM, run 1 | 0.7558 | 0.4581 | 0.5701 |
| **Our BiLSTM, run 2** | **0.7299** | **0.4899** | 0.5847 |
| Published, as printed | 0.48 | 0.74 | 0.60 |
| Published, **un-swapped** | **0.74** | **0.48** | 0.60 |

Our independent reproduction lands almost exactly on their un-swapped values,
and the match got **closer** after longer training: 0.730 against 0.74, and
0.490 against 0.48. Both within 0.01.

**We now have two independent lines of evidence for the same claim**: one from
reading their code, one from reproducing their model and landing in the
mirror-image position. This belongs in the paper as a correction for anyone else
using this dataset.

It also tells us something practical: our model is not broken and is not
behaving differently from theirs. It is a high-precision, low-recall model,
exactly as theirs was. The remaining 0.03 gap is not an architecture problem.

### Where the remaining 0.03 comes from

**1. We train on 6,000 tweets, they trained on 7,500.** We carved 1,500 off for
validation; they almost certainly used the full official split. That is 20% less
data. Fix: choose hyperparameters on validation, then rebuild on the full 7,500
with those settings and score test once - the same protocol we used for the word
list.

**2. FIXED. The models were stopping too early.** In run 1, seed 4 hit the
30-epoch limit with its best score at epoch 30, still improving. Validation F1
is noisy here, swinging eight points between adjacent epochs, so `patience=5`
killed runs that were still trending up. Run 2 with `--epochs 60 --patience 12`
recovered +0.0146 F1. Best epochs in run 2 were 30, 30, 13, 42, 16 - so the
longer budget was genuinely used.

**3. FIXED. Recall was still climbing.** Confirmed: recall rose 0.458 -> 0.490
between runs while precision fell only 0.756 -> 0.730. Recall remains the weak
side and is the main target for Phase 2.

### Parameter count - the efficiency claim

| | Count |
|---|---|
| Total | 8,724,458 |
| **Trainable** | **187,658** |
| Frozen (fastText matrix) | 8,536,800 |

Against XLM-R-large at roughly 560,000,000 parameters:

- **64x smaller** by total parameters
- **~3,000x smaller** by trainable parameters

Report both, and state clearly that the embedding matrix is a frozen lookup
table rather than a learned component. The honesty costs nothing and the numbers
are still striking.

### Known issue: speed

873 seconds per seed is 73 minutes for five seeds. Too slow for Phase 2, where
dozens of ablation configurations each need five seeds. MPS is often slower than
CPU for small LSTMs because of kernel-launch overhead, and CRF decoding does not
parallelise well.

To try, in order: `--batch-size 64`, then forcing CPU to compare, then Colab's
free T4 GPU. Owner: Person 5. A 3x speedup pays for itself many times over in
Phase 2.

## Step 5: seeds, stability, and freezing the configuration

### Seed variance is real and must be reported

| Seed | Best epoch | Test F1 |
|---|---|---|
| 1 | 30 | 0.6023 |
| 2 | 30 | 0.5828 |
| 3 | 13 | **0.5545** |
| 4 | 42 | 0.6019 |
| 5 | 16 | 0.5823 |

Mean 0.5847, standard deviation 0.0195. The spread from best to worst is
**0.048 F1**, which is larger than most of the improvements we hope to measure
in Phase 2.

**This is why five seeds are mandatory.** A single run of this model could
report anything from 0.55 to 0.60. If we ran one seed of a Phase 2 component and
got 0.60, we would have no idea whether the component helped or whether we got a
lucky seed. Every number in the paper is a mean over 5 seeds with its standard
deviation printed next to it.

Seed 3 is the outlier. It peaked at epoch 13 and never recovered across the next
12 epochs, so it stopped early. It is a genuine bad initialisation, not a bug.
**We keep it.** Dropping an inconvenient seed is exactly the kind of quiet
dishonesty that gets papers rejected on correctness.

### Configuration frozen

Everything below is fixed. Phase 2 changes one thing at a time against it.

| Setting | Value |
|---|---|
| Architecture | Embedding -> BiLSTM(64, bidirectional) -> Linear -> CRF |
| Embeddings | 300d fastText `cc.si.300`, frozen |
| Vocabulary | 28,456 words, `min_freq=1`, train-part only |
| Truncation | none; dynamic padding to the batch maximum |
| Loss | plain CRF negative log likelihood, no class weights |
| Optimiser | Adam, lr 1e-3, batch 32, dropout 0.5, grad clip 5.0 |
| Epochs / patience | 60 / 12 |
| Model selection | best validation offensive-class F1 |
| Seeds | 1, 2, 3, 4, 5 |
| **Test offensive F1** | **0.5847 +/- 0.0195** |

Tag this commit. Every Phase 2 result is reported as a difference from this row.

### OUTSTANDING: refit on the full training split

We train on 6,000 tweets. The published baseline almost certainly used all
7,500. That is 20% less data and is the most likely remaining source of our
0.015 gap.

The fix is the protocol we already used for the word list. Hyperparameters are
now chosen, so rebuild on the full 7,500 and score test once.

The one complication: with no validation split left, there is nothing to early
stop on. Use a **fixed epoch budget** taken from the validation runs instead.
Best epochs were 30, 30, 13, 42, 16, giving a mean of 26 and a median of 30.
**Train for 30 epochs, no early stopping, 5 seeds.**

This is legitimate because the budget comes from validation, not from test. Run
it, report whatever it gives, and do not tune further.

## BLOCKING PROBLEM FOR PHASE 2: training is too slow

| Run | Time per seed | Time for 5 seeds |
|---|---|---|
| Run 1 (30 epochs max) | 873s | 73 minutes |
| Run 2 (60 epochs max) | 2,128s | **3 hours** |

Phase 2 has four components, each needing an ablation, plus hyperparameter
sweeps. That is dozens of configurations. At 3 hours each, a single ablation
table costs a week.

**This must be fixed before Phase 2 starts.** Things to try, in order:

1. `--batch-size 64` or 128. Cheapest possible win.
2. Force CPU and compare against MPS. Apple MPS is often *slower* than CPU for
   small LSTMs because of kernel-launch overhead, and CRF decoding is sequential
   and does not parallelise well.
3. Google Colab free T4 GPU.
4. Profile whether CRF decoding dominates. If it does, use `--no-crf` for the
   ablation sweeps and only add the CRF back for final reported runs.

Owner: Person 5. Target: under 10 minutes for 5 seeds. Report the measured
numbers - they also feed the computational analysis section of the paper.

## Step 5b: full-train refit — PHASE 1 CLOSED

Steps 4 and 5 trained on 6,000 tweets because 1,500 were held out for choosing
hyperparameters. Once those choices were frozen, validation had no job left, so
we folded it back in and retrained on all 7,500 - the same protocol that
produced our word-list result.

With no validation split there is nothing to early stop on, so we used a fixed
budget of **30 epochs**, the median best-epoch from the Step 5 validation runs
(30, 30, 13, 42, 16). The budget came from validation, never from test.

### Final result

| | Mean | Std |
|---|---|---|
| Test offensive precision | 0.7452 | 0.0161 |
| Test offensive recall | 0.4979 | 0.0211 |
| **Test offensive F1** | **0.5965** | **0.0103** |

Per-seed: 0.5884, 0.6116, 0.5999, 0.5966, 0.5858. Five seeds, 30 fixed epochs,
1,468s per seed on Apple MPS.

### The reproduction is essentially exact

| | Precision | Recall | F1 |
|---|---|---|---|
| **Our full-train refit** | **0.7452** | **0.4979** | **0.5965** |
| Published, as printed | 0.48 | 0.74 | 0.60 |
| Published, **un-swapped** | **0.74** | **0.48** | **0.60** |

All three numbers match. F1 differs by 0.0035, precision by 0.005, recall by
0.018. This is the strongest form of the swapped-argument evidence: we did not
merely land in the same *regime* as their model, we landed on their exact
numbers once un-swapped.

### Two secondary findings

**More data helped: +0.0118 F1** (0.5847 with 6,000 tweets, 0.5965 with 7,500).
The gain came almost entirely from recall, 0.490 -> 0.498, while precision held
at roughly 0.73-0.75.

**More data also made training more stable.** Standard deviation dropped from
0.0195 to 0.0103, close to half. With 6,000 tweets the seed spread was 0.048;
with 7,500 it is 0.026. Worth one sentence in the paper: on a small
low-resource dataset, the last 25% of training data buys stability as much as
accuracy.

### The vocabulary grew

Rebuilding from the full split gave 33,006 words instead of 28,456. Vector
coverage fell slightly, 81.9% -> 80.8%, because the extra words are rare ones
fastText is less likely to have. Total parameters rose to 10,089,458 purely
from the larger frozen embedding table. **Trainable parameters are unchanged at
187,658.**

### FINAL PHASE 1 BASELINE — frozen, do not change

| Setting | Value |
|---|---|
| Architecture | Embedding -> BiLSTM(64, bidirectional) -> Linear -> CRF |
| Trained on | 7,500 tweets, full official train split |
| Epochs | 30 fixed, budget from validation |
| Batch size | 32 |
| Vocabulary | 33,006 words, `min_freq=1`, from full train |
| Embeddings | 300d fastText, frozen, 80.8% real vectors |
| Loss | plain CRF negative log likelihood |
| Optimiser | Adam, lr 1e-3, dropout 0.5, grad clip 5.0 |
| Seeds | 1, 2, 3, 4, 5 |
| **Test offensive F1** | **0.5965 +/- 0.0103** |
| Parameters | 10,089,458 total / 187,658 trainable |
| Time | 1,468s per seed |

Tag this commit. Every Phase 2 result is reported as a change from this row.

### Where Phase 1 leaves us

| Model | F1 | Precision | Recall |
|---|---|---|---|
| Our BiLSTM baseline | 0.5965 | 0.745 | 0.498 |
| **Our word list** | **0.6521** | 0.636 | 0.669 |
| Published XLM-R | 0.72 | 0.68 | 0.76 |

Read the precision and recall columns. Our model is **accurate but timid** - when
it flags a word it is usually right, but it stays silent far too often. The word
list fires more freely and wins on F1 despite being less accurate. XLM-R has
both.

**Recall is the gap.** Phase 2 must raise recall without giving away the
precision we already have.

## Phase 2, Step 0: moved to Colab GPU — COMPLETE

| batch | CRF | per epoch | 5 seeds x 30 ep | vs laptop |
|---|---|---|---|---|
| 32 | on | 19.2s | 48 min | 2.6x |
| 32 | off | 3.3s | 8 min | 15.0x |
| 64 | on | 11.0s | 28 min | 4.4x |
| 64 | off | 2.4s | 6 min | 20.8x |
| 128 | on | 7.1s | 18 min | 6.9x |
| 128 | off | 1.8s | 5 min | 26.8x |

**The CRF is the bottleneck, not the batch size.** Turning it off is worth 6-8x
at every batch size; batch size alone is worth 2-3x.

**Verification passed.** Colab seed 1 gave 0.6052 against the laptop's 0.6023, a
difference of 0.0029. That is floating-point variation between hardware, not a
real difference. Phase 2 numbers are comparable to Phase 1.

**Batch 64 slightly hurt.** Colab seeds 1-3 at batch 64 gave 0.5722 against the
laptop's 0.5799 for the same seeds at batch 32. Inside the noise, but recall
dropped 0.49 -> 0.46 and there is no upside.

**Decisions.**
- **Reported results: batch 32, CRF on.** 48 minutes for 5 seeds is workable and
  keeps us identical to the frozen Phase 1 configuration.
- **Hyperparameter sweeps: batch 128, CRF off.** 5 minutes for 5 seeds. Validate
  once that the no-CRF ranking agrees with the with-CRF ranking on the top two
  or three settings, then trust it.

## Phase 2, Step 6: subword tokenizer — COMPLETE

Swept unigram and BPE at vocab 1k-32k, trained on **train-part text only**.

### Corpus ceiling

Unigram caps at **20,832 pieces**; requests of 24,000 and 32,000 produce the
identical model. BPE reaches the full 32,000 and peaks at 24,000 (79.8% root
sharing, falling to 75.0% at 32,000). So both ends of the curve are explored and
there is no need to sweep higher.

### Root sharing

The metric: an unseen test word contains a piece of >=3 codepoints that also
occurs inside some training word, so the model can transfer what it learned.

| vocab | unigram | bpe |
|---|---|---|
| 1,000 | 44.7% | 45.0% |
| 4,000 | 62.1% | 66.9% |
| 8,000 | 69.4% | 74.9% |
| 16,000 | 78.9% | 78.5% |
| 24,000 | **82.7%** (at 20,832 cap) | **79.8%** |
| 32,000 | — | 75.0% |

## KEY FINDING: BPE respects Sinhala grapheme boundaries better than unigram

Sinhala writes a syllable as a base consonant plus **combining marks** - vowel
signs and the hal kirima. These are separate Unicode codepoints but not separate
letters. `සෙ` is ONE written letter: `ස` plus the combining sign `ෙ`.

A tokenizer piece that **begins with a combining mark** was cut out of the middle
of a written letter. It cannot correspond to any morpheme.

| vocab | unigram broken | bpe broken |
|---|---|---|
| 1,000 | 13.6% | 13.9% |
| 2,000 | 12.5% | **12.0%** |
| 4,000 | 12.0% | **10.2%** |
| 8,000 | 11.5% | **8.8%** |
| 16,000 | 11.4% | **8.0%** |
| 24,000 | 11.0% | **7.9%** |
| 32,000 | — | **7.7%** |

**BPE wins at every size from 2,000 up, and the gap widens.** Unigram plateaus
around 11% while BPE keeps falling to 7.7%. Six matched comparisons pointing the
same way.

Concretely, at vocab 8,000:

```
ටික්   unigram  ->  ටික | ්      the hal kirima stripped loose - broken
ටික්   bpe      ->  ටි | ක්      grapheme-correct
```

**No standard tokenizer metric reveals this.** Pieces-per-word is 1.36 for
unigram and 1.38 for BPE - effectively identical. Only a script-aware check
finds it.

This is a small but genuine contribution: for low-resource languages with
complex scripts, tokenizer evaluation needs a grapheme-integrity measure
alongside fragmentation counts.

### The head-to-head at the top

| | root shared | broken | pcs/word |
|---|---|---|---|
| unigram 20,832 | **82.7%** | 11.0% | 1.18 |
| bpe 24,000 | 79.8% | **7.9%** | 1.17 |

Unigram wins root-sharing by 2.9 points; BPE wins grapheme integrity by 3.1.
We weight grapheme integrity more heavily, because the pieces unigram loses on
are linguistically impossible - a fragment beginning with `්` cannot be a
morpheme. But this is a judgement, so we test it (see setting 4 below).

### Offensive words are kept whole at every setting

`සක්කිලි`, `පොන්න`, `පුක`, `වේසි` and the rest are single pieces even at vocab
1,000. So the subword channel adds capacity **without disturbing detection that
already works**. Every gain must come from unseen morphological variants
reaching a shared root piece - which also bounds how much this component can buy.

### Two caveats recorded

`MIN_ROOT_LEN` counts **codepoints, not visible letters**. `සක්කිලි` is 7
codepoints but 4 letters, so the 3-codepoint threshold is really about 2 letters
and the root-sharing figures are more generous than they look.

The test split contains junk tokens: `┓┓┓┓┓┃` appears 7 times, `62ක්` 4 times.
Left in, since it is the official split and every published baseline saw it too.
Worth one line in the limitations section.

### Two measurements we tried and discarded

Recorded so nobody repeats them or cites the meaningless number.

1. **"What fraction of an unseen word's pieces were seen in training?"** Saturated
   at 99.7-99.9% for all settings. Pieces bottom out at single characters and
   Sinhala's character inventory is small, so it was measuring "have we seen
   these characters", which is trivially yes.
2. **Root-sharing tested on the most COMMON training words.** Those are exactly
   the words SentencePiece keeps whole, so the check could only ever show what it
   showed. Both are replaced by measurements on the unseen test words.

### Experiment design for Piece 1

Four settings. Three vary size with the algorithm held constant; the fourth is
size-matched against the second to isolate the algorithm.

| # | Setting | Words kept whole | Question it answers |
|---|---|---|---|
| 1 | bpe 24,000 | 84.7% | does the near-word-level setting win? |
| 2 | bpe 8,000 | 73.5% | is the middle better? |
| 3 | bpe 2,000 | 55.3% | does real decomposition help? |
| 4 | unigram 8,000 | 76.0% | **size-matched control vs #2 - does grapheme breaking actually cost F1?** |

Setting 4 is what turns the grapheme finding from an observation into a result.

Validation F1 decides. Everything in this section is a proxy, not F1.

## Phase 2, Step 7: subword channel — MAJOR RESULT

Added a second input channel to the frozen Phase 1 model. Nothing removed:
300 fastText numbers (frozen) concatenated with 100 subword numbers (learned),
so the BiLSTM input grows 300 -> 400. Everything above that is unchanged.

### Result

| | Precision | Recall | **F1** | Std |
|---|---|---|---|---|
| Phase 1 baseline (word only) | 0.7452 | 0.4979 | 0.5965 | 0.0103 |
| **+ subword (bpe_2000)** | **0.7463** | **0.6672** | **0.7043** | **0.0052** |
| change | **+0.0011** | **+0.1693** | **+0.1078** | |

Five seeds, batch 32, CRF on - the frozen Phase 1 settings, so the number is
directly comparable. Per-seed test F1: 0.6992, 0.7011, 0.7016, 0.7088, 0.7109.

### Where this puts us

| Model | F1 | Trainable params |
|---|---|---|
| Published BiLSTM + CBOW | 0.58 | - |
| Published BiLSTM + fastText | 0.60 | - |
| Published SinBERT | 0.62 | ~110M |
| Our word list | 0.6521 | 0 |
| Published XLM-T | 0.70 | ~270M |
| **Ours** | **0.7043** | **379,658** |
| Published XLM-R | 0.72 | ~560M |
| Published XLM-R + TSD transfer | 0.73 | ~560M |

Above XLM-T, within 0.016 of XLM-R, with roughly **1,475x fewer trainable
parameters**.

## KEY FINDING: the gain is precision-neutral and almost entirely recall

Precision moved by 0.0011. Recall gained 0.1693.

This is the predicted mechanism, confirmed. Phase 1 left the model accurate but
timid - when it fired it was usually right, but it stayed silent far too often.
We hypothesised that a large share of what it missed was unseen morphological
forms, invisible to a word-level model, and that subwords would let it fire on
those forms **without** loosening its decision threshold.

A recall gain of this size at constant precision is what that mechanism looks
like. It is not what a generic capacity increase or a threshold shift produces.

Supporting evidence that this is real rather than a lucky draw:
- validation-to-test gap is only **-0.0024**; validation predicted test almost exactly
- seed variance **halved**, 0.0103 -> 0.0052

## KEY FINDING: our tokenizer-quality proxy anti-correlated with F1

Step 6 ranked the four settings by root sharing. Validation F1 reversed it.

| Setting | root sharing (Step 6) | words kept whole | val F1 | trainable |
|---|---|---|---|---|
| bpe_24000 | **79.8%** (ranked 1st) | 84.7% | **0.6783** (last) | 1,479,650 |
| bpe_8000 | 74.9% | 73.5% | 0.7027 | 679,650 |
| unigram_8000 | 69.4% | 76.0% | 0.7030 | 679,650 |
| bpe_2000 | 56.4% (ranked last) | 55.3% | **0.7060** (1st) | 379,650 |

The setting the proxy ranked first came last, by 0.028 - five times the seed
noise. **Fragmentation predicted F1; root sharing predicted the opposite.**

The degenerate-win warning was the right instinct: at 84.7% words-whole the
subword channel is nearly a copy of the word channel, adding 1.24M parameters of
near-redundant capacity on only 6,000 training tweets.

This is a methodological finding worth reporting. Intrinsic tokenizer metrics
are not a substitute for downstream evaluation.

### CONFOUND, not yet resolved

bpe_2000 has **both** more decomposition **and** far fewer parameters (379k vs
1.48M). With 6,000 training tweets the smaller model may simply overfit less.
These cannot be separated from the current data.

**Control still owed:** run bpe_24000 with `--subword-dim` reduced so its
parameter count matches bpe_2000. If it still loses, decomposition is the cause.
If it catches up, capacity was. One run, and it turns a plausible story into a
demonstrated one.

### The grapheme control came back null

bpe_8000 0.7027 vs unigram_8000 0.7030, difference **-0.0004** against a pooled
std of 0.0036. Same vocabulary size, different algorithm.

Step 6's finding that BPE breaks fewer Sinhala graphemes (7.9% vs 11.0%) does
**not** translate into an accuracy difference. Report it as a tokenizer-quality
observation, never as an accuracy claim. Recording the null result rather than
quietly dropping the experiment.

### The curve has not turned over

bpe_2000 is the **smallest** setting tested and it won. Sweep bpe_1000 and
bpe_500 before concluding 2,000 is optimal. Step 6 measured bpe_1000 at 2.10
pieces per word (within range) and bpe_500 at 2.55 (crosses the shredding
threshold), so 1,000 is the one to try.

### The CRF may now be redundant

Sweep (no CRF, batch 128) gave validation 0.7060. Final (CRF, batch 32) gave
validation 0.7067. The CRF contributed essentially nothing once subwords were
present.

If a proper ablation confirms it, we can drop the CRF from every remaining
experiment for a 6-8x speedup, and we gain a paper row: **the CRF becomes
redundant once the input representation is good enough.** That is an
interesting claim about where the sequence structure was coming from.

### Ablations still owed for Piece 1

| Run | Question |
|---|---|
| `--pooling mean` | does piece ORDER matter, or is averaging enough? |
| `--no-word-channel` | is fastText still needed once we have subwords? |
| `--sp bpe_1000` | has the vocabulary curve turned over? |
| bpe_24000 at matched params | decomposition or capacity? |
| `--no-crf` at 5 seeds | is the CRF now redundant? |

All at 5 seeds, frozen settings.

### Alignment safety

`tests/test_subword_alignment.py` - 10 checks that run before any training.
Labels are one per word; pieces are smaller than words. If the piece tensor ever
has the wrong number of word-rows, labels shift silently, the loss still falls,
and the score is quietly terrible with nothing crashing. Test 5 verifies pieces
do not bleed across word boundaries; test 9 installs a deliberately broken
encoder and asserts a clear `RuntimeError` fires.

## Phase 2, Step 7b: ablations — PIECE 1 COMPLETE

All runs: 5 seeds, batch 32, CRF on, trained on train-part (6,000).
**Selection is on VALIDATION.** Test columns are reported but never used to choose.

| Configuration | val F1 | test F1 | trainable | pieces/word |
|---|---|---|---|---|
| Phase 1 baseline, word only | - | 0.5965 +/- 0.0103 | 187,658 | - |
| **bpe_1000, both channels** | **0.7083 +/- 0.0063** | 0.7104 +/- 0.0069 | 329,658 | 2.10 |
| bpe_1000, **subword only** | 0.7071 +/- 0.0031 | 0.7059 +/- 0.0040 | **176,058** | 2.10 |
| bpe_2000, both channels | 0.7067 +/- 0.0046 | 0.7043 +/- 0.0052 | 379,658 | 1.79 |
| bpe_2000, no CRF | 0.7060 +/- 0.0024 | 0.7064 +/- 0.0036 | 379,650 | 1.79 |
| unigram_8000 (sweep) | 0.7030 +/- 0.0028 | - | 679,650 | 1.36 |
| bpe_8000 (sweep) | 0.7027 +/- 0.0044 | - | 679,650 | 1.38 |
| bpe_24000, piece_dim 50 (sweep) | 0.6783 +/- 0.0005 | - | 1,479,650 | 1.17 |
| bpe_24000, **piece_dim 4** (control) | 0.6380 +/- 0.0044 | 0.6321 +/- 0.0090 | 357,258 | 1.17 |

### Tokenizer choice: bpe_1000

bpe_1000 and bpe_2000 are **statistically indistinguishable on validation**
(0.7083 vs 0.7067, difference 0.0016 against standard deviations of 0.005-0.006).
We select bpe_1000 for its marginally higher validation mean and smaller
parameter count, not because it "won".

The curve has flattened: 2000 -> 1000 gained only +0.0016. Step 6 measured
bpe_500 at 2.55 pieces per word, past the shredding threshold, so we stop here.

## KEY FINDING: fragmentation predicts F1; parameter count does not

The capacity/decomposition confound is **resolved**, from two directions.

**More capacity did not help.** bpe_24000 with 1,240,800 subword parameters
scored 0.6783. bpe_1000 with 90,800 scored 0.7083. Thirteen times the capacity,
0.030 worse.

**Matched capacity did not rescue it.** At 118,400 subword parameters against
bpe_1000's 90,800, bpe_24000 scored 0.6380 - a gap of 0.070, roughly fifteen
times the seed noise.

Across all six configurations, F1 tracks pieces-per-word monotonically and
shows no relationship to parameter count. It is **decomposition** that helps,
not extra capacity.

*Caveat, recorded:* `piece_dim=4` cost bpe_24000 about 0.04 on its own
(0.6783 -> 0.6380), so the matched control is imperfect. It errs toward making
bpe_24000 look worse. The unmatched comparison already settles the question, so
the conclusion holds either way.

## KEY FINDING: fastText can be removed almost for free

| | val F1 | trainable | pretrained resources needed |
|---|---|---|---|
| bpe_1000, both channels | 0.7083 +/- 0.0063 | 329,658 | fastText, 600 MB |
| bpe_1000, **subword only** | 0.7071 +/- 0.0031 | **176,058** | **none** |
| difference | **-0.0012** | -47% | |

Removing the entire 300-dimensional fastText channel costs **0.0012 on
validation**, far inside the noise.

The subword-only model needs **no pretrained embeddings of any kind**. No
600 MB vector file, no 8.5M frozen parameters. It is a completely
self-contained 176,058-parameter system reaching 0.706 against XLM-R's 0.72 -
roughly **3,180x fewer parameters** than XLM-R-large.

We therefore report TWO models: the best (both channels) and the minimal
(subword only, no pretrained resources whatsoever). The second is arguably the
more interesting contribution for deployment.

## The CRF has become redundant

| | test F1 |
|---|---|
| bpe_2000 with CRF | 0.7043 +/- 0.0052 |
| bpe_2000 without CRF | 0.7064 +/- 0.0036 |

Difference +0.0021 against a pooled std of 0.0044. Inside the noise.

In Phase 1 the CRF was doing real work. Once the input representation is good
enough, the sequence structure it supplied is already present. We keep the CRF
for the reported models to stay identical to the frozen Phase 1 architecture,
but the redundancy is a finding worth a paragraph: **it says where the sequence
structure was coming from.**

### Correction to the Step 0 speed conclusion

Step 0 found the CRF cost 6-8x. That was measured **before** the subword channel
existed and no longer applies. Per-epoch times now:

| run | s/epoch |
|---|---|
| bpe_2000 + CRF | 48.7 |
| bpe_2000 no CRF | 57.2 |
| bpe_1000 + CRF | 72.0 |

The **subword encoder is now the bottleneck**, not the CRF. Its cost scales with
pieces-per-word, so bpe_1000 (2.10) is slower than bpe_2000 (1.79). Removing the
CRF does not speed anything up.

### The grapheme control came back null

bpe_8000 0.7027 vs unigram_8000 0.7030, difference **-0.0004** against a pooled
std of 0.0036. Same vocabulary size, different algorithm.

Step 6's finding that BPE breaks fewer Sinhala graphemes (7.9% vs 11.0%) does
**not** translate into an accuracy difference. Reported as a tokenizer-quality
observation, never as an accuracy claim. Recording the null rather than quietly
dropping the experiment.

### Still owed

`--pooling mean` on bpe_1000: does piece ORDER matter, or is averaging enough?
It may also be much faster, since the per-word BiLSTM is now the bottleneck.

### A note on the comparison being conservative

The subword models train on **6,000** tweets; the Phase 1 headline of 0.5965 came
from the full-train refit on **7,500**. So +0.1139 understates the effect.
Against the matched 6,000-tweet baseline of 0.5847 the gain is **+0.1236**.

The final model is a full-train refit and should gain further - Phase 1 gained
+0.0118 from the same extra 25% of data.

## Phase 2, Step 8: FINAL MODEL — PIECE 1 COMPLETE

Full-train refit. The tokenizer, word vocabulary and embedding matrix were all
rebuilt on the full 7,500-tweet split. No early stopping (no validation set is
left), so a **fixed 35 epochs** was used - the median best-epoch from the
bpe_1000 validation runs (30, 30, 41, 35, 37). That budget came from validation,
never from test.

Two models trained, 5 seeds each, test scored once.

| Model | F1 | std | P | R | Trainable | Needs pretrained? |
|---|---|---|---|---|---|---|
| Published BiLSTM + CBOW | 0.58 | | | | | |
| Published BiLSTM + fastText | 0.60 | | 0.74 | 0.48 | | |
| Our Phase 1 baseline | 0.5965 | 0.0103 | 0.7452 | 0.4979 | 187,658 | fastText |
| Published SinBERT | 0.62 | | | | ~110M | yes |
| Our word list | 0.6521 | | 0.6361 | 0.6689 | 0 | no |
| *ours, mean pooling (ablation)* | *0.6822* | *0.0092* | *0.7931* | *0.5991* | *293,958* | |
| Published XLM-T | 0.70 | | 0.64 | 0.77 | ~270M | yes |
| **OURS: word + subword** | **0.7002** | 0.0034 | 0.7284 | 0.6751 | 329,658 | fastText |
| **OURS: subword only** | **0.7079** | 0.0066 | 0.7426 | 0.6771 | **176,058** | **none** |
| Published XLM-R | 0.72 | | 0.68 | 0.76 | ~560M | yes |
| Published XLM-R + TSD transfer | 0.73 | | | | ~560M | yes |

Minimal model per-seed: 0.7123, 0.7062, 0.7170, 0.7008, 0.7034. 825s per seed
on a Colab T4.

## KEY FINDING: the best model needs no pretrained embeddings at all

The **subword-only** model - no fastText, no PLM, nothing downloaded - scored
**0.7079** against the full model's 0.7002.

Removing the entire 300-dimensional fastText channel did not merely cost
nothing; it scored slightly higher. The gap is about 1.5 pooled standard
deviations, so the honest phrasing is **indistinguishable or marginally
better**, not "better". Either way the conclusion holds: fastText is not
carrying the result.

That makes the minimal model our strongest claim:

- **176,058 trainable parameters**
- **no pretrained embeddings of any kind** - no 600 MB vector file, no frozen
  embedding table, nothing to download
- **0.7079 F1**, above published XLM-T (0.70) and within 0.012 of XLM-R (0.72)
- **3,181x fewer trainable parameters** than XLM-R-large

For deployment this is the interesting model. It is fully self-contained.

## KEY FINDING: piece ORDER matters — mean pooling loses

The last owed ablation, answered.

| Pooling | F1 | P | R |
|---|---|---|---|
| BiLSTM over pieces | 0.7002 | 0.7284 | 0.6751 |
| Mean over pieces | **0.6822** | **0.7931** | **0.5991** |

A drop of 0.018 at roughly 2.6 pooled standard deviations. Real, not noise.

**How it fails is the interesting part.** Mean pooling raises precision to 0.7931
and drops recall to 0.5991 - it reverts toward the timid Phase 1 behaviour.
Averaging the pieces destroys the information about WHERE in the word each piece
sits.

This supports the agglutinative argument directly: in Sinhala a word's ENDING
carries grammatical information distinct from its stem, and a representation that
cannot tell a stem from a suffix loses exactly the signal that lets the model
recognise an unseen variant of a familiar root.

## The full-train refit did NOT help this time

| Trained on | Model | test F1 |
|---|---|---|
| 6,000 (Step 7b) | bpe_1000, both channels | 0.7104 +/- 0.0069 |
| 7,500 (Step 8) | bpe_1000, both channels | 0.7002 +/- 0.0034 |

The extra 25% of data made it **worse** by 0.010. Phase 1 gained +0.0118 from
the same extra data, so this is a genuine difference in behaviour and we report
it rather than quietly using the higher number.

Most likely cause: **the fixed epoch budget**. At 6,000 tweets each seed
early-stopped at its own best epoch (30, 30, 41, 35, 37) chosen on validation.
The refit gives every seed the same 35 epochs with no per-seed selection. The
minimal model, by contrast, improved slightly with more data (0.7059 -> 0.7079),
so the effect is not uniform.

**We report the refit numbers as the headline** because they match the protocol
used for the 0.5965 baseline, which was also a full-train refit. Like-for-like
comparison matters more than the larger number.

## Piece 1 summary: what we can claim

**Result.** Adding a subword channel to a BiLSTM token classifier raises
offensive-token F1 from 0.5965 to 0.7002 with both channels, or 0.7079 with the
subword channel alone.

**Mechanism.** The gain is recall-driven at near-constant precision: recall
0.4979 -> 0.6771 while precision moves 0.7452 -> 0.7426. This was predicted in
advance from the measured 48.7% unseen-type rate, and it is not what a threshold
shift or a generic capacity increase produces.

**Cause isolated.** Fragmentation predicts F1 across six configurations;
parameter count does not, in either direction (13x more capacity scored worse;
matched capacity scored worse still). It is decomposition that helps.

**Efficiency.** 176,058 trainable parameters with no pretrained resources,
against XLM-R-large's ~560M.

**What we cannot claim.** That BPE's better Sinhala grapheme integrity improves
accuracy - that control came back null. That we match XLM-R - we do not, we are
0.012 below it. That more training data helps this architecture - it did not
here.

# PHASE 2 PLAN

## We keep the same model. We do not start again.

The BiLSTM from Step 4 stays. It is the skeleton. Phase 2 adds four pieces to
it, one at a time.

```
   Phase 1 (done)                  Phase 2 (next)
   ------------------              ------------------------------------
   words                           words
     |                               |
     |                             [1] split into subword pieces
     v                               v
   fastText vectors                fastText vectors
     |                               |
     v                               v
   BiLSTM (64)                     BiLSTM (64)              <- unchanged
     |                               |
     v                               |-----------------+
   CRF                               v                 v
     |                             CRF               [2] sentence head
     v                               |                 |
   word labels                       v                 v
                                   word labels      offensive yes/no
                                     |                 |
                                   [3] balanced loss   |
                                                    [4] learn from SemiSOLD
```

Everything in the middle column stays. We are bolting parts on, not rebuilding.

**Why one at a time.** If we add all four together and the score goes up, we
learn nothing about which piece helped. Reviewers will ask, and "we do not know"
is not an answer. Adding them one at a time gives us a table showing what each
piece is worth. That table is called an **ablation study** and it is what makes
the paper credible.

It also protects us. If we run out of time after two pieces, we still have a
complete, honest paper about those two pieces.

## The number to beat

| | F1 |
|---|---|
| Our BiLSTM baseline (Phase 1, frozen) | **0.5965 +/- 0.0103** |
| Our word list | **0.6521** |
| + subword channel, 6,000 tweets (Step 7) | 0.7043 +/- 0.0052 |
| **FINAL: subword only, full train (Step 8)** | **0.7079 +/- 0.0066**  <- current best |
| Published SinBERT | 0.62 |
| Published XLM-R | 0.72 |

A word list with no learning beats our neural model. Phase 2 has to fix that.
Beating 0.60 is not enough. **We must beat 0.6521.**

## Step 0 of Phase 2: fix the speed. Nothing else starts first.

Right now 5 seeds take 3 hours. Phase 2 needs dozens of setups, each run 5
times. At this speed one comparison table takes a week.

Try in order: `--batch-size 64`, then CPU versus MPS, then Google Colab's free
T4 GPU, then check whether CRF decoding is the bottleneck.
**Target: 5 seeds in under 10 minutes.** Owner: Person 5.

These measurements also go straight into the paper's efficiency section, so the
work is not a detour.

## Piece 1: subword input

**The problem.** Our model sees `සක්කිලි` and `සක්කිලියා` as two completely
unrelated ID numbers, even though they are the same insult with different
endings. Sinhala packs grammar into word endings, so one root appears in many
forms. **48.7% of the distinct words in our test set never appeared in
training.** The model has nothing at all for those.

**The fix.** Break each word into smaller pieces before the model sees it, using
a tool called SentencePiece. Then a word the model has never met can still be
understood, because it shares pieces with words it knows.

**Why we believe it will work.** Three pieces of evidence, all measured by us in
Step 3: the 48.7% unseen rate; fastText's nearest neighbours in Sinhala are
almost all morphological variants rather than synonyms; and fastText already
beats word2vec on Sinhala precisely because it uses subword information.

**Watch out for.** Our labels are one per *word*, but subwords are smaller than
words. We must recombine the pieces back into one vector per word before the
label layer. Decide the rule once, write it down, never change it.

**Tune on validation:** subword vocabulary size (try 2k, 4k, 8k, 16k).

## Piece 2: joint sentence + token head

**The idea.** Right now the model has one job: label each word. We add a second
job: say whether the whole tweet is offensive. Both jobs share the same BiLSTM
underneath.

**Why it helps.** The two jobs are two views of the same thing. Forcing one
BiLSTM to be good at both makes its internal representation better. This is
especially useful for us because offensive word labels are rare (4%) and we only
have 6,000 training tweets.

**The real reason it matters.** This head is the docking port for Piece 4. Our
extra 145,000 tweets only have *sentence-level* scores. Without a sentence head
we cannot use them at all. So Pieces 2 and 4 are one idea, not two.

**Tune on validation:** lambda, the balance between the two jobs.

## Piece 3: balanced loss

**The problem.** Only 4 in every 100 tokens are offensive. A model can say "not
offensive" to everything, be 96% correct, and be useless. We already saw a
version of this collapse during testing.

**The fix.** Change the loss function so rare offensive tokens count for more.
Four options to compare:

| Option | What it does |
|---|---|
| Plain cross-entropy | The control. What the baseline uses now. |
| Weighted cross-entropy | Multiply the offensive class's loss by a constant. Simple, often competitive. |
| Focal loss | Ignore examples it already gets right, focus on hard ones. |
| Dice loss | Optimise something F1-like directly. Built for imbalanced NLP. Public code exists. |

**Watch out for.** Our model is precision 0.73, recall 0.49 — it is accurate but
timid. These losses push recall up. If we push too hard we could end up at
recall 0.90 and precision 0.35, which has a **worse** F1 than we started with.
Always report both columns.

## Piece 4: offline distillation from SemiSOLD

**What we have.** 145,000 extra Sinhala tweets with no human labels, but with
scores already saved from 11 earlier models.

**What we do.** Our small model learns to copy those saved scores. Like a
student studying an answer key without ever meeting the teacher.

**Why this does not break the no-PLM rule.** The teacher models were run by the
dataset authors in 2022. Their answers are static numbers in a public file. No
pretrained language model is ever loaded, run, or trained inside our network. We
are consuming a published file, exactly as we consume the gold labels.

**The catch, and the design consequence.** Those 145,000 scores are
**sentence-level only**. There are no token labels. So distillation cannot teach
our word-labelling head directly. It teaches the *sentence* head from Piece 2,
and because both heads sit on one shared BiLSTM, a better sentence head pulls
the shared part in a better direction, which helps the word labels.

**Write that sentence in the paper.** It is the spine of our contribution:
multi-task learning is what makes distillation reachable at token level.

**Use the authors' own findings.** They filtered these extra tweets by model
uncertainty and found a threshold of 0.1 best (about 8,474 tweets). A looser
0.15 added 47,746 tweets but made results *worse* from the noise. They also
found lightweight models gain most: BiLSTM+CBOW gained +2.78% while XLM-R gained
only +0.63%. Our model is the lightweight kind. That is where the headroom is.

**Do this last.** It is the fiddliest piece. If it stalls, report it as a
negative result and move on. A careful negative result is a legitimate finding.

## Order of work

| Order | Piece | Why this position |
|---|---|---|
| 0 | Fix speed | Everything else depends on it |
| 1 | Subword | Cheapest, best evidence, most likely to work |
| 2 | Multi-task | Needed before distillation is possible |
| 3 | Balanced loss | Independent, can be done in parallel |
| 4 | Distillation | Hardest, most likely to fail, do last |

After each piece: run 5 seeds, record mean and standard deviation, append to
`results/results.csv`, compare against 0.5965.

## What success looks like

| Outcome | Result | Verdict |
|---|---|---|
| Floor | Beat 0.6521 clearly | Solid paper |
| Target | 0.65 - 0.70 | Strong paper |
| Stretch | Match or beat 0.72 | Very strong |
| Negative | A piece does not help | **Still report it** |

For context, 36 teams did this same task in **English** at SemEval-2021 with far
more data, and the winner got about 0.68. This task is hard everywhere. Modest,
well-evidenced gains are normal, not a disappointment.

## The rule that decides whether this gets published

Every number is a mean over 5 seeds with its standard deviation next to it.
Our seed spread in Phase 1 was **0.026** even with the full training split -
wider than most improvements we are chasing. One run proves nothing.

Every hyperparameter is chosen on validation. Test is opened once per
configuration we intend to publish.

Every claim in the paper points to a number in a table. If it does not, we
soften it or delete it.

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
│   ├── embeddings.py           vocabulary and fastText loading
│   ├── subword.py              SentencePiece training and diagnostics
│   ├── dataset.py              padding, masking, batching
│   ├── model.py                BiLSTM + CRF
│   └── train.py                training loop, seeding, early stopping
├── notebooks/
│   ├── 01_data_exploration.py  Step 1 checks
│   ├── 02_metric_check.py      Step 2 baselines
│   ├── 03_embeddings.py        Step 3 vocabulary and vectors
│   ├── 04_baseline.py          Step 4 BiLSTM baseline
│   ├── 05_full_train_refit.py  Step 5b full-train refit
│   ├── 06_subword_tokenizer.py Step 6 subword tokenizer sweep
│   ├── 07_subword_model.py     Step 7 subword model
│   └── 08_final_model.py       Step 8 final full-train refit
├── tests/
│   ├── test_metrics.py         9 unit tests, must always pass
│   └── test_subword_alignment.py 10 alignment tests, run before training
└── results/
    ├── results.csv             every run we have ever done
    ├── step1_report.txt        output of the exploration script
    ├── step2_report.txt        output of the metric check
    ├── step3_report.txt        output of the embedding check
    └── step4_report.txt        output of the baseline run

Not committed (see .gitignore):
  embeddings/   cc.si.300.vec.gz, several hundred MB
  artifacts/    embedding_matrix.npy and vocab.txt, regenerable
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

# Step 3 needs the Sinhala fastText vectors first:
mkdir -p embeddings && cd embeddings
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.si.300.vec.gz
cd ..
python notebooks/03_embeddings.py > results/step3_report.txt

# Step 4 - run in this order, do not skip to the last one
pip install torch pytorch-crf
python notebooks/04_baseline.py --no-crf --seeds 1 --epochs 10 --tag smoke
python notebooks/04_baseline.py --seeds 1 --epochs 10 --tag crf
python notebooks/04_baseline.py --epochs 60 --patience 12 > results/step4_report.txt
```

Run everything from the project root, not from inside `notebooks/`.
Do not unzip the vector file. The loader reads `.gz` directly.

## Progress

- [x] **Step 0** Repository set up
- [x] **Step 1** Data loaded and verified. All checks passed.
- [x] **Step 2** Metric copied from the official code. 9 unit tests pass.
      Validation split created. Baselines run honestly.
      **Word list = 0.6521. This is the floor our model must beat.**
- [x] **Step 3** fastText loaded. 28,456-word vocabulary, 81.9% real vectors.
      Morphological evidence for the subword component found.
- [x] **Step 4** BiLSTM + CRF baseline reproduced.
- [x] **Step 5** Re-run with `--epochs 60 --patience 12`.
      **0.5847 +/- 0.0195** over 5 seeds. Config frozen.
- [x] **Step 5b** Full-train refit. **0.5965 +/- 0.0103.** Reproduction is
      essentially exact against the published 0.60.

**PHASE 1 COMPLETE.**

Phase 2:
- [x] **Step 0** Colab GPU. CRF is the bottleneck (6-8x), not batch size.
      Verified: Colab 0.6052 vs laptop 0.6023. Reported runs use batch 32 +
      CRF; sweeps use batch 128, no CRF.
- [x] **Step 6** Subword tokenizer swept. **BPE beats unigram on Sinhala
      grapheme integrity at every size.** 4 settings shortlisted.
- [x] **Step 7** Subword encoder built, 10 alignment tests pass.
      **0.7043 +/- 0.0052 (+0.1078).** Above XLM-T, within 0.016 of XLM-R,
      with 379,658 trainable parameters.
- [x] **Step 7b** Ablations done. bpe_1000 selected. Capacity confound
      resolved: **fragmentation predicts F1, parameter count does not.**
      fastText removable for -0.0012. CRF now redundant.
- [x] **Step 8** FINAL MODEL. Full-train refit, both models, 5 seeds.
      **word+subword 0.7002 +/- 0.0034; subword-only 0.7079 +/- 0.0066 with
      176,058 params and NO pretrained embeddings.**
- [x] **Step 8b** `--pooling mean` ablation: **0.6822, piece order matters.**

**PIECE 1 COMPLETE.**

- [ ] **Piece 2** Joint sentence + token head
- [ ] **Piece 3** Balanced loss
- [ ] **Piece 4** Offline distillation from SemiSOLD
- [ ] **Piece 2** Joint sentence + token head
- [ ] **Piece 3** Balanced loss
- [ ] **Piece 4** Offline distillation from SemiSOLD

## Data source

Ranasinghe, T., Anuradha, I., Premasiri, D., Silva, K., Hettiarachchi, H., Uyangodage, L.,
& Zampieri, M. (2024). SOLD: Sinhala offensive language dataset. *Language Resources and
Evaluation*. https://doi.org/10.1007/s10579-024-09723-1

Dataset: https://huggingface.co/datasets/sinhala-nlp/SOLD
Code: https://github.com/Sinhala-NLP/SOLD

## Warning

This repository works with real offensive social media text. The dataset contains
language that many people will find upsetting.