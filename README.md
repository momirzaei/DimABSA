# DimABSA: Dimensional Aspect-Based Sentiment Analysis

This repository contains notebook implementations for **Dimensional Aspect-Based Sentiment Analysis (DimABSA)** on the **restaurant** and **laptop** domains. Instead of predicting only discrete sentiment labels such as positive/negative/neutral, the models predict **continuous Valence–Arousal (VA)** values:

- **Valence**: how positive or negative the sentiment is.
- **Arousal**: how emotionally intense or activated the sentiment is.

Each VA value is stored as:

```text
valence#arousal
```

Example:

```json
"VA": "7.75#7.38"
```

The project is organized around three related subtasks:

| Subtask | Name | Input | Output |
|---|---|---|---|
| Subtask 1 | DimASR: Dimensional Aspect Sentiment Regression | Text + known aspect terms | VA score for each aspect |
| Subtask 2 | DimASTE: Dimensional Aspect Sentiment Triplet Extraction | Text only | Aspect, opinion, VA triplets |
| Subtask 3 | DimASQP: Dimensional Aspect Sentiment Quadruple Prediction | Text only | Aspect, category, opinion, VA quadruplets |

The repository includes multiple implementations for each subtask. The notebooks progress from baseline models to stronger versions with better preprocessing, specialized losses, span validation, VA-aware decoding, and domain-specific tricks.

---

## Repository contents

Recommended repository structure:

```text
DimABSA/
├── README.md
├── data/
│   ├── eng_laptop_test_task1.jsonl
│   ├── eng_laptop_test_task2.jsonl
│   ├── eng_laptop_test_task3.jsonl
│   ├── eng_restaurant_test_task1.jsonl
│   ├── eng_restaurant_test_task2.jsonl
│   └── eng_restaurant_test_task3.jsonl
├── notebooks/
│   ├── evaluate_subtasks.ipynb
│   ├── subtask1/
│   │   ├── subtask1-dimasr.ipynb
│   │   ├── subtask1-dimasr2.ipynb
│   │   ├── subtask1-dimasr3.ipynb
│   │   └── subtask1-dimasr4.ipynb
│   ├── subtask2/
│   │   ├── subtask2-dimaste.ipynb
│   │   ├── subtask2-dimaste2.ipynb
│   │   └── subtask2-diamaste3.ipynb
│   └── subtask3/
│       ├── subtask3-dimasqp1.ipynb
│       ├── subtask3-disasqp2.ipynb
│       └── subtask3-disasqp3.ipynb
├── outputs/
│   ├── subtask1_outputs/
│   ├── subtask2_outputs/
│   └── subtask3_outputs/
└── checkpoints/
    ├── subtask1_checkpoints/
    ├── subtask2_checkpoints/
    └── subtask3_checkpoints/
```

The notebooks were written in a Kaggle-style workflow, so several paths point to inputs such as:

```python
/kaggle/input/dimasr/subtask1
/kaggle/input/dimaste/subtask2
/kaggle/input/dimasqp/subtask3
```

When running locally, update those paths or place the files in the expected directories.

---

## Installation

The notebooks use PyTorch and Hugging Face Transformers.

```bash
pip install torch transformers datasets accelerate sentencepiece scipy pandas numpy tqdm nltk
```

For Subtask 1 version 4, WordNet-based augmentation is used. Run this once if WordNet is not already downloaded:

```python
import nltk
nltk.download("wordnet")
```

---

## Dataset format

The provided test files follow three different formats depending on the task.

### Subtask 1: Aspect + VA

File examples:

```text
eng_laptop_test_task1.jsonl
eng_restaurant_test_task1.jsonl
```

Each record contains text and a list of known aspects with VA scores.

```json
{
  "ID": "rest26_aspect_va_test_1",
  "Text": "A friend suggested this cafe for a lunch date a while back and I cannot stay away",
  "Aspect_VA": [
    {
      "Aspect": "cafe",
      "VA": "7.12#7.12"
    }
  ]
}
```

The model does **not** need to extract the aspect in this task. It receives the aspect and predicts the VA score.

### Subtask 2: Aspect + opinion + VA triplets

File examples:

```text
eng_laptop_test_task2.jsonl
eng_restaurant_test_task2.jsonl
```

Each record contains text and triplets.

```json
{
  "ID": "rest26_aste_test_1",
  "Text": "My fiance had stewed chicken and I had the curried chicken- both were excellent",
  "Triplet": [
    {
      "Aspect": "curried chicken",
      "Opinion": "excellent",
      "VA": "8.25#8.12"
    },
    {
      "Aspect": "stewed chicken",
      "Opinion": "excellent",
      "VA": "8.25#8.12"
    }
  ]
}
```

The model must extract the aspect span, opinion span, and continuous VA score.

### Subtask 3: Aspect + category + opinion + VA quadruplets

File examples:

```text
eng_laptop_test_task3.jsonl
eng_restaurant_test_task3.jsonl
```

Each record contains text and quadruplets.

```json
{
  "ID": "lap26_asqp_test_1",
  "Text": "Granted , the battery life is fairly low , and it's big for a laptop , but this is a desktop replacement , so that was to be expected",
  "Quadruplet": [
    {
      "Aspect": "battery life",
      "Category": "BATTERY#OPERATION_PERFORMANCE",
      "Opinion": "fairly low",
      "VA": "3.75#6.00"
    },
    {
      "Aspect": "laptop",
      "Category": "LAPTOP#PORTABILITY",
      "Opinion": "big",
      "VA": "4.50#6.50"
    }
  ]
}
```

The model must extract the full quadruple, including the aspect category.

---

## Dataset summary

The uploaded test sets contain 1,000 samples per domain per subtask.

| File | Records | Annotation type | Total annotations | Avg annotations / record |
|---|---:|---|---:|---:|
| `eng_laptop_test_task1.jsonl` | 1,000 | `Aspect_VA` | 1,421 | 1.421 |
| `eng_restaurant_test_task1.jsonl` | 1,000 | `Aspect_VA` | 1,504 | 1.504 |
| `eng_laptop_test_task2.jsonl` | 1,000 | `Triplet` | 1,974 | 1.974 |
| `eng_restaurant_test_task2.jsonl` | 1,000 | `Triplet` | 2,129 | 2.129 |
| `eng_laptop_test_task3.jsonl` | 1,000 | `Quadruplet` | 1,975 | 1.975 |
| `eng_restaurant_test_task3.jsonl` | 1,000 | `Quadruplet` | 2,129 | 2.129 |

Some useful observations:

- Restaurant reviews generally contain more extracted sentiment units per review than laptop reviews.
- Subtask 2 and Subtask 3 have almost the same annotation counts because Subtask 3 adds the category field to the triplet structure.
- The laptop domain has many more category labels in Subtask 3 than the restaurant domain.
  - Laptop test set: 107 unique categories.
  - Restaurant test set: 20 unique categories.
- The test files do not contain `NULL` aspects or `NULL` opinions, so implementations that filter `NULL` values are mainly cleaning the training/dev data.

---

## Shared VA preprocessing

All subtasks use the same VA scale:

```text
VA_MIN = 1.0
VA_MAX = 9.0
```

For regression models, VA values are normalized to `[0, 1]`:

```python
value_norm = (value - 1.0) / 8.0
```

They are converted back to the original scale by:

```python
value = value_norm * 8.0 + 1.0
```

Predictions are clipped to `[1, 9]` before being written to JSONL:

```python
value = np.clip(value, 1.0, 9.0)
```

This clipping is important because invalid VA values hurt the official continuous scoring.

---

# Subtask 1: DimASR

## Goal

Subtask 1 predicts the VA score for each known aspect in a sentence.

Input:

```json
{
  "Text": "The processor was really fast especially for the price of the laptop",
  "Aspect": "processor"
}
```

Output:

```json
{
  "Aspect": "processor",
  "VA": "7.75#7.38"
}
```

The task is a **continuous regression problem**.

---

## Subtask 1 implementation 1: `subtask1-dimasr.ipynb`

This is the baseline BERT regression implementation.

### Main idea

The model uses `bert-base-uncased` as an encoder and predicts two continuous values:

```text
[valence, arousal]
```

The input is encoded as a pair:

```python
tokenizer(text, aspect)
```

This creates the usual BERT format:

```text
[CLS] text [SEP] aspect [SEP]
```

### Preprocessing

The dataset is flattened so that each aspect becomes one training sample.

Example source item:

```json
{
  "Text": "The processor was really fast especially for the price of the laptop",
  "Aspect_VA": [
    {"Aspect": "processor", "VA": "7.75#7.38"}
  ]
}
```

Becomes:

```json
{
  "text": "...",
  "aspect": "processor",
  "valence": 0.84375,
  "arousal": 0.7975
}
```

The code also handles training files that use `Quadruplet` by reading the aspect and VA fields from each quadruple.

### Model architecture

```text
BERT encoder
    ↓
[CLS] hidden state
    ↓
Linear → ReLU → Dropout
    ↓
Linear → ReLU → Dropout
    ↓
Linear → 2
    ↓
Sigmoid
```

The final `Sigmoid` constrains predictions to `[0, 1]`, matching the normalized VA targets.

### Loss function

The baseline uses Smooth L1 loss separately for valence and arousal:

```python
valence_loss = smooth_l1_loss(pred_v, true_v)
arousal_loss = smooth_l1_loss(pred_a, true_a)
loss = valence_loss + arousal_loss
```

Smooth L1 is less sensitive to outliers than plain MSE, which is useful because VA labels can vary strongly across examples.

### Training setup

Important settings:

```python
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
```

A linear warmup scheduler is used.

### Extra features

- Early stopping based on validation RMSE.
- Best model checkpoint saving.
- Denormalized RMSE calculation for validation.
- Prediction grouping back to the original JSONL format.

---

## Subtask 1 implementation 2: `subtask1-dimasr2.ipynb`

This version improves the baseline with stronger preprocessing, a deeper head, and a custom VA loss.

### What it adds compared with implementation 1

| Area | Implementation 1 | Implementation 2 |
|---|---|---|
| Input format | BERT pair: text + aspect | Aspect marked directly inside text |
| Training data handling | Reads `Aspect_VA` or `Quadruplet` | Explicitly converts `Quadruplet` to `Aspect_VA` |
| NULL handling | Minimal | Filters `NULL` aspects |
| Regression head | Medium MLP | Deeper MLP |
| Loss | Smooth L1 | Weighted MSE + Smooth L1 |
| Batch size | 32 | 64 |
| Epochs | 20 | 30 |
| LR | `2e-5` | `5e-5` |

### Better preprocessing: aspect marking

Instead of only passing the aspect as a second sequence, the aspect span is marked inside the sentence:

```text
The [ASPECT] processor [/ASPECT] was really fast especially for the price of the laptop
```

If the aspect is not found literally in the text, the code falls back to:

```text
text [SEP] [ASPECT] aspect [/ASPECT]
```

This helps the encoder focus on the exact span whose emotion should be predicted.

### Loss function

The custom loss combines MSE and Smooth L1 for both VA dimensions:

```python
v_mse = mse_loss(pred_v, true_v)
a_mse = mse_loss(pred_a, true_a)

v_smooth = smooth_l1_loss(pred_v, true_v, beta=0.1)
a_smooth = smooth_l1_loss(pred_a, true_a, beta=0.1)

loss = 1.2 * (v_mse + v_smooth) / 2 + 0.8 * (a_mse + a_smooth) / 2
```

Why this is useful:

- MSE strongly penalizes large errors.
- Smooth L1 reduces instability from noisy examples.
- Valence is weighted more heavily than arousal because valence is usually more important for sentiment polarity.

### Model change

The head is deeper:

```text
Encoder hidden
    ↓
Linear(hidden → hidden*2)
    ↓
ReLU + Dropout
    ↓
Linear(hidden*2 → hidden)
    ↓
ReLU + Dropout
    ↓
Linear(hidden → hidden/2)
    ↓
ReLU + smaller Dropout
    ↓
Linear(hidden/2 → 2)
```

This gives the model more capacity to map contextual representations to continuous VA scores.

### Evaluation improvements

The notebook computes:

- RMSE for valence.
- RMSE for arousal.
- Overall RMSE.
- MAE.
- Pearson correlation for valence and arousal.
- Prediction standard deviation to detect collapse.

Prediction standard deviation is useful because a regression model can look stable while predicting values close to the global mean for every example.

---

## Subtask 1 implementation 3: `subtask1-dimasr3.ipynb`

This version switches from BERT to DeBERTa and separates the valence and arousal heads.

### What it adds compared with implementation 2

| Area | Implementation 2 | Implementation 3 |
|---|---|---|
| Encoder | `bert-base-uncased` | `microsoft/deberta-v3-base` |
| VA prediction | Shared regression head | Separate valence and arousal heads |
| Activations | ReLU | GELU |
| Normalization | Basic MLP | LayerNorm in each head |
| Scheduler | Linear warmup | Cosine schedule with warmup |
| Optimizer LR | Same LR for all parameters | Lower LR for encoder, higher LR for heads |
| Metrics | RMSE, MAE, correlation | RMSE, MAE, PCC, training log |

### Why DeBERTa helps

DeBERTa generally gives stronger contextual representations than BERT because it uses disentangled attention and improved pretraining. In this task, that helps identify subtle emotional differences attached to aspect terms.

### Separate valence and arousal heads

The model predicts valence and arousal using two different heads:

```text
Encoder output
    ├── Valence head → valence
    └── Arousal head → arousal
```

This is useful because valence and arousal are related but not identical. For example:

- `excellent` is high valence and often high arousal.
- `calm` may be positive valence but low arousal.
- `terrible` is low valence but high arousal.

Separate heads let each dimension learn its own mapping.

### Differential learning rates

The optimizer uses lower learning rate for the pretrained encoder:

```python
encoder_lr = learning_rate * 0.1
head_lr = learning_rate
```

This prevents destroying pretrained language knowledge while still allowing the regression heads to learn quickly.

### Loss function

This version uses MSE over normalized VA:

```python
loss = mse_loss(predicted_va, target_va)
```

Because the output heads end with `Sigmoid`, predictions remain in `[0, 1]`.

### Best effect

This version produced the strongest Task 1 evaluation in the included evaluation notebook, especially for the restaurant domain.

---

## Subtask 1 implementation 4: `subtask1-dimasr4.ipynb`

This version explores domain-specific ensembling and data augmentation.

### What it adds compared with implementation 3

| Area | Implementation 3 | Implementation 4 |
|---|---|---|
| Encoder | DeBERTa | BERT |
| Training | One combined model | Multiple domain-specific models |
| Augmentation | None | Synonym replacement, random insertion, VA noise |
| Loss | MSE | Combined MSE + Focal MSE |
| Regularization | Dropout | Higher dropout + higher weight decay |
| Inference | Single model | Ensemble averaging |
| Seeds | One seed | Multiple seeds: 42, 123, 456 |

### Data augmentation

Three augmentation strategies are used.

#### 1. Synonym replacement

Random non-aspect words are replaced with WordNet synonyms.

Aspect words are protected so that the target aspect is not changed.

Example:

```text
The screen is bright
```

Could become:

```text
The display is bright
```

but if the aspect is `screen`, the word `screen` is not replaced.

#### 2. Random insertion

Small modifiers are inserted:

```text
really, quite, very, somewhat, rather, extremely, fairly, pretty, absolutely
```

This increases phrasing variety.

#### 3. VA noise / label smoothing

Small noise is added to normalized VA labels:

```python
v = clip(v + uniform(-epsilon, epsilon), 0, 1)
a = clip(a + uniform(-epsilon, epsilon), 0, 1)
```

This prevents the model from overfitting exact numeric values.

### Focal MSE loss

The notebook defines a focal-style regression loss:

```python
mse = (pred - target) ** 2
weight = (1 + mse.detach()) ** gamma
focal_mse = mean(weight * mse)
```

Then it combines normal MSE and focal MSE:

```python
loss = 0.7 * mse_loss + 0.3 * focal_mse
```

Why this helps:

- Easy examples receive normal MSE treatment.
- Hard examples with larger error receive extra weight.
- The model focuses more on examples that are poorly predicted.

### Domain-specific ensemble

The notebook trains separate models for:

- Restaurant domain.
- Laptop domain.

For each domain, it trains multiple seeds and averages predictions during inference.

This can improve robustness, but it can also overfit if the domain models are not balanced or if augmentation changes the distribution too much.

---

## Subtask 1 evaluation results

The evaluation notebook computes Pearson correlation and normalized VA RMSE.

Lower RMSE is better. Higher PCC is better.

| Prediction set | Laptop PCC_V | Laptop PCC_A | Laptop RMSE_VA | Restaurant PCC_V | Restaurant PCC_A | Restaurant RMSE_VA |
|---|---:|---:|---:|---:|---:|---:|
| Task 1 set 1 | 0.8024 | 0.5032 | 0.1311 | 0.8241 | 0.5656 | 0.1269 |
| Task 1 set 2 | 0.8079 | 0.4882 | 0.1283 | 0.8253 | 0.5541 | 0.1256 |
| Task 1 set 3 | 0.8022 | 0.4720 | **0.1253** | **0.8551** | **0.5757** | **0.1124** |
| Task 1 set 4 | 0.7937 | 0.4789 | 0.1317 | 0.8216 | 0.5582 | 0.1269 |

The strongest overall Task 1 result is prediction set 3, which corresponds to the DeBERTa-style implementation.

---

# Subtask 2: DimASTE

## Goal

Subtask 2 extracts triplets:

```text
Aspect, Opinion, VA
```

Input:

```text
The fans are loud, especially when you are on Performance Mode.
```

Output:

```json
{
  "Aspect": "fans",
  "Opinion": "loud",
  "VA": "3.00#7.33"
}
```

This is harder than Subtask 1 because the model must both extract spans and predict continuous emotion values.

---

## Subtask 2 implementation 1: `subtask2-dimaste.ipynb`

This is the baseline T5 sequence-to-sequence implementation.

### Main idea

The model converts extraction into text generation.

Input prompt:

```text
extract triplets: <review text>
```

Target sequence:

```text
[A]aspect[O]opinion[V]valence#arousal[SEP][A]aspect[O]opinion[V]valence#arousal
```

Example:

```text
[A]fans[O]loud[V]3.00#7.33
```

### Preprocessing

The training files may use `Quadruplet`, so the code converts quadruplets into triplets:

```python
Triplet = {
    "Aspect": quad["Aspect"],
    "Opinion": quad["Opinion"],
    "VA": quad["VA"]
}
```

It also filters:

```text
Aspect == "NULL"
Opinion == "NULL"
```

This is a useful trick because the test files contain explicit spans, so training on `NULL` examples may confuse the span generation model.

### Token-level loss

T5 uses teacher forcing with cross-entropy over target tokens.

The labels use:

```python
labels[labels == tokenizer.pad_token_id] = -100
```

This makes the loss ignore padding tokens.

Conceptually:

```python
L_gen = CrossEntropy(predicted_tokens, target_tokens)
```

The notebook stores a `LABEL_SMOOTHING` config value, but this baseline mainly relies on the standard `outputs.loss` returned by `T5ForConditionalGeneration`.

### Decoding

Beam search is used:

```python
num_beams = 4
```

The generated sequence is parsed with regex:

```python
[A](...)[O](...)[V](...)[SEP]
```

The parser also tries to salvage malformed VA values. For example, if only one numeric value is generated, it can duplicate it for both valence and arousal.

### Evaluation during training

The notebook computes:

- Triplet precision.
- Triplet recall.
- Triplet F1 based on exact aspect and opinion span match.
- VA RMSE only for correctly extracted triplets.
- VA variance to detect prediction collapse.

---

## Subtask 2 implementation 2: `subtask2-dimaste2.ipynb`

This version adds span matching, a separate VA regression head, fallback extraction, and VA diversity regularization.

### What it adds compared with implementation 1

| Area | Implementation 1 | Implementation 2 |
|---|---|---|
| Output format | Aspect + opinion + VA generated together | Aspect + opinion generated; VA predicted by separate head |
| Span validation | Regex only | Fuzzy span matching against original text |
| VA modeling | Generated as text | Separate neural VA head |
| Empty predictions | No fallback | Sentiment-word fallback |
| VA collapse prevention | VA variance monitoring | VA diversity loss |
| VA weighting | None | Progressive VA weight schedule |
| Beam search | 4 beams | 5 beams |

### SpanMatcher

The model may generate a span that is close but not identical to the original text. The `SpanMatcher` fixes this by checking:

1. Exact lowercase substring match.
2. Punctuation-cleaned match.
3. Single-word fuzzy match.
4. Multi-word sliding-window fuzzy match.

This reduces hallucinated spans and maps generated text back to real spans.

### New target format

The target no longer includes VA in the generated text. It uses:

```text
aspect | opinion [SEP] aspect | opinion
```

VA is learned separately through a neural regression head.

### VAHead

The model adds a separate head on top of T5 hidden states:

```text
T5 hidden states
    ↓
Mean pooling
    ├── Valence head → sigmoid → normalized valence
    └── Arousal head → sigmoid → normalized arousal
```

This is helpful because generating exact numeric VA strings is difficult for seq2seq models. Numeric regression is usually more stable.

### Combined loss

The implementation combines generation loss and VA regression loss:

```python
loss = (1 - va_weight) * gen_loss + va_weight * va_loss
loss += VA_DIVERSITY_WEIGHT * diversity_loss
```

Where:

```python
va_loss = mse_loss(predicted_v, target_v) + mse_loss(predicted_a, target_a)
```

The VA weight increases over training:

```text
Early epochs:  0.3  → focus on extraction
Middle epochs: 0.5  → balance extraction and VA
Late epochs:   0.7  → refine VA prediction
```

### VA diversity loss

The notebook uses:

```python
diversity_loss = -(var(predicted_valence) + var(predicted_arousal))
```

Adding this term discourages the model from predicting the same VA values for every sample.

### Sentiment fallback

If generation produces no valid triplets, the `SentimentDetector` searches for simple sentiment words such as:

```text
good, bad, great, excellent, poor, terrible, amazing, awful, slow, fast, clean, dirty
```

Then it creates a fallback triplet with heuristic VA values.

This is a recall-improving trick. It may introduce false positives, but it avoids completely empty outputs for obviously sentiment-bearing sentences.

---

## Subtask 2 implementation 3: `subtask2-diamaste3.ipynb`

This is the strongest Subtask 2 variant in the notebooks.

### What it adds compared with implementation 2

Implementation 3 keeps the main improved architecture from implementation 2 and strengthens evaluation and VA handling.

Important improvements:

- Uses the official-style continuous F1 formula.
- Keeps the separate VA regression head.
- Keeps fuzzy span matching.
- Keeps fallback extraction for empty predictions.
- Tracks generation loss, VA loss, diversity loss, precision, recall, F1, VA RMSE, and VA variance.
- Produces the best Subtask 2 F1 among the included variants.

### Why this version performs better

The main improvement is the separation of concerns:

```text
T5 generation: aspect/opinion extraction
VA head: continuous valence/arousal regression
Postprocessing: span correction + fallback + clipping
```

This is better than forcing T5 to generate exact floating-point numbers inside a text sequence.

---

## Subtask 2 evaluation results

The evaluation notebook computes official-style continuous scores.

| Prediction set | Laptop cPrecision | Laptop cRecall | Laptop cF1 | Restaurant cPrecision | Restaurant cRecall | Restaurant cF1 |
|---|---:|---:|---:|---:|---:|---:|
| Task 2 set 1 | 0.5596 | 0.4000 | 0.4665 | 0.5911 | 0.3673 | 0.4531 |
| Task 2 set 2 | 0.5942 | 0.3799 | 0.4635 | 0.6623 | 0.4894 | 0.5629 |
| Task 2 set 3 | **0.6064** | **0.3953** | **0.4786** | **0.6662** | **0.4969** | **0.5692** |

The third version gives the best cF1 for both domains.

---

# Subtask 3: DimASQP

## Goal

Subtask 3 extracts quadruplets:

```text
Aspect, Category, Opinion, VA
```

Input:

```text
The fans are loud.
```

Output:

```json
{
  "Aspect": "fans",
  "Category": "FANS_COOLING#OPERATION_PERFORMANCE",
  "Opinion": "loud",
  "VA": "3.00#7.33"
}
```

This is the hardest subtask because the model must perform:

1. Aspect extraction.
2. Opinion extraction.
3. Category classification.
4. VA regression/generation.

---

## Subtask 3 implementation 1: `subtask3-dimasqp1.ipynb`

This is the baseline T5 sequence-to-sequence implementation.

### Main idea

Input prompt:

```text
extract quadruples: <review text>
```

Target sequence:

```text
[A]aspect[C]category[O]opinion[V]valence#arousal[SEP]...
```

Example:

```text
[A]fans[C]FANS_COOLING#OPERATION_PERFORMANCE[O]loud[V]3.00#7.33
```

### Preprocessing

The notebook converts each quadruple into a sequence. Empty quadruple lists are represented as:

```text
none
```

Padding tokens in the target are ignored using `-100`.

### Category validation

The baseline has a simple `validate_category` function. If the generated category is invalid, it tries:

1. Exact match.
2. Fuzzy substring match.
3. Keyword fallback.

Example keyword fallback:

```text
FOOD      → FOOD#QUALITY
DRINK     → DRINKS#QUALITY
SERVICE   → SERVICE#GENERAL
AMBIENCE  → AMBIENCE#GENERAL
```

### Loss

The model uses the standard T5 generation loss:

```python
loss = outputs.loss
```

This is token-level cross-entropy over the generated target sequence.

### Limitation

Generating the whole quadruple as text is difficult, especially category strings and numeric VA values. In the evaluation notebook, the first prediction set has strong restaurant results but fails on laptop because the generated laptop categories do not align well with the gold category labels.

---

## Subtask 3 implementation 2: `subtask3-disasqp2.ipynb`

This version adds major postprocessing and training improvements.

### What it adds compared with implementation 1

| Area | Implementation 1 | Implementation 2 |
|---|---|---|
| Category set | Static / simple validation | Extracts all valid categories from training data |
| Span validation | None | Validates aspect and opinion against original text |
| NULL handling | Minimal | Filters NULL aspect/opinion examples |
| Loss | T5 cross-entropy | Cross-entropy with label smoothing |
| Category correction | Simple keyword fallback | Domain-aware category validation |
| Evaluation | Loss only | Continuous F1, cPrecision, cRecall |
| Collapse check | None | Token diversity penalty |

### Category extraction from training data

Instead of relying only on a hardcoded category list, the notebook extracts all categories from the training files:

```python
VALID_CATEGORIES, category_freq = extract_all_categories(train_files)
MOST_FREQUENT_CATEGORY = VALID_CATEGORIES[0]
```

This makes validation domain-aware and avoids impossible category labels.

### NULL filtering

The dataset removes training quadruples where:

```text
Aspect == "NULL"
Opinion == "NULL"
```

This aligns training with the test distribution, where the target outputs use explicit aspect and opinion spans.

### Span validation

Generated aspect and opinion spans are checked against the original text.

The logic accepts:

- Exact matches.
- Multi-word partial matches.
- Single-word partial matches.

Invalid generated spans are skipped.

This reduces hallucinations.

### Enhanced category validation

The improved validator checks:

1. Exact match.
2. Case-insensitive match.
3. Substring match.
4. Main category match before `#`.
5. Domain keyword mapping.

Examples:

```text
BATTERY   → BATTERY#OPERATION_PERFORMANCE
SCREEN    → DISPLAY#QUALITY
KEYBOARD  → KEYBOARD#QUALITY
SERVICE   → SERVICE#GENERAL
FOOD      → FOOD#QUALITY
```

### Label smoothing

Instead of relying only on `outputs.loss`, this notebook computes cross-entropy directly:

```python
loss_fct = nn.CrossEntropyLoss(
    label_smoothing=0.1,
    ignore_index=-100
)
loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
```

Label smoothing helps prevent overconfident token predictions, which is useful for generation tasks with long structured outputs.

### Diversity penalty

The notebook checks whether generated token predictions are too repetitive:

```python
unique_ratio = len(torch.unique(pred_ids)) / pred_ids.numel()
```

If the unique ratio is below `0.3`, it adds a penalty:

```python
diversity_loss = VA_DIVERSITY_WEIGHT * (0.3 - unique_ratio)
```

This is a simple anti-collapse trick.

### Continuous F1

The notebook implements continuous F1, where a structural match gets partial credit depending on VA distance.

For a matching structural tuple:

```python
va_error = sqrt((pred_v - gold_v)^2 + (pred_a - gold_a)^2)
score = 1 - va_error / threshold
```

In the later implementation this is corrected to use the official maximum distance.

---

## Subtask 3 implementation 3: `subtask3-disasqp3.ipynb`

This is the strongest Subtask 3 variant.

### What it adds compared with implementation 2

| Area | Implementation 2 | Implementation 3 |
|---|---|---|
| Span matching | Basic validation | More precise fuzzy span recovery |
| Opinion validation | Minimal | Opinion sentiment validation |
| Category consistency | Category validation only | Category-aspect consistency check |
| cF1 formula | Threshold-based VA penalty | Official `D_max = sqrt(128)` normalization |
| Postprocessing | Strong | Stronger and more conservative |

### SpanMatcher

The improved `SpanMatcher` attempts to recover the exact original span:

1. Exact lowercase substring match.
2. Punctuation-cleaned match.
3. Single-word fuzzy matching.
4. Multi-word sliding-window fuzzy matching.

This means a generated span such as:

```text
battery-life
```

can be mapped back to:

```text
battery life
```

if the similarity is high enough.

### OpinionValidator

The opinion validator checks whether the extracted opinion is plausible.

It uses sentiment words and simple rules:

- Opinion must not be empty.
- Opinion should not be suspiciously long.
- Opinion should contain or resemble sentiment-bearing text.

This helps avoid outputs where the model accidentally extracts a neutral noun phrase as the opinion.

### CategoryAspectChecker

This component checks whether the aspect and category are consistent.

Example:

```text
Aspect: "battery life"
Category: "BATTERY#OPERATION_PERFORMANCE"
```

is consistent.

But:

```text
Aspect: "battery life"
Category: "FOOD#QUALITY"
```

is inconsistent.

The checker uses category-specific keyword lists for restaurant and laptop categories.

### Official continuous F1 correction

The final implementation uses:

```python
D_max = sqrt(128)
```

This is the maximum possible Euclidean distance in the VA plane because both valence and arousal range from 1 to 9:

```text
sqrt((9 - 1)^2 + (9 - 1)^2) = sqrt(64 + 64) = sqrt(128)
```

The VA contribution is:

```python
score = 1 - va_error / sqrt(128)
```

This is better than using an arbitrary threshold because it matches the full possible VA range.

---

## Subtask 3 evaluation results

| Prediction set | Laptop cPrecision | Laptop cRecall | Laptop cF1 | Restaurant cPrecision | Restaurant cRecall | Restaurant cF1 |
|---|---:|---:|---:|---:|---:|---:|
| Task 3 set 1 | 0.0000 | 0.0000 | 0.0000 | 0.4787 | 0.4791 | 0.4789 |
| Task 3 set 2 | 0.2844 | 0.2458 | 0.2637 | 0.5620 | 0.5010 | 0.5297 |
| Task 3 set 3 | **0.2961** | **0.2559** | **0.2746** | **0.5653** | **0.5082** | **0.5353** |

The third version improves both domains. The largest improvement is in the laptop domain, where better category validation and span correction prevent the complete failure seen in the baseline prediction set.

---

# Evaluation logic

The notebook `evaluate_subtasks.ipynb` contains evaluation functions for all three subtasks.

---

## Task 1 metric: normalized VA RMSE + Pearson correlation

For Task 1, predictions are matched by `ID` and `Aspect`.

The evaluator extracts:

```text
gold_v, gold_a, pred_v, pred_a
```

It computes Pearson correlation separately:

```python
PCC_V = pearsonr(pred_v, gold_v)
PCC_A = pearsonr(pred_a, gold_a)
```

It also computes normalized VA RMSE:

```python
mse = sum((gold - pred)^2 over V and A) / number_of_aspects
rmse_va = sqrt(mse) / sqrt(128)
```

The division by `sqrt(128)` normalizes the score by the maximum possible VA distance.

---

## Task 2 and Task 3 metric: continuous F1

For Task 2, a structural match requires:

```text
Aspect + Opinion
```

For Task 3, a structural match requires:

```text
Aspect + Opinion + Category
```

If the structure matches, the VA score is not simply correct/incorrect. It receives partial credit based on distance:

```python
va_euclid = sqrt((pred_v - gold_v)^2 + (pred_a - gold_a)^2)
D_max = sqrt(128)
cTP = max(0, 1 - va_euclid / D_max)
```

Then:

```python
cPrecision = cTP_total / (TP + FP)
cRecall    = cTP_total / (TP + FN)
cF1        = 2 * cPrecision * cRecall / (cPrecision + cRecall)
```

This metric rewards exact span/category extraction and also rewards closer VA predictions.

---

# Important implementation tricks

## 1. Flattening aspect-level data

For Subtask 1, one review may have multiple aspects. The code converts:

```json
{
  "Text": "...",
  "Aspect_VA": [
    {"Aspect": "screen", "VA": "..."},
    {"Aspect": "keyboard", "VA": "..."}
  ]
}
```

into two training samples. This is necessary because the model predicts one VA pair per aspect.

---

## 2. Converting quadruplets to simpler structures

The training files are shared across tasks and often contain `Quadruplet`. For Subtask 1 and 2, the code extracts only the fields needed.

For Subtask 1:

```python
Quadruplet → Aspect_VA
```

For Subtask 2:

```python
Quadruplet → Triplet
```

This avoids maintaining separate training data for each task.

---

## 3. Filtering NULL spans

Several notebooks remove examples where aspect or opinion is `NULL`.

This is useful because the test data contains explicit spans, and the extraction models are expected to return text spans. Training on many `NULL` targets can make the model produce invalid or empty outputs.

---

## 4. Aspect marking

Aspect marking is used in improved Subtask 1 notebooks:

```text
[ASPECT] battery life [/ASPECT]
```

This is especially useful when a sentence has multiple possible sentiment targets.

Example:

```text
The screen is beautiful but the keyboard is terrible.
```

Without aspect marking, the model may average the sentiment of both aspects. With aspect marking, it can focus on the requested aspect.

---

## 5. Span matching instead of trusting generation

For Subtask 2 and 3, generated text may not exactly match the original review. Fuzzy span matching maps predictions back to real text spans.

This improves evaluation because official matching is span-sensitive.

---

## 6. Category validation

Subtask 3 category strings are long and easy to generate incorrectly.

The improved notebooks validate categories using:

- Training category inventory.
- Exact and case-insensitive matching.
- Substring matching.
- Main category before `#`.
- Keyword fallback.
- Category-aspect consistency.

This is one of the most important improvements for the laptop domain.

---

## 7. VA as regression instead of text generation

Generating numbers such as `7.75#7.38` is difficult for T5.

The stronger Subtask 2 implementations therefore use:

```text
T5 for span extraction
Separate neural head for VA regression
```

This is usually more stable than forcing the language model to generate exact continuous values.

---

## 8. VA diversity regularization

Regression models sometimes collapse to predicting average values like:

```text
6.50#6.50
```

The notebooks monitor and penalize low diversity using:

```python
var(predicted_valence) + var(predicted_arousal)
```

This encourages a wider and more realistic range of VA scores.

---

## 9. Progressive VA weighting

In extraction + regression models, early training should prioritize extraction. Later training can focus more on VA refinement.

The notebooks use:

```text
Early:  VA weight = 0.3
Middle: VA weight = 0.5
Late:   VA weight = 0.7
```

This prevents the model from failing extraction while trying to learn numeric regression too early.

---

## 10. Domain-specific ensembling

Subtask 1 implementation 4 trains separate models for restaurant and laptop data, then averages several seeds.

This can reduce variance but does not always improve final test score. In the included evaluation, the DeBERTa single-model approach performed better than the BERT ensemble.

---

# Output formats

## Subtask 1 output

```json
{
  "ID": "lap26_aspect_va_test_1",
  "Aspect_VA": [
    {
      "Aspect": "Dell",
      "VA": "7.57#7.63"
    }
  ]
}
```

## Subtask 2 output

```json
{
  "ID": "lap26_aste_test_2",
  "Triplet": [
    {
      "Aspect": "fans",
      "Opinion": "loud",
      "VA": "3.00#7.33"
    }
  ]
}
```

## Subtask 3 output

```json
{
  "ID": "lap26_asqp_test_2",
  "Quadruplet": [
    {
      "Aspect": "fans",
      "Category": "FANS_COOLING#OPERATION_PERFORMANCE",
      "Opinion": "loud",
      "VA": "3.00#7.33"
    }
  ]
}
```

---

# How to run

## 1. Prepare data

Place the train/dev/test files in the expected notebook folders or change the notebook config paths:

```python
TRAIN_ZIP = "/path/to/subtask_data"
DATA_DIR = "./subtask_data"
OUTPUT_DIR = "./subtask_outputs"
CHECKPOINT_DIR = "./subtask_checkpoints"
```

## 2. Run notebooks in order

Suggested order:

```text
1. Subtask 1 notebooks
2. Subtask 2 notebooks
3. Subtask 3 notebooks
4. evaluate_subtasks.ipynb
```

## 3. Evaluate predictions

The evaluation notebook expects prediction files under task-specific folders such as:

```text
1/laptop_test_predictions.jsonl
2/restaurant_test_predictions3.jsonl
3/laptop_test_predictions3.jsonl
```

Update the file paths in `evaluate_subtasks.ipynb` if your output folder names are different.

---

# Recommended final models

Based on the included evaluation notebook:

| Task | Recommended notebook | Reason |
|---|---|---|
| Subtask 1 | `subtask1-dimasr3.ipynb` | Best normalized VA RMSE, especially on restaurant |
| Subtask 2 | `subtask2-diamaste3.ipynb` | Best cF1 for both laptop and restaurant |
| Subtask 3 | `subtask3-disasqp3.ipynb` | Best cF1 for both laptop and restaurant |

---

# Known cleanup opportunities

The notebooks are experimental and contain some minor notebook-style issues that should be cleaned before turning this into a production package:

1. Some cells contain duplicated lines such as repeated `return_tensors='pt'` or repeated assignments.
2. Some output file names use inconsistent numbering.
3. The Kaggle input paths should be replaced by a config file or command-line arguments.
4. Shared functions such as `parse_va`, `format_va`, `load_jsonl`, and evaluation code should be moved into reusable Python modules.
5. Subtask 2 and 3 postprocessing should be centralized because both use span matching, VA formatting, and validation logic.
6. The generated prediction folders should use clear names such as `baseline`, `improved`, and `final` instead of only `predictions1`, `predictions2`, etc.

A cleaner future version could use:

```text
src/
├── data.py
├── metrics.py
├── models/
│   ├── dimasr.py
│   ├── dimaste.py
│   └── dimasqp.py
├── preprocessing.py
├── postprocessing.py
└── train.py
```

---

# Key takeaways

1. **Subtask 1 works best as regression.** The strongest version uses DeBERTa with separate heads for valence and arousal.
2. **Subtask 2 benefits from separating extraction and VA prediction.** T5 is useful for aspect/opinion generation, while a regression head is more stable for VA.
3. **Subtask 3 needs strong postprocessing.** Category validation and span correction are essential, especially for laptop categories.
4. **VA normalization and clipping are critical.** Invalid values outside `[1, 9]` damage the continuous scoring.
5. **Diversity checks matter.** Many models tend to collapse toward average VA values, so variance monitoring and diversity losses are useful.
6. **The best implementations are not only better models; they are better pipelines.** Preprocessing, validation, decoding, and metric-aware training are as important as the encoder architecture.
