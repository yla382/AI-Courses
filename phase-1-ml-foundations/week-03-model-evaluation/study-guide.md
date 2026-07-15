# Study Guide — Week 3: Model Evaluation + Stats II

---

## Knowledge Gaps from Week 2 — Covered Here First

Before the new material, four concepts from week 2 that needed more depth.

---

### Gap 1 — StandardScaler: What It Actually Does

In week 1 you used `StandardScaler` without knowing what was happening inside. Here it is.

**The problem it solves:**

Imagine your dataset has two features:
- Age: values between 20–80
- Salary: values between 30,000–200,000

Many ML algorithms compute distances or dot products between features. If salary is 1000× larger than age numerically, the model will treat salary as 1000× more important — even if it isn't. That's a bug, not a feature.

**What StandardScaler does:**

It transforms each feature so that:
- Mean = 0
- Standard deviation = 1

Formula for each value:
```
scaled = (value - mean) / std
```

Example — scaling both age and salary:
```
ages   = [22,    25,    28,    35,    40   ]   mean=30,    std=7
salary = [45000, 72000, 88000, 110000, 51000]  mean=73200, std=25800

scaled ages   = [-1.14, -0.71, -0.29,  0.71,  1.43]
scaled salary = [-1.09, -0.05,  0.57,  1.42, -0.86]
```

Now both features live in roughly the same -2 to +2 range. A value of +1.0 means "1 standard deviation above average" for both age and salary — regardless of their original units. The model can compare them fairly without salary dominating just because its numbers are 2000× larger.

**Three methods you need to know:**

Think of the scaler as a calculator that needs two steps: first measure, then apply.

- **`fit(data)`** — goes through the data and computes the mean and std for each feature. Stores those numbers internally. Does NOT change the data at all — it just remembers the numbers.
- **`transform(data)`** — takes the mean and std that were stored by `fit()`, and applies the formula `(value - mean) / std` to every value. This is what actually changes the numbers.
- **`fit_transform(data)`** — runs `fit()` then `transform()` on the same data in one call. Shorthand for doing both steps at once.

```python
# These two blocks do exactly the same thing:

# Option A — two steps
scaler.fit(X_train)                      # step 1: compute mean/std from X_train, store them
X_train_scaled = scaler.transform(X_train)  # step 2: apply (value - mean) / std to X_train

# Option B — one step shorthand
X_train_scaled = scaler.fit_transform(X_train)  # compute mean/std AND apply in one call
```

**Critical rule — why you fit on train only, then transform test separately:**

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # computes mean/std from training data, then scales it
X_test_scaled  = scaler.transform(X_test)        # uses the SAME mean/std from training, scales test data
```

Why not just call `fit_transform` on the test data too? Because `fit_transform` would compute a NEW mean and std from the test data — and that would be wrong. The scaler must use the same mean/std it learned from training, applied consistently to all future data. If you refit on test data, you're treating each dataset as if it exists in isolation, which breaks the consistency between how training and test data are scaled.

Never fit on test data. If you do, you're using future information to scale your training data — that's data leakage.

**Why this actually matters — two concrete examples:**

*kNN (k-Nearest Neighbors)* classifies a new point by finding the k closest training points using distance:
```
distance = sqrt((age_diff)² + (salary_diff)²)
```

Without scaling:
```
Person A: age=25, salary=50000
Person B: age=26, salary=51000  ← 1 year older, $1000 more
Person C: age=35, salary=50100  ← 10 years older, $100 more

Distance A→B = sqrt(1²    + 1000²) = 1000.0
Distance A→C = sqrt(10²   + 100²)  = 100.5
```

The model thinks C is 10× closer to A than B is — purely because the salary difference is numerically smaller, not because age actually matters less. Age contributes almost nothing to the distance. The model ends up classifying almost entirely by salary, making category boundaries that look random with respect to age. After scaling, both features contribute equally.

*Logistic Regression / Neural Networks* learn weights via gradient descent. Features with large values produce large gradients, causing the optimizer to update those weights much more aggressively than weights for small-scale features. This causes slow, unstable training or convergence to a poor solution.

**Which models need it:**
- Logistic Regression ✅
- SVM ✅
- kNN ✅
- Neural Networks ✅
- Decision Trees / Random Forests ❌ — tree models split on thresholds, scale doesn't matter

---

### Gap 2 — Correlation Interpretation: What's Weak vs Strong

The correlation coefficient ranges from -1 to +1. Here's how to interpret the magnitude:

| Absolute value of r | Interpretation |
|---------------------|----------------|
| 0.00 – 0.09 | Negligible — essentially no linear relationship |
| 0.10 – 0.29 | Weak — slight trend, not reliable on its own |
| 0.30 – 0.49 | Moderate — meaningful signal, worth investigating |
| 0.50 – 0.69 | Strong — clear relationship |
| 0.70 – 0.89 | Very strong — features move closely together |
| 0.90 – 1.00 | Near perfect — almost certainly redundant features |

**Examples from Titanic (week 2):**
- `pclass` vs `survived`: -0.34 → moderate negative. Class matters but isn't everything.
- `pclass` vs `fare`: -0.55 → strong negative. Higher class number = much lower fare.
- `age` vs `survived`: -0.07 → negligible. Age barely predicts survival on its own.
- `sibsp` vs `parch`: +0.41 → moderate positive. People with siblings tend to also travel with parents.

**Important caveats:**
1. These thresholds are guidelines, not rules. Domain context matters.
2. Correlation only measures *linear* relationships. A feature can be powerfully predictive in a nonlinear way while showing near-zero correlation.
3. In ML, even a "weak" correlation of 0.15 can be useful — especially combined with other features.

---

### Gap 3 — Kurtosis: What Values Mean

You saw `df['fare'].kurtosis()` in week 2. Here's what to do with the number.

Kurtosis measures how heavy the **tails** of a distribution are compared to a normal distribution.

```python
series.kurtosis()  # pandas uses "excess kurtosis" — normal distribution = 0
```

| Value | Meaning |
|-------|---------|
| ~0 | Normal-like tails (mesokurtic) |
| > 0 (positive) | Heavier tails than normal — more extreme outliers (leptokurtic) |
| < 0 (negative) | Lighter tails than normal — fewer extremes, flatter shape (platykurtic) |

**Rule of thumb:**
- `|kurtosis| < 1` → tails are roughly normal, not a concern
- `kurtosis > 3` → heavy tails, expect lots of outliers — consider robust scaling or log transform
- `kurtosis > 10` → extreme outliers present, definitely investigate

**Titanic fare kurtosis** was around 33 — extremely heavy tails. This confirmed what we saw in the histogram: a handful of passengers paid $500+ while most paid under $30. That's why the log transform helped so much.

**In practice:** kurtosis is a secondary check. Always look at the histogram first — the shape will tell you more than the number alone.

---

### Gap 4 — Log Transform: Why and When

**The problem:** many real-world features are right-skewed — a few very large values with most data clustered at the low end. Income, fare prices, website traffic, population counts all look like this.

```
Original fare distribution:
████ (most people paid $7–30)
█
█
█                        · · ·  (a few paid $200–500)
0    50   100   150   200   250 ...
```

Skewed features hurt linear models because they assume the relationship between feature and target is roughly linear across the whole range. A $1 increase from $7 to $8 is treated the same as $7 to $8 increase from $493 to $500 — but in reality the first is a 14% increase and the second is 1.4%.

**The fix — log transform:**

```python
import numpy as np

# np.log1p = log(1 + x)
# The +1 handles zeros — log(0) is undefined, log(1+0) = 0
fare_log = np.log1p(df['fare'])
```

What log does: compresses large values, spreads out small values.
```
log(1)   = 0
log(10)  = 2.3
log(100) = 4.6    # 100x increase → only 2x increase in log space
log(500) = 6.2
```

**When to use it:**
- Skewness > 1 → consider it
- Skewness > 2 → strongly recommended for linear models
- Tree models (Random Forest, XGBoost) → usually not needed

**When NOT to use it:**
- Features with negative values (log of negative = undefined)
- Features that are already symmetric
- When you need to interpret the raw feature value (log transform makes interpretation harder)

**After transforming, remember:** your model learns on the log scale. When you feed new data in for prediction, you must log-transform it first too.

---

## §1 — Why Accuracy Alone Lies

Now the new material for week 3.

Accuracy is the most intuitive metric:
```
Accuracy = correct predictions / total predictions
```

But it can be completely misleading. Here's why.

**The imbalanced data problem:**

Imagine a model that predicts whether a patient has a rare disease. The disease affects 1% of the population. A model that predicts "no disease" for *every single patient* would have:

```
Accuracy = 990/1000 = 99%
```

99% accuracy. Sounds great. But the model is completely useless — it never detects the disease at all.

This is why you need more than accuracy. You need to know:
- Of the patients the model flagged as sick, how many were actually sick? (**Precision**)
- Of all the patients who were actually sick, how many did the model catch? (**Recall**)

These questions require understanding the **confusion matrix** first.

---

## §2 — Stats II: Probability Basics

### What is Probability?

Probability is a number between 0 and 1 that represents how likely something is.
- 0 = impossible
- 1 = certain
- 0.7 = 70% likely

When a classification model outputs probabilities (like `predict_proba()` in sklearn), it's saying: *"I'm 70% confident this is class 1."* You then apply a threshold (usually 0.5) to convert that to a hard prediction.

### Conditional Probability

P(A | B) = "probability of A, given that B is true"

Example:
- P(survived) = 0.38 — 38% of all passengers survived
- P(survived | female) = 0.74 — 74% of female passengers survived
- P(survived | male) = 0.19 — only 19% of male passengers survived

The condition (female/male) completely changes the probability. This is the foundation of how ML models work — they find features that change the probability of the outcome.

### Bayes Theorem (Intuition)

Bayes theorem answers this question: **given that something happened, how does that change what I believe?**

The formula:
```
P(A|B) = (P(B|A) × P(A)) / P(B)
```

Instead of memorizing the formula, let's build intuition through a concrete example with real numbers.

**Medical test — worked through step by step:**

Setup:
- Disease affects 1% of the population → 1 in 100 people have it
- Test is 95% accurate → if you have the disease, it correctly says positive 95% of the time. If you don't have it, it correctly says negative 95% of the time (meaning 5% false alarm rate).

Imagine testing **10,000 people**:

```
10,000 people total
├── 100 actually have the disease (1%)
│   ├──  95 test POSITIVE  ✅  (correctly caught)
│   └──   5 test NEGATIVE  ❌  (missed — false negative)
└── 9,900 don't have the disease (99%)
    ├── 495 test POSITIVE  ❌  (false alarm — false positive)
    └── 9,405 test NEGATIVE ✅  (correctly cleared)
```

Say you want to find the real probability you're sick given a positive test — not just "the test is 95% accurate" but the actual chance accounting for how rare the disease is. Bayes theorem gives us exactly that:

```
P(disease | positive test) = P(positive test | disease) × P(disease)
                             ─────────────────────────────────────────
                                      P(positive test)
```

Fill in each piece:
```
P(disease)               = 0.01   — 1% of population has the disease (base rate)
P(positive test | disease) = 0.95   — if you're sick, test catches it 95% of the time
P(positive test)         = ?      — overall chance of testing positive, from any reason
```

P(positive test) has two ways it can happen — you're either sick or healthy. To see where the multiplication comes from, start with 10,000 people:

```
Step 1: split by disease
  10,000 × 0.01 = 100 sick
  10,000 × 0.99 = 9,900 healthy

Step 2: of the 100 sick, how many test positive?
  100 × 0.95 = 95 people

Step 3: of the 9,900 healthy, how many test positive (false alarms)?
  9,900 × 0.05 = 495 people

Step 4: total who test positive
  95 + 495 = 590 people

Step 5: express as fraction of 10,000
  590 / 10,000 = 0.059
```

Now look at where 0.0095 came from:
```
95 / 10,000
= (10,000 × 0.01 × 0.95) / 10,000
= 0.01 × 0.95
= P(sick) × P(positive | sick)
```

The 10,000 cancels out. Multiplying probabilities is just what's left when you split a group twice and express it as a fraction of the total. So:

```
P(positive test) = P(positive | sick)    × P(sick)
                 + P(positive | healthy) × P(healthy)
                 = 0.95 × 0.01  +  0.05 × 0.99
                 = 0.0095       +  0.0495
                 = 0.059
```

Now plug everything in:
```
P(disease | positive test) = (0.95 × 0.01) / 0.059
                           = 0.0095 / 0.059
                           = 0.16 = 16%
```

This matches the counting approach: 95 / 590 = 16%. The formula and the counting method are two ways of computing the same thing.

**You test positive on a 95% accurate test — and you only have a 16% chance of actually being sick.**

Why? Because the disease is so rare (1%) that even a small false alarm rate (5%) generates far more false alarms than true positives. There are 99× more healthy people than sick people, so catching just 5% of them as false positives (495 people) completely swamps the 95 true positives.

This is the core insight Bayes gives you: **the base rate (how common the thing actually is) matters enormously.** A "95% accurate" test on a rare event will still produce mostly false positives in practice.

This is why:
- Fraud detection flags many valid transactions — fraud is ~0.1% of transactions, so false alarms dominate
- Medical screening always needs a follow-up confirmatory test
- Spam filters work better — spam is common enough that true positives outnumber false alarms

The key takeaway for ML: **a model's confidence score is not the same as the true probability of being correct** — especially when one class is much rarer than the other. This is exactly why accuracy fails on imbalanced problems, and why precision and recall tell a more honest story.

### Deep Dive Pointers
- [3Blue1Brown — Bayes theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) (15 min) — best visual explanation
- [StatQuest — Probability](https://www.youtube.com/watch?v=uzkc-qNVoOk)

---

## §3 — The Confusion Matrix

The confusion matrix is a 2×2 table that breaks down every prediction your model made.

```
                    Predicted: No    Predicted: Yes
Actual: No      |      TN          |      FP        |
Actual: Yes     |      FN          |      TP        |
```

- **TP (True Positive):** Model said YES, reality is YES. Correct.
- **TN (True Negative):** Model said NO, reality is NO. Correct.
- **FP (False Positive):** Model said YES, reality is NO. Wrong — "false alarm."
- **FN (False Negative):** Model said NO, reality is YES. Wrong — "missed it."

**Titanic example** — predicting survival:
```
                    Predicted: Died    Predicted: Survived
Actual: Died    |       TN=490      |       FP=59        |
Actual: Survived|       FN=91       |       TP=251       |
```

- 490 correctly predicted deaths ✅
- 251 correctly predicted survivals ✅
- 59 passengers predicted to survive who actually died ❌ (false alarm)
- 91 survivors the model missed ❌ (missed detections)

**Every metric you'll see in ML — precision, recall, F1, accuracy — is just arithmetic on these four numbers.**

---

## §4 — Precision, Recall, and F1

Quick reminder of what TP, FP, FN mean in a medical context:
- **TP (True Positive):** Patient is sick, model says sick. Correct catch.
- **FP (False Positive):** Patient is healthy, model says sick. False alarm.
- **FN (False Negative):** Patient is sick, model says healthy. Missed case.

### Precision

```
Precision = TP / (TP + FP)
```

**"Of everyone the model flagged as sick — how many were actually sick?"**

The denominator is everyone the model predicted positive: true positives + false positives. Precision asks: of those predictions, what fraction was correct?

High precision = when the model raises an alarm, it's almost always real. Few false alarms.
Low precision = the model flags too many healthy people as sick.

**Titanic:** Precision = 251 / (251 + 59) = 0.81 → when the model predicted survival, it was right 81% of the time.

### Recall (Sensitivity)

```
Recall = TP / (TP + FN)
```

**"Of all the patients who were actually sick — how many did the model catch?"**

The denominator is everyone who was actually positive: true positives + false negatives (the ones the model missed). Recall asks: of all the real cases, what fraction did we find?

High recall = model catches most real cases. Few slipping through undetected.
Low recall = model misses a lot of sick patients.

**Titanic:** Recall = 251 / (251 + 91) = 0.73 → the model caught 73% of actual survivors.

### Why they pull in opposite directions

Precision and recall measure two different failure modes:
- **Low precision** → too many false alarms (healthy people flagged as sick)
- **Low recall** → too many missed cases (sick people told they're fine)

You can always make recall 100% by predicting everyone is sick — you'd catch every real case, but precision would crater because you're also flagging all the healthy people. The model has to balance both.

### The Precision-Recall Tradeoff

You can almost always increase one by decreasing the other. The threshold (default 0.5) controls this balance.

- **Lower threshold** (e.g., 0.3): model predicts "yes" more often → catches more positives (↑ recall) but makes more false alarms (↓ precision)
- **Higher threshold** (e.g., 0.7): model predicts "yes" less often → fewer false alarms (↑ precision) but misses more positives (↓ recall)

**Which matters more depends on the problem:**

| Scenario | Optimize For | Why |
|----------|-------------|-----|
| Cancer screening | Recall | Missing a cancer case is catastrophic |
| Spam filter | Precision | Blocking legitimate email is very annoying |
| Fraud detection | Recall | Missing fraud is expensive |
| Recommendation system | Precision | Showing irrelevant recommendations is annoying |

### F1 Score

When you can't decide between precision and recall, F1 is the harmonic mean of both:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 is high only when **both** precision and recall are high. If either is low, F1 is low.

**Interpretation:**
- F1 > 0.9 → excellent
- F1 0.8–0.9 → good
- F1 0.7–0.8 → acceptable for many problems
- F1 < 0.7 → model needs work

Use F1 as your go-to metric for imbalanced classification problems.

### Deep Dive Pointers
- [StatQuest — Confusion Matrix](https://www.youtube.com/watch?v=Kdsp6soqA7o)
- [StatQuest — Sensitivity and Specificity](https://www.youtube.com/watch?v=vP06aMoz4v8)

---

## §5 — ROC Curve and AUC

### What the ROC Curve Shows

ROC stands for Receiver Operating Characteristic. It's a curve that shows the tradeoff between:
- **True Positive Rate (TPR)** = Recall = TP / (TP + FN) — on the Y axis
- **False Positive Rate (FPR)** = FP / (FP + TN) — on the X axis

At each possible threshold (0.0 to 1.0), you get a different (FPR, TPR) pair. The ROC curve plots all of these.

```
TPR
1.0 |          ___----
    |      ---/
    |    -/
    |  -/
    | /
    |/ (diagonal = random guessing)
0.0 +-------------- FPR
   0.0              1.0
```

A perfect model hugs the top-left corner (high TPR, low FPR). A random model falls on the diagonal.

### AUC — One Number to Rule Them All

AUC = Area Under the Curve. It summarizes the entire ROC curve as a single number.

| AUC | Interpretation |
|-----|----------------|
| 1.0 | Perfect model |
| 0.9–1.0 | Excellent |
| 0.8–0.9 | Good |
| 0.7–0.8 | Fair |
| 0.6–0.7 | Poor |
| 0.5 | No better than random guessing |
| < 0.5 | Worse than random (predictions are inverted) |

**Why use AUC instead of accuracy?**
- AUC is threshold-independent — it measures overall discriminative ability
- AUC is robust to class imbalance — accuracy isn't
- AUC = 0.85 means: given a random positive and a random negative example, the model ranks the positive higher 85% of the time

### Deep Dive Pointers
- [StatQuest — ROC and AUC](https://www.youtube.com/watch?v=4jRBRDbJemM) — the clearest explanation available

---

## §6 — Cross-Validation

### The Problem with a Single Train/Test Split

When you split data once (80/20), your evaluation depends heavily on *which* 20% ended up in the test set. You might get lucky (easy test cases) or unlucky (hard ones). The result has high variance.

### k-Fold Cross-Validation

Instead of one split, you make k splits:

```
Fold 1: [TEST] [train] [train] [train] [train]
Fold 2: [train] [TEST] [train] [train] [train]
Fold 3: [train] [train] [TEST] [train] [train]
Fold 4: [train] [train] [train] [TEST] [train]
Fold 5: [train] [train] [train] [train] [TEST]
```

Each fold uses a different 20% as test data. You train 5 models and get 5 scores. Then average them.

**Result:** a much more reliable estimate of true model performance, plus a standard deviation that tells you how consistent the model is.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f'F1 scores: {scores.round(3)}')
print(f'Mean F1: {scores.mean():.3f} ± {scores.std():.3f}')
```

### When to Use It

- **Always** when comparing models — don't pick the winner based on one split
- **Always** when tuning hyperparameters
- **Not needed** on huge datasets (millions of rows) — single split is reliable enough

### k=5 or k=10?

- k=5: faster, slightly more bias, standard choice for most problems
- k=10: slower, slightly less bias, common in research
- Start with k=5

### Deep Dive Pointers
- [StatQuest — Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw)

---

## Week 3 Checklist

- [ ] Understand why StandardScaler is needed and what it computes
- [ ] Can interpret a correlation value as weak/moderate/strong
- [ ] Know what kurtosis tells you and when to act on it
- [ ] Know when and why to apply a log transform
- [ ] Can explain why accuracy fails on imbalanced data
- [ ] Can read a confusion matrix and identify TP, TN, FP, FN
- [ ] Can compute precision, recall, and F1 from a confusion matrix by hand
- [ ] Know when to prioritize precision vs recall
- [ ] Can interpret an AUC score
- [ ] Understand what cross-validation does and why it's more reliable than a single split
