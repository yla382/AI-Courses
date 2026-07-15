# Week 5 Study Guide — Classical ML Deep Dive

Work through each section, then immediately do that section in `tutorial.ipynb` before moving on.

---

## What Are Classical ML Algorithms?

Last week you built the learning engine — gradient descent. This week you use it. Classical ML algorithms are the workhorses of real-world ML: fast to train, easy to interpret, and often hard to beat on structured/tabular data even with neural networks.

Each algorithm has a different answer to the question: **"what does it mean for two data points to be similar?"**

---

## §1 — k-Nearest Neighbors (kNN)

### The idea

kNN makes no assumptions about the data. To classify a new point, it simply looks at the `k` closest training points and takes a majority vote.

```
New patient data → find 5 most similar patients in training set
                → 4 had cancer, 1 didn't
                → predict: cancer
```

### Distance matters

"Closest" is measured by Euclidean distance:

```
distance = sqrt( (x₁-x₂)² + (y₁-y₂)² + ... )
```

This is why StandardScaler matters for kNN — if age is 0–80 and salary is 0–100,000, salary dominates the distance calculation completely.

### Choosing k

- **k too small (k=1):** overfits — noisy, sensitive to outliers
- **k too large (k=100):** underfits — ignores local structure
- **Rule of thumb:** start with `k = sqrt(n)` where n is training size, tune from there

### Pros and cons
```
✅ Simple, no training required (lazy learner)
✅ Naturally handles multi-class problems
❌ Slow at prediction time — must compute distance to every training point
❌ Breaks down in high dimensions (curse of dimensionality)
❌ Sensitive to irrelevant features and scale
```

---

## §2 — Logistic Regression (deeper dive)

You've used it already — now understand what it's actually doing.

### The sigmoid function

Linear regression outputs any number. For classification you need a probability (0 to 1). The sigmoid squashes any number into that range:

```
σ(z) = 1 / (1 + e^(-z))

z = -∞  →  σ = 0
z =  0  →  σ = 0.5
z = +∞  →  σ = 1
```

### The full forward pass

Here is the complete pipeline, step by step:

```
Input: x1, x2, ..., xn  (your features)
         |
         v
z = w1*x1 + w2*x2 + ... + b     (linear combination, same as linear regression)
         |
         v
p = 1 / (1 + e^(-z))            (sigmoid squashes z into a probability 0-1)
         |
         v
predict 1 if p >= 0.5, else 0
```

Nothing new in the math — the only addition is the sigmoid step before predicting.

### Why log loss instead of MSE?

In week 4 you minimized MSE: `(y_pred - y_actual)^2`.

For classification, `y_pred` is a probability (0 to 1) and `y_actual` is 0 or 1. If you plug probabilities into MSE, the loss surface has many local minima — gradient descent gets stuck.

Log loss (also called cross-entropy) fixes this. The formula:

```
Log Loss = -1/n * sum( y * log(p) + (1-y) * log(1-p) )
```

What each part does:
- When `y = 1` (true label is positive): only `y * log(p)` matters. If p is close to 1 (confident and correct), log(p) is close to 0, small loss. If p is close to 0 (confident but WRONG), log(p) is a very large negative number, huge loss.
- When `y = 0` (true label is negative): only `(1-y) * log(1-p)` matters. Same idea flipped.

Plain English: **log loss punishes confident wrong predictions very heavily.** The log function grows fast near 0 — being 99% sure of the wrong answer is penalized far more than being 60% sure of the wrong answer.

Log loss gives a smooth convex surface with one global minimum — gradient descent converges reliably.

### Training loop (same as week 4)

The training process is identical to what you did in week 4, just with different loss:

```
For each epoch:
    1. z = X*w + b                   (forward pass: linear combination)
    2. p = sigmoid(z)                (forward pass: get probabilities)
    3. compute Log Loss from p and y (measure how wrong we are)
    4. compute gradients dL/dw, dL/db
    5. w = w - lr * dL/dw            (step downhill)
    6. b = b - lr * dL/db
```

sklearn's `LogisticRegression.fit()` runs all 6 steps for you — same as `LinearRegression.fit()` ran gradient descent internally in week 4.

### Decision boundary

Logistic regression draws a straight line (or hyperplane in higher dimensions) between classes. It can only separate linearly separable data — if the boundary is curved, it will struggle.

### Pros and cons
```
✅ Fast, interpretable — weights tell you feature importance
✅ Outputs calibrated probabilities
✅ Works well when classes are linearly separable
❌ Fails on non-linear boundaries without feature engineering
```

---

## §3 — Support Vector Machines (SVM)

### The idea

SVM also draws a line between classes — but it specifically finds the line that **maximizes the margin** between the two classes.

```
Class A: ●  ●  ●
                    |  ← maximum margin boundary
Class B:            |        ○  ○  ○
```

The points closest to the boundary are called **support vectors** — they're the only points that matter for defining the boundary.

### Kernel trick

What if the data isn't linearly separable? SVM can map data into higher dimensions where it becomes separable, using a **kernel function** — without actually computing the high-dimensional coordinates.

Common kernels:
- `linear` — straight line boundary
- `rbf` (radial basis function) — circular/blobby boundary, most common
- `poly` — polynomial curve boundary

```
linear kernel:          rbf kernel:             poly kernel:
                                                  
  A  A  |  B  B          A  A  A                   B  B
  A  A  |  B  B          A  +---+ B              B  \   B
  A  A  |  B  B          A  | B | B             B    \   B
                          A  +---+ B           A  A   \  B
  straight line           A  A  A             A  A     \ B
                                                         \
                          B clusters inside    curved polynomial
                          a circular boundary  boundary
```

rbf is the go-to default — it handles any blob-shaped cluster. poly is rarely used unless you know the boundary has a specific polynomial shape.

### The C parameter

C controls the tradeoff between a clean margin and misclassifying training points:

```
Low C (e.g. 0.1):              High C (e.g. 100):

  A  A  |    B  B              A  A  A|B  B
  A     |  B    B              A  A   |   B
  A  A  |    B                 A    A |B  B
         ^                            ^
  wide margin, allows          narrow margin, tries to
  some misclassification       classify everything correctly
  (more generalization)        (risk of overfitting)
```

- **Low C** — prioritizes a wide margin, accepts some training errors. Better generalization.
- **High C** — prioritizes classifying all training points correctly. Can overfit.

Start with `C=1` (default) and tune from there with cross-validation.

### Pros and cons
```
✅ Very effective in high-dimensional spaces
✅ Robust — only support vectors matter, outliers have less effect
✅ Works well when classes are clearly separated
❌ Slow on large datasets
❌ Hard to interpret
❌ Needs careful tuning of kernel and C parameter
```

---

## §4 — Naive Bayes

### The idea

Uses Bayes theorem (week 3) to classify. For each class, it asks: "what's the probability this data point belongs here?"

```
P(class | features) ∝ P(features | class) × P(class)
```

**Naive** means it assumes all features are independent of each other — which is almost never true in reality, but it works surprisingly well anyway.

### Why it works despite the naive assumption

Even if features aren't truly independent, the relative probabilities still tend to point to the right class. It's wrong in an absolute sense but right in a comparative sense.

### Pros and cons
```
✅ Extremely fast to train and predict
✅ Works well with small datasets
✅ Great for text classification (spam detection, sentiment)
❌ The independence assumption is almost always wrong
❌ Probability estimates are unreliable (bad calibration)
```

---

## §5 — Comparing Algorithms

No single algorithm wins on all datasets. The right choice depends on:

| Factor | Best choice |
|--------|-------------|
| Small dataset | kNN, Naive Bayes |
| Large dataset | Logistic Regression, SVM with linear kernel |
| Non-linear boundary | SVM with rbf kernel |
| Need interpretability | Logistic Regression |
| Text data | Naive Bayes |
| Speed matters | Logistic Regression, Naive Bayes |

**In practice:** try several, compare with cross-validation F1, pick the best one for your dataset.

---

## §6 — PCA (Principal Component Analysis)

### The problem

The breast cancer dataset has 30 features. You can't visualize 30 dimensions. More features also means slower training and risk of overfitting.

PCA reduces dimensions while preserving as much information (variance) as possible.

### The idea

Find the directions in the data where variance is highest — those are the **principal components**. Project the data onto the top k of those directions.

```
30 features → PCA → 2 components
```

Each component is a linear combination of the original features. The first component captures the most variance, the second captures the second most, and so on.

### Explained variance

Each component explains some percentage of the total variance:

```
Component 1: 44% of variance
Component 2: 19% of variance
──────────────────────────────
Together:    63% of variance
```

You choose how many components to keep based on how much variance you want to preserve (typically 95%).

### When to use PCA
- Visualizing high-dimensional data (reduce to 2D or 3D)
- Speeding up training when you have many features
- Removing noise (low-variance components often capture noise)

### Pros and cons
```
✅ Reduces training time and memory
✅ Can improve model performance by removing noisy features
✅ Essential for visualization
❌ Components are hard to interpret (they're combinations of original features)
❌ Can discard information that matters for the task
```

---

## Checklist

- [ ] Can explain how kNN makes predictions and why scaling matters
- [ ] Know what the sigmoid function does and why logistic regression uses it
- [ ] Can explain what "maximum margin" means in SVM
- [ ] Know what "naive" means in Naive Bayes and why it still works
- [ ] Can compare algorithms and pick one for a given scenario
- [ ] Understand what PCA does and what "explained variance" means
