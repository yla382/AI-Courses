# Week 6 Study Guide — Ensemble Methods

Work through each section, then immediately do that section in `tutorial.ipynb` before moving on.

---

## Why Ensembles?

A single decision tree is easy to understand but prone to overfitting — it memorizes training data. The key insight behind ensemble methods:

**Many weak models combined beat one strong model.**

A "weak" model is just one that's slightly better than random guessing. If you combine hundreds of them in the right way, the errors cancel out and accuracy goes up.

---

## §1 — Decision Trees (Foundation)

Before ensembles, you need to understand what they're built from.

### The idea

A decision tree splits data by asking yes/no questions:

```
Is alcohol > 13.0?
    Yes -> Is color_intensity > 5.0?
               Yes -> class 0 (wine type A)
               No  -> class 1 (wine type B)
    No  -> class 2 (wine type C)
```

Each split tries to separate the classes as cleanly as possible.

### The problem: overfitting

A tree can keep splitting until every leaf has exactly one training sample — 100% training accuracy, terrible test accuracy. It memorized the data instead of learning patterns.

```
max_depth=2  ->  underfits (too simple)
max_depth=20 ->  overfits (memorizes training data)
```

`max_depth` controls how deep the tree can go.

### Pros and cons
```
+ Easy to visualize and interpret
+ No scaling needed
+ Handles mixed feature types
- Very prone to overfitting
- Unstable: small data changes -> very different tree
```

---

## §2 — Random Forest (Bagging)

### The idea

Train many decision trees, each on a random subset of the data and a random subset of features. To predict, every tree votes and you take the majority.

```
Tree 1 (trained on random 80% of data): predicts class A
Tree 2 (trained on different 80%):      predicts class A
Tree 3 (trained on different 80%):      predicts class B
...
Tree 100:                                predicts class A

Majority vote -> class A
```

This is called **bagging** (bootstrap aggregating).

### Why it works

Each tree makes different errors because it sees different data. When you average, the errors cancel out. The signal stays, the noise averages to zero.

Key requirement: the trees must be **uncorrelated** — if they all make the same mistakes, averaging doesn't help. Random feature selection at each split ensures this.

### Key parameters

```
n_estimators   — number of trees (more = better, diminishing returns after ~200)
max_depth      — max depth per tree (None = grow fully, which is fine for RF)
max_features   — features considered at each split ('sqrt' is default, good)
```

### Feature importance

Random Forest gives you `feature_importances_` — how much each feature contributed to reducing errors across all trees. This is one of the most useful outputs in practice.

### Pros and cons
```
+ Rarely overfits
+ Built-in feature importance
+ Robust to outliers and missing patterns
+ Little tuning needed
- Slower than a single tree
- Hard to interpret individual predictions
- Memory intensive with many trees
```

---

## §3 — Gradient Boosting

### The idea

Instead of training trees in parallel (bagging), train them **sequentially**. Each new tree tries to fix the mistakes of all previous trees.

```
Tree 1: makes predictions, has errors
Tree 2: trained on the errors of tree 1
Tree 3: trained on the errors of tree 1 + tree 2
...
Final prediction = sum of all trees
```

This is called **boosting**.

### Connection to gradient descent

Remember week 4 — gradient descent takes small steps downhill on the loss surface. Gradient boosting does the same thing, but each "step" is a new decision tree fitted to the residual errors. Trees are the steps, learning rate controls how big each step is.

```
prediction = tree1(x) * lr
           + tree2(x) * lr    (tree2 fitted to errors of tree1)
           + tree3(x) * lr    (tree3 fitted to errors of tree1+tree2)
           + ...
```

### Key parameters

```
n_estimators    — number of trees (boosting can overfit with too many)
learning_rate   — how much each tree contributes (lower = more trees needed)
max_depth       — depth of each tree (keep low: 3-6 for boosting)
subsample       — fraction of data used per tree (adds randomness, helps generalization)
```

Lower `learning_rate` + more `n_estimators` generally gives better results but takes longer.

### XGBoost

XGBoost is an optimized implementation of gradient boosting — faster, regularization built in, handles missing values. It wins most tabular data Kaggle competitions.

```python
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
```

### Pros and cons
```
+ Often the most accurate algorithm on tabular data
+ Handles mixed feature types well
+ Built-in regularization (XGBoost)
- More parameters to tune than Random Forest
- Can overfit if learning_rate is too high or n_estimators too large
- Slower to train than Random Forest
```

---

## §4 — Bias-Variance Tradeoff

This is the core concept that explains why ensembles work.

### Bias

How wrong the model is on average across many different training sets. A model with high bias is too simple — it misses real patterns regardless of what data you give it.

Think of it as: the model has already made up its mind before seeing the data.

```
Example: you have curved data but force a straight line fit.
No matter how much data you give it, the straight line will always
be wrong in the curved region. That wrongness is bias.

High bias = low CV accuracy overall (model can't learn the pattern)
```

### Variance

How much the model's predictions change when you train it on different subsets of data. High variance means the model is too sensitive to the specific training samples it saw.

Think of it as: the model changes its mind completely when it sees slightly different data.

```
Example: decision tree with max_depth=None
- Train on dataset A -> learns one set of rules
- Train on dataset B (slightly different) -> learns completely different rules
- The predictions swing wildly depending on which data it saw

High variance = large gap between train and CV accuracy
             = high CV std (scores swing across folds)
```

### Intuition with a student analogy

```
High bias student:
  Studied the wrong material entirely.
  Fails every exam no matter which questions are asked.
  More studying doesn't help — the approach is wrong.

High variance student:
  Memorized last year's exam answers word for word.
  Aces practice tests perfectly.
  Fails the real exam because the questions are slightly different.

Good student:
  Understood the concepts.
  Performs consistently whether it's a practice test or real exam.
```

### What each metric tells you

```
Train accuracy low + CV accuracy low  = high bias (underfitting)
  -> model too simple, increase complexity

Train accuracy high + CV accuracy low = high variance (overfitting)
  -> large gap = memorizing, not generalizing
  -> high CV std = unstable across folds

Train accuracy high + CV accuracy high = sweet spot
  -> small gap, low std, generalizing well
```

### The tradeoff

```
Simple model  ->  high bias,    low variance   (underfitting)
Complex model ->  low bias,     high variance  (overfitting)
Goal          ->  low bias AND  low variance
```

You can't eliminate both completely — making the model more complex reduces bias but increases variance. The goal is finding the sweet spot where CV accuracy peaks.

### How ensembles fix this

- **Bagging (Random Forest)**: keeps bias the same, reduces variance by averaging. Many complex trees that each overfit, but their errors cancel out across trees.
- **Boosting (XGBoost)**: reduces bias by sequentially fixing errors. Each tree is simple (low variance), but together they model complex patterns.

---

## §5 — Hyperparameter Tuning

Training accuracy is easy — just memorize. The goal is generalizing to new data, which requires finding the right hyperparameters.

### Grid Search

Try every combination of parameters:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, None]
}
# tries all 6 combinations
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(grid.best_params_)
```

### Random Search

Sample random combinations — faster when you have many parameters:

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 10, None],
    'learning_rate': [0.01, 0.05, 0.1, 0.3]
}
# tries 20 random combinations instead of all 64
search = RandomizedSearchCV(XGBClassifier(), param_dist, n_iter=20, cv=5)
```

### When to use which

- Grid search: few parameters, small ranges
- Random search: many parameters, wide ranges (almost always in practice)

---

## Checklist

- [ ] Can explain the difference between bagging and boosting
- [ ] Know why Random Forest rarely overfits
- [ ] Understand how gradient boosting connects to gradient descent from week 4
- [ ] Know what bias and variance mean and how ensembles address each
- [ ] Can run GridSearchCV and RandomizedSearchCV and read the results
