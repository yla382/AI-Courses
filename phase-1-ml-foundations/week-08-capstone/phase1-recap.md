# Phase 1 Recap — ML Foundations

Quick reference before moving into Phase 2. Skim this, not study it.

---

## The Big Picture

Every ML model does the same thing:
```
features (X) -> model -> prediction (y_pred)
```

Training = finding weights that minimize loss. That's it. Everything else is details.

---

## Week 1-2 — Data Foundations

**What you learned:**
- NumPy: arrays, matrix operations, dot products
- Pandas: DataFrames, groupby, merge, handling missing values
- EDA: histograms, correlation, describe()

**Key formulas:**
```
mean     = sum(x) / n
std      = sqrt(1/n * sum((x - mean)^2))
scaled   = (x - mean) / std        (StandardScaler)
```

**When scaling matters:** kNN, SVM, Logistic Regression, gradient descent. Tree models don't care.

---

## Week 3 — Model Evaluation

**The core metrics:**
```
Precision = TP / (TP + FP)    "of all predicted positive, how many were right?"
Recall    = TP / (TP + FN)    "of all actual positive, how many did we catch?"
F1        = 2 * P * R / (P+R) "harmonic mean of both"
```

**When to use which:**
```
Cancer screening  -> maximize Recall   (missing a case is catastrophic)
Spam filter       -> maximize Precision (false positives are annoying)
Balanced dataset  -> accuracy is fine
```

**Cross-validation:** always run CV on training data only. Test set touched once at the very end.

**Bayes theorem:**
```
P(disease | positive) = P(positive | disease) * P(disease) / P(positive)
```

---

## Week 4 — Gradient Descent

**The training loop (everything uses this):**
```
1. z = X*W + b               (forward pass)
2. compute loss from z and y (how wrong are we)
3. dL/dW, dL/db              (gradients)
4. W = W - lr * dL/dW        (step downhill)
5. b = b - lr * dL/db
   repeat
```

**Key concepts:**
```
learning rate too high -> overshoots, diverges
learning rate too low  -> converges too slowly
MSE loss   = 1/n * sum((y_pred - y)^2)
dL/dW      = 2/n * X^T * (y_pred - y)
dL/db      = 2/n * sum(y_pred - y)
```

**Chain rule:** multiply derivatives of each step. Foundation of backprop.

---

## Week 5 — Classical ML

**kNN:**
- No training. At prediction time, find k nearest neighbors, majority vote.
- Always scale. k too small = overfit, k too large = underfit.

**Logistic Regression:**
```
z = X*W + b
p = 1 / (1 + e^(-z))     (sigmoid)
predict 1 if p >= 0.5

Loss = -1/n * sum(y*log(p) + (1-y)*log(1-p))   (log loss)
```
Same gradient descent loop as week 4, just sigmoid + log loss instead of linear + MSE.

**SVM:**
- Finds maximum margin boundary between classes
- Kernel trick: rbf for curved boundaries, linear for straight
- C parameter: small C = wide margin (generalize), large C = narrow margin (overfit risk)

**Naive Bayes:**
- Uses Bayes theorem. Assumes features are independent (naive).
- Fast, works well on small data and text. Bad probability estimates.

**PCA:**
- Finds directions of highest variance, projects data onto them
- Use for visualization, dimensionality reduction, removing noise
- `explained_variance_ratio_` tells you how much each component kept

**Algorithm selection:**
```
Small dataset          -> kNN, Naive Bayes
Need interpretability  -> Logistic Regression
Non-linear boundary    -> SVM (rbf kernel)
Text classification    -> Naive Bayes
Speed matters          -> Logistic Regression
```

---

## Week 6 — Ensemble Methods

**Bias-Variance tradeoff:**
```
High bias (underfit):   train low, CV low, small gap    -> model too simple
High variance (overfit): train high, CV low, large gap  -> model memorizing
Sweet spot:             train high, CV high, small gap
```

**Random Forest (bagging):**
- Many trees, each on random subset of data + features
- Majority vote at prediction time
- Reduces variance. Rarely overfits. Built-in feature importance.

**Gradient Boosting (XGBoost):**
- Trees trained sequentially, each fixing errors of previous
- Same idea as gradient descent but each "step" is a tree
- More accurate than RF on tabular data. More parameters to tune.
- `learning_rate` controls step size. Lower lr + more trees = better but slower.

**Hyperparameter tuning:**
```
GridSearchCV      -> try every combination (small param grids)
RandomizedSearchCV -> try random combinations (large param grids, faster)
```

---

## Week 7 — Feature Engineering

**Encoding:**
```
< 10 categories   -> one-hot (pd.get_dummies)
ordered categories -> ordinal (map to 0,1,2...)
50+ categories    -> target encoding (replace with mean target per category)
```

**Log transform:**
```
log_feature = log(1 + x)    (np.log1p)
```
Use on right-skewed features. Helps distribution-sensitive models (Linear Regression, Naive Bayes, Logistic Regression). Tree models don't need it.

**Feature interactions:**
```
rooms_per_person = total_rooms / population
```
Domain knowledge > automated polynomial features.

**Regularization:**
```
L2 (Ridge): Loss + lambda * sum(w^2)   -> shrinks weights, keeps all features
L1 (Lasso): Loss + lambda * sum(|w|)   -> drives some weights to zero (feature selection)

In sklearn:
  Ridge/Lasso:         alpha = lambda  (larger alpha = more regularization)
  LogisticRegression:  C = 1/lambda    (smaller C = more regularization)
```

Large weights = model betting too much on one feature = overfitting.
Regularization keeps weights small = smoother boundary = better generalization.

**Feature selection:**
```
SelectKBest  -> score each feature, keep top k
RFE          -> recursively remove least important
L1 (Lasso)   -> automatically zeroes out irrelevant features
RF importances -> rank by contribution across all trees
```

---

## The Full Training Pipeline

```
Raw data
   |
   v
EDA + clean (handle missing, check distributions)
   |
   v
Feature engineering (encode, transform, interactions)
   |
   v
Train/test split (stratify if classification)
   |
   v
Scale if needed (kNN, SVM, LogReg, gradient descent)
   |
   v
Cross-validate on train set (pick algorithm + hyperparams)
   |
   v
Final evaluation on test set (once, at the very end)
   |
   v
Classification report / confusion matrix / R2
```

---

## What Phase 2 Builds On

- Gradient descent (week 4) -> backpropagation in neural networks
- Log loss (week 5) -> same loss used in neural network output layers
- Chain rule (week 4) -> how gradients flow through layers
- Bias-variance (week 6) -> dropout, batch norm, regularization in deep learning
- Feature engineering (week 7) -> learned automatically by deep networks
