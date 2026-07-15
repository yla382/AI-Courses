# Study Guide — Week 1: Your First ML Model + Linear Algebra I

---

## §1 — What Is Machine Learning (and How Is It Different from Normal Programming)?

### The Core Idea

In normal programming:
```
Rules + Data → Output
```

In machine learning:
```
Data + Output → Rules (learned automatically)
```

You stop writing the logic. You show the machine examples, it figures out the pattern.

### Three Types of ML (Know These Cold)

| Type              | What You Give It         | What It Learns     | Example               |
| ----------------- | ------------------------ | ------------------ | --------------------- |
| **Supervised**    | Labeled examples (X → Y) | A mapping function | Email → spam/not spam |
| **Unsupervised**  | Unlabeled data           | Hidden structure   | Customer segments     |
| **Reinforcement** | Environment + rewards    | A policy/strategy  | Game-playing agents   |

**This week:** supervised learning. It's 80% of what ML engineers do day-to-day.

### Key Vocabulary

- **Feature (X):** Input variable. E.g., email word count, sender reputation.
- **Label (Y):** What you're predicting. E.g., spam=1, not spam=0.
- **Model:** A mathematical function that maps X → Y.
- **Training:** Adjusting the model's internal parameters so its predictions get closer to Y.
- **Inference:** Using the trained model on new, unseen data.

### The ML Workflow (Memorize This Loop)

```
1. Get data
2. Explore & clean data
3. Split into train/test sets
4. Choose a model
5. Train the model
6. Evaluate on test set
7. Iterate
8. Deploy
```

You will repeat this loop for the rest of your career. This week you do it end-to-end for the first time.

### Deep Dive Pointers
- [3Blue1Brown — "But What Is a Neural Network?"](https://www.youtube.com/watch?v=aircAruvnKk) (18 min) — best intuition video for how models learn
- [Scikit-learn docs — Getting Started](https://scikit-learn.org/stable/getting_started.html)

---

## §2 — Linear Algebra I: Vectors & Matrices

*Why now?* Because every dataset you'll ever touch is a matrix. Every feature is a vector. Once you see this, ML code becomes readable at a deeper level.

### Vectors

A vector is an ordered list of numbers. In ML, **one data point = one vector**.

```
patient = [age, weight, blood_pressure, cholesterol]
         = [45,  180,    120,            200]
```

This is a vector in 4-dimensional space. You can think of it as a point, or as an arrow from the origin.

**Operations you need:**
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Addition — combine two vectors
a + b  # [5, 7, 9]

# Scalar multiplication — scale a vector
2 * a  # [2, 4, 6]

# Dot product — measures similarity between two vectors
np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

# Magnitude (length) of a vector
np.linalg.norm(a)  # sqrt(1² + 2² + 3²) = 3.74
```

**The dot product is everywhere in ML.** When a linear model makes a prediction, it's computing a dot product between your features and its learned weights.

### Matrices

A matrix is a 2D array of numbers. In ML, **your entire dataset = one matrix**.

```
X = [
  [age, weight, bp, chol],   # patient 1
  [age, weight, bp, chol],   # patient 2
  ...                        # patient N
]
```

Shape: `(N samples, M features)` — you'll see this everywhere.

**Operations you need:**
```python
A = np.array([[1, 2], [3, 4], [5, 6]])  # shape (3, 2)
B = np.array([[7, 8, 9], [10, 11, 12]]) # shape (2, 3)

# Matrix multiplication — the core operation in every neural network
C = A @ B   # shape (3, 3)
# Rule: (m×n) @ (n×p) = (m×p) — inner dimensions must match

# Transpose — flip rows and columns
A.T  # shape (2, 3)

# Shape inspection (use this constantly)
A.shape   # (3, 2)
```

### Why This Matters for ML

When a linear model predicts on your full dataset at once:
```
predictions = X @ weights + bias
```
- `X` is your data matrix: `(N, M)`
- `weights` is a vector: `(M,)`
- Result: `(N,)` — one prediction per sample

This single line of math is how logistic regression, linear regression, and the first layer of every neural network work. See it in `tutorial.ipynb`.

### Deep Dive Pointers
- [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (chapters 1–4 this week)
- NumPy docs: [Array operations](https://numpy.org/doc/stable/reference/routines.array-manipulation.html)

---

## §3 — What Does "Learning" Actually Mean?

When we say a model "learns," here's what's actually happening:

### Parameters

A model has internal numbers called **parameters** (or weights). For a simple linear model:

```
prediction = w1*feature1 + w2*feature2 + ... + b
```

- `w1, w2, ...` are weights
- `b` is the bias (intercept)
- Learning = finding the right values for these numbers

### Loss Function

A **loss function** measures how wrong the model is. The most common for regression:

```
MSE = (1/N) × Σ(prediction - actual)²
```

Low loss = good model. High loss = bad model. Training is the process of minimizing loss.

### The Learning Loop

```
1. Make prediction with current weights
2. Compute loss (how wrong were we?)
3. Adjust weights to reduce loss
4. Repeat until loss is small enough
```

Step 3 is where calculus comes in — you'll build the math for this in Week 4. For now, trust that scikit-learn handles it internally.

### Overfitting vs. Underfitting

```
Underfitting: model too simple → bad on training data AND test data
Overfitting:  model too complex → great on training data, bad on test data
Goal:         generalizes well to data it's never seen
```

This is why you always keep a **test set** separate — it's the only honest measure of your model.

### Deep Dive Pointers
- [StatQuest — Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA)
- [Scikit-learn — Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)

---

## Week 1 Checklist

- [ ] Can explain supervised vs. unsupervised to a non-technical person
- [ ] Know what features, labels, training, and inference mean
- [ ] Can create and manipulate vectors/matrices with NumPy
- [ ] Understand what a dot product computes and why it matters
- [ ] Trained at least one classifier with scikit-learn
- [ ] Understand what loss is and what training is trying to minimize
