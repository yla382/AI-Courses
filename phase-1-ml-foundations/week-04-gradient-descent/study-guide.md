# Week 4 Study Guide — Gradient Descent from Scratch

Work through each section, then immediately do that section in `tutorial.ipynb` before moving on.

---

## What Is Gradient Descent?

Every time you call `.fit()` on a model, something has to figure out the right parameters — the weights that make the model's predictions as accurate as possible. That something is **gradient descent**.

Here's the core idea. Imagine you're blindfolded on a hilly landscape and you want to get to the lowest point. You can't see the whole landscape — you can only feel the slope under your feet right now. So you take a small step in the direction that goes downhill, feel the slope again, take another step, and repeat until it's flat under your feet.

That's gradient descent:
1. You have a **loss function** — a number measuring how wrong your model is (lower = better)
2. You compute the **gradient** — which direction increases the loss (the slope)
3. You step in the **opposite direction** — downhill, toward lower loss
4. Repeat until the loss stops improving

```
Start with random weights
↓
Make predictions → compute loss (how wrong am I?)
↓
Compute gradient (which way is uphill?)
↓
Step downhill (update weights)
↓
Repeat → loss keeps decreasing → model keeps improving
```

The rest of this week builds up the math you need to actually implement this — derivatives first, then loss functions, then the full algorithm.

---

## §1 — What Is a Derivative?

A derivative tells you the **slope** of a function at a specific point — how fast the output is changing as the input changes.

```
f(x) = x²

At x = 2:  slope = 4   (output rising fast)
At x = 0:  slope = 0   (flat — at the bottom)
At x = -3: slope = -6  (output falling)
```

The slope tells you two things:
- **Sign** (+ or −): which direction is downhill
- **Magnitude**: how steep it is

The derivative of f(x) = x² is f'(x) = 2x. That's the formula for the slope at any x.

**Why it matters for ML:** your model has a loss function (how wrong it is). The derivative tells you which direction to move the parameters to make the loss smaller.

### The difference quotient — where derivatives come from

```
f'(x) = lim   f(x + h) - f(x)
        h→0   ───────────────
                     h
```

This is just "rise over run" as the run shrinks to zero. You don't need to memorize this — sklearn and PyTorch compute derivatives for you — but seeing it once makes the concept concrete.

---

## §2 — Loss Functions

Before gradient descent can run, you need something to minimize. That's the **loss function** — a number that measures how wrong your model is.

### Mean Squared Error (MSE)

For regression (predicting a number):

```
MSE = (1/n) × Σ (predicted - actual)²
```

- Squaring makes all errors positive
- Larger errors are penalized more (squaring amplifies them)
- Lower MSE = better model

**Example:**
```
Actual:    [3, 5, 7]
Predicted: [2, 6, 6]
Errors:    [-1, +1, -1]
Squared:   [1, 1, 1]
MSE = (1+1+1)/3 = 1.0
```

### The loss surface

Imagine your model has one parameter `w`. For every possible value of `w`, you get a different MSE. Plot that — you get a curve (the loss surface). Gradient descent's job is to find the bottom of that curve.

With two parameters (w and b), the loss surface is a 3D bowl. Same idea — find the lowest point.

---

## §3 — Gradient Descent

The algorithm:

```
1. Start at a random point on the loss surface
2. Compute the gradient (slope) at that point
3. Take a small step in the opposite direction (downhill)
4. Repeat until the loss stops improving
```

In math:
```
w = w - learning_rate × gradient
```

The minus sign is key — gradient points **uphill**, so you subtract it to go downhill.

### Learning rate

Controls how big each step is.

```
Too large:  overshoots the minimum, loss bounces around or diverges
Too small:  takes forever to converge
Just right: smooth descent to the minimum
```

Typical starting values: 0.001, 0.01, 0.1. You tune it based on how loss behaves.

### Gradient for linear regression

For a linear model `y_pred = w × x + b`, the gradients are:

```
dL/dw = (2/n) × Σ (y_pred - y_actual) × x
dL/db = (2/n) × Σ (y_pred - y_actual)
```

**What each formula means:**

`dL/dw` — "how does the loss change if I increase w slightly?"
- `(y_pred - y_actual)` is the error for each data point
- You multiply by `x` because w and x are multiplied in the model — changing w has more impact when x is large
- Average over all n points with `(2/n) × Σ`
- If this is positive → loss goes up when w increases → decrease w
- If this is negative → loss goes down when w increases → increase w

`dL/db` — "how does the loss change if I increase b slightly?"
- Same idea but no `× x` because b is added directly, not multiplied by anything
- Just the average error across all points

**How to use them to update w and b:**

```
w_new = w - learning_rate × dL/dw
b_new = b - learning_rate × dL/db
```

Subtract because the gradient points uphill — you want to go the opposite direction.

---

## §4 — Implementing Linear Regression from Scratch

Linear regression finds the best line through data:

```
y_pred = w × x + b
```

- `w` = weight (slope of the line)
- `b` = bias (intercept)

Training means finding the `w` and `b` that minimize MSE using gradient descent.

The full training loop:

```python
for each epoch:
    y_pred = w * X + b           # forward pass
    loss = MSE(y_pred, y_true)   # compute loss
    dw, db = gradients(...)      # compute gradients
    w = w - lr * dw              # update w
    b = b - lr * db              # update b
```

This exact pattern — forward pass, loss, gradients, update — is how every neural network trains. PyTorch automates steps 3 and 4, but the structure is identical.

---

## §5 — Chain Rule (Why Backprop Works)

**What is backpropagation?**

A neural network is many layers of functions stacked on top of each other — input goes through layer 1, then layer 2, then layer 3, and so on until you get a prediction. To train it, you need the gradient of the loss with respect to every single weight in every layer. That's potentially millions of gradients.

Backpropagation is the algorithm that computes all those gradients efficiently. It works by starting at the output (where you know the loss) and working **backwards** through each layer, using the chain rule at each step to compute that layer's gradients from the layer ahead of it. That's where the name comes from — propagating gradients backward through the network.

The chain rule is the math that makes this possible. Without it, you'd have no way to compute how a weight in layer 1 affects the final loss. You'll implement backprop from scratch in week 9 — for now, understanding chain rule is the foundation.

When functions are chained together, the derivative of the whole thing is the product of the individual derivatives.

```
If z = f(g(x)), then dz/dx = f'(g(x)) × g'(x)
```

**Concrete example:**

```
x → (×2) → u → (square) → z

z = (2x)²

dz/dx = dz/du × du/dx
      = 2u    × 2
      = 2(2x) × 2
      = 8x
```

**Why it matters:** a neural network is many functions chained together. The chain rule lets you compute the gradient at any layer by multiplying gradients layer by layer — that's exactly what backpropagation does.

### Gradient checking

How do you know your chain rule implementation is correct? You can verify it numerically using the **finite difference** approximation:

```
numerical gradient ≈ (f(x + h) - f(x - h)) / (2h)
```

where `h` is a tiny number like `0.00001`. This is a brute-force estimate of the slope — nudge `x` slightly right, nudge it slightly left, see how much the output changes.

If your analytical gradient (from chain rule) matches the numerical gradient closely, your math is correct. This is called a **gradient check** and is a standard debugging tool when implementing backprop from scratch.

---

## §6 — What sklearn `.fit()` is Actually Doing

When you call `model.fit(X, y)` on a LogisticRegression or LinearRegression, sklearn is running gradient descent (or a close relative) under the hood:

1. Initialize weights randomly
2. Make predictions on the training data
3. Compute the loss
4. Compute gradients
5. Update weights
6. Repeat for many iterations

This week you've built that loop yourself. From now on when you call `.fit()`, you know what's happening inside.

---

## Checklist

- [ ] Can explain what a derivative tells you in plain words
- [ ] Can compute MSE by hand from predictions and actuals
- [ ] Can describe the gradient descent update rule (w = w - lr × gradient)
- [ ] Know what happens when learning rate is too large or too small
- [ ] Can implement a training loop from scratch
- [ ] Can explain what the chain rule is and why it matters for neural networks
