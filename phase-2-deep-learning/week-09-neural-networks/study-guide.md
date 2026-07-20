# Week 9 Study Guide — Neural Networks

Work through each section, then immediately do that section in `tutorial.ipynb` before moving on.

---

## The Big Shift

In phase 1 you used algorithms that were designed by humans — kNN measures distance, decision trees split on thresholds, logistic regression draws a line. The parameters were interpretable.

Neural networks are different. Instead of designing the algorithm, you design the architecture — the structure — and let the network figure out the algorithm itself through training. The parameters (weights) are not interpretable, but the network can learn patterns too complex for any human to design manually.

Everything from phase 1 still applies:
- Same gradient descent loop (week 4)
- Same log loss (week 5)
- Same bias-variance tradeoff (week 6)

The only new thing: instead of one layer of weights, you stack many layers.

---

## §1 — From Logistic Regression to a Neural Network

### Logistic regression recap

```
input x -> z = w*x + b -> sigmoid(z) -> prediction
```

One layer, one set of weights. Can only learn linear boundaries.

### One hidden layer

```
input x -> [hidden layer] -> [output layer] -> prediction
```

The hidden layer is just logistic regression applied multiple times in parallel:

```
z1 = w11*x1 + w12*x2 + b1    \
z2 = w21*x1 + w22*x2 + b2     -> activation -> output layer -> prediction
z3 = w31*x1 + w32*x2 + b3    /
```

Each node in the hidden layer learns a different linear combination of inputs, then an activation function adds non-linearity.

### Why multiple layers?

- 1 layer: can learn any function given enough neurons (universal approximation theorem)
- Multiple layers: can learn hierarchical features more efficiently

Example for image recognition:
```
Layer 1: learns edges (horizontal, vertical, diagonal)
Layer 2: learns shapes (circles, corners, curves)
Layer 3: learns parts (eyes, wheels, windows)
Layer 4: learns objects (face, car, building)
```

---

## §2 — Activation Functions

### What is an activation function?

An activation function is a function applied to each node's output that decides how strongly that node "fires" given its input.

Without it, each node just does `z = XW + b` — a straight line. The problem: stacking straight lines on top of each other still gives you a straight line. No matter how many layers you add, the whole network collapses into a single linear equation. You can't learn curves, boundaries, or any complex pattern.

The activation function bends that straight line (non-linear just means "not a straight line" — anything with a curve or bend). Once you have a bend, stacking layers lets you learn increasingly complex shapes.

**Concrete example:**

```
Without activation (useless stacking):
Layer 1: z1 = 2x + 1
Layer 2: z2 = 3*z1 + 1 = 3*(2x+1) + 1 = 6x + 4

That's still just one line: 6x + 4
No matter how many layers, you get one line.

With ReLU activation:
Layer 1: a1 = ReLU(2x + 1)   <- bends at x = -0.5 (where 2x+1 = 0)
Layer 2: z2 = 3*a1 + 1       <- applies another transformation to that bent line

Now you have a piecewise shape, not a straight line.
Stack more layers -> more bends -> more complex shapes.
```

So activation functions are what make depth useful.

### Sigmoid

```
sigma(z) = 1 / (1 + e^(-z))   output range: (0, 1)
```

You already know this from logistic regression. Problem: for deep networks it causes vanishing gradients (gradients become tiny, early layers stop learning).

### ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)   output range: (0, infinity)
```

Most common activation for hidden layers. Simple, fast, doesn't saturate for positive values.

```
z = -3  ->  ReLU = 0
z =  0  ->  ReLU = 0
z =  5  ->  ReLU = 5
```

### Leaky ReLU

```
LeakyReLU(z) = z if z > 0 else 0.01 * z
```

Fixes the "dying ReLU" problem — neurons stuck at 0 forever. The small slope for negative values keeps gradients flowing.

### Softmax (output layer for multiclass)

```
softmax(z_i) = e^(z_i) / sum(e^(z_j) for all j)
```

Converts a vector of raw scores into probabilities that sum to 1. Used in the output layer when predicting more than 2 classes.

```
raw scores: [2.0, 1.0, 0.5]
softmax:    [0.63, 0.23, 0.14]   (sums to 1.0)
```

### Which to use

```
Hidden layers:   ReLU (default)
Output layer:    sigmoid (binary classification)
                 softmax (multiclass classification)
                 none / linear (regression)
```

---

## §3 — Forward Pass

The forward pass is computing a prediction given current weights. Data flows left to right through the network.

### Example: 2-layer network

```
Input:   x = [x1, x2]             (2 features)
Layer 1: z1 = W1*x + b1            (matrix multiply)
         a1 = ReLU(z1)             (activation)
Layer 2: z2 = W2*a1 + b2           (matrix multiply)
         a2 = sigmoid(z2)          (output activation)
Output:  prediction = a2
```

`W1` is a matrix of shape `(hidden_size, input_size)`. Every neuron in layer 1 has one weight per input feature.

### Shapes matter

```
Input x:    (batch_size, n_features)
W1:         (n_features, hidden_size)
z1 = x @ W1 + b1:  (batch_size, hidden_size)
a1 = ReLU(z1):      (batch_size, hidden_size)
W2:         (hidden_size, 1)
z2 = a1 @ W2 + b2:  (batch_size, 1)
output = sigmoid(z2): (batch_size, 1)
```

Understanding shapes is essential for debugging neural networks.

---

## §4 — Backpropagation

After the forward pass you have a prediction and a loss. Backpropagation computes the gradient of the loss with respect to every weight in the network, using the chain rule from week 4.

### The chain rule revisited

```
Loss L depends on a2
a2 depends on z2
z2 depends on W2 and a1
a1 depends on z1
z1 depends on W1 and x
```

To get dL/dW1 you multiply the chain of derivatives:

```
dL/dW1 = dL/da2 * da2/dz2 * dz2/da1 * da1/dz1 * dz1/dW1
```

This is backpropagation — applying the chain rule backwards through every layer.

### Why it's called backprop

You compute gradients starting from the output (where loss is) and working backwards to the input. Each layer receives the gradient from the layer ahead of it, multiplies by its local derivative, and passes the result to the layer behind it.

```
forward:   x -> layer1 -> layer2 -> loss
backward:  dL/dW1 <- layer1 <- layer2 <- dL/da2
```

### Vanishing gradients

The gradient gets multiplied at every layer. If the activation function squashes gradients (like sigmoid), multiplying many small numbers together makes the gradient tiny by the time it reaches early layers. Those layers barely update — they stop learning.

This is why:
- ReLU replaced sigmoid in hidden layers (gradient is 1 for positive inputs, doesn't shrink)
- Very deep networks need batch normalization or residual connections (week 12)

---

## §5 — Training a Neural Network

The training loop is the same as week 4, just applied to many more parameters:

```
For each epoch:
    for each batch:
        1. forward pass: compute predictions
        2. compute loss
        3. backward pass: compute gradients (backprop)
        4. update all weights: w = w - lr * gradient
```

### Key hyperparameters

```
hidden_size    — number of neurons per layer
n_layers       — number of hidden layers
learning_rate  — step size for gradient descent
batch_size     — how many samples per weight update
epochs         — how many times to go through all data
```

### Batch size

Instead of updating weights after every single sample (slow) or after the entire dataset (less stable), you update after each mini-batch:

```
batch_size = 32  -> compute gradient on 32 samples, update, repeat
```

This is called Stochastic Gradient Descent (SGD) or mini-batch gradient descent.

### Loss functions

```
Binary classification:   Binary Cross-Entropy (log loss from week 5)
Multiclass:              Cross-Entropy (softmax + log loss)
Regression:              MSE
```

---

## §6 — Neural Networks in sklearn vs PyTorch

This week you'll build neural networks in two ways:

### sklearn MLPClassifier

Quick and familiar — same API as everything in phase 1:

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # 2 hidden layers: 64 then 32 neurons
    activation='relu',
    max_iter=500,
    random_state=42
)
model.fit(X_train, y_train)
```

Good for: quick experiments, tabular data, when you don't need custom architecture.

### PyTorch (next week)

Full control over architecture, training loop, custom loss functions. Required for anything beyond simple feedforward networks.

This week: sklearn. Week 10: PyTorch.

---

## Further Watching

These videos build intuition for the concepts in this week. Watch in order.

**Neural Networks — Core Intuition**
- [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) (19 min) — the best visual explanation of what a neural network actually is. Watch this first.
- [3Blue1Brown — Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) (21 min) — connects week 4 gradient descent to neural network training visually.

**Backpropagation**
- [3Blue1Brown — Backpropagation, the chain rule](https://www.youtube.com/watch?v=tIeHLnjs5U8) (10 min) — visual walkthrough of how gradients flow backwards through layers.
- [Andrej Karpathy — The spelled-out intro to neural networks (micrograd)](https://www.youtube.com/watch?v=VMj-3S1tku0) (2h 25min) — builds backprop from scratch in Python. Long but extremely valuable. Watch after finishing the tutorial.

**Activation Functions and Vanishing Gradients**
- [StatQuest — Neural Networks Part 3: ReLU](https://www.youtube.com/watch?v=68BZ5f7P94E) (10 min) — clear explanation of why ReLU replaced sigmoid in hidden layers.

---

## Checklist

- [ ] Can explain why stacking linear layers without activation functions is pointless
- [ ] Know what ReLU does and why it's preferred over sigmoid in hidden layers
- [ ] Can trace a forward pass through a 2-layer network and track tensor shapes
- [ ] Understand what backpropagation is computing and why it uses the chain rule
- [ ] Know what vanishing gradients are and which activation function causes them
- [ ] Can train an MLPClassifier and tune hidden layer sizes
