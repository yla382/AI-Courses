# Week 10 Study Guide — PyTorch Fundamentals

Work through each section, then immediately do that section in `tutorial.ipynb` before moving on.

---

## The Big Shift

In week 9 you built a neural network from scratch with numpy, then used sklearn's MLPClassifier. Both have limits:

- **numpy from scratch**: you had to write every gradient manually — not scalable to real networks
- **sklearn MLP**: black box — you can't change the architecture, loss function, or training loop

PyTorch sits in the middle: it computes gradients automatically (no more manual backprop), while giving you full control over everything else. This is what the industry actually uses.

Everything from weeks 9 still applies:
- Same forward pass logic (z = XW + b → activation)
- Same loss functions (log loss, cross entropy)
- Same gradient descent loop
- Same backprop — PyTorch just does it for you automatically

The only new thing: PyTorch's way of expressing those ideas.

---

## §1 — Tensors

A tensor is PyTorch's version of a numpy array. Same idea, but can run on GPU and tracks gradients.

```python
import torch
import numpy as np

# creating tensors
x = torch.tensor([1.0, 2.0, 3.0])
x = torch.zeros(3, 4)
x = torch.randn(3, 4)        # random normal

# from numpy
arr = np.array([1.0, 2.0, 3.0])
x = torch.from_numpy(arr)

# to numpy
arr = x.numpy()
```

### Key tensor attributes

```python
x = torch.randn(3, 4)
x.shape       # torch.Size([3, 4])
x.dtype       # torch.float32
x.device      # cpu
```

### Tensor operations

Same as numpy:
```python
a = torch.randn(3, 4)
b = torch.randn(4, 2)

a + 1          # add scalar
a * 2          # multiply scalar
a @ b          # matrix multiply → (3, 2)
a.T            # transpose → (4, 3)
a.mean()       # mean of all elements
a.sum(dim=0)   # sum along rows → (4,)
```

### GPU transfer

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = x.to(device)
```

Everything needs to be on the same device — model and data both on GPU, or both on CPU.

---

## §2 — Autograd

This is the core of PyTorch. Instead of computing gradients by hand (week 9), PyTorch tracks every operation and computes gradients automatically.

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2        # y = x²
y.backward()      # compute dy/dx
print(x.grad)     # tensor(6.)  ← dy/dx = 2x = 2×3 = 6
```

`requires_grad=True` tells PyTorch: track every operation on this tensor so you can compute gradients later.

### How it works

PyTorch builds a **computation graph** as you run operations:

```
x → (square) → y → (backward) → gradient flows back to x
```

When you call `.backward()`, PyTorch walks the graph backwards and applies the chain rule automatically. This is exactly what you did manually in week 9 — PyTorch just does it for you.

### Gradient accumulation

Gradients accumulate by default — you must zero them before each backward pass:

```python
optimizer.zero_grad()   # clear gradients from previous step
loss.backward()         # compute new gradients
optimizer.step()        # update weights
```

If you forget `zero_grad()`, gradients from the previous batch add to the current batch — weights update incorrectly.

---

## §3 — nn.Module

`nn.Module` is the base class for all PyTorch models. You define your network by subclassing it.

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet(input_size=3, hidden_size=4, output_size=1)
```

### nn.Linear

`nn.Linear(in, out)` is exactly `z = XW + b` — it holds the weight matrix and bias and applies them.

```python
layer = nn.Linear(3, 4)
layer.weight.shape   # (4, 3)  ← note: PyTorch stores W transposed vs week 9
layer.bias.shape     # (4,)
```

### nn.Sequential

For simple networks where layers go one after another:

```python
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)
```

Same network as week 9's manual numpy version, now in 4 lines.

### Output layer — raw logits

The output layer of `forward` should always return **raw logits** — the unscaled scores before any activation is applied. For example `[2.1, 0.3, -1.4]` for a 3-class problem.

You do not apply activation functions (sigmoid for binary, softmax for multiclass) inside `forward`. The reason:

- **During training**: PyTorch's loss functions (`BCEWithLogitsLoss`, `CrossEntropyLoss`) apply the activation internally for numerical stability — doing sigmoid/softmax then log loss separately loses floating point precision. So the loss expects raw logits.
- **During prediction**: you apply the activation manually after the forward pass to get probabilities if you need them.

```python
model.eval()
with torch.no_grad():
    logits = model(X)                        # raw scores: [2.1, 0.3, -1.4]
    probs  = torch.softmax(logits, dim=1)    # probabilities: [0.78, 0.17, 0.05]
    preds  = torch.argmax(logits, dim=1)     # predicted class: 0
```

For predicting the class label you don't even need the activation — `argmax` picks the highest score, and the highest raw score is always the same class as the highest probability. You only need the activation when you want the actual probability values (e.g. reporting confidence to a user).

---

## §4 — Training Loop

The training loop in PyTorch is explicit — you write it yourself. This is more code than sklearn, but gives you full control.

Before the loop you set up two things:
```python
criterion = nn.CrossEntropyLoss()                    # loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)  # weight update rule
```

`criterion` is just a variable name convention for the loss function — you could call it anything. `optimizer` holds the algorithm that updates weights using the gradients.

```python
# standard PyTorch training loop
for epoch in range(num_epochs):
    model.train()                     # set training mode
    
    optimizer.zero_grad()             # 1. clear gradients
    
    y_pred = model(X_train)           # 2. forward pass
    loss = criterion(y_pred, y_train) # 3. compute loss
    
    loss.backward()                   # 4. backward pass (autograd)
    optimizer.step()                  # 5. update weights
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

Compare to week 9's manual version:
```
week 9 numpy:    you computed dL_dW2, dL_dW1, etc. by hand
week 10 pytorch: loss.backward() does all of that automatically
```

### Loss functions

```python
nn.BCEWithLogitsLoss()   # sigmoid + binary cross-entropy (binary classification)
nn.CrossEntropyLoss()    # softmax + cross-entropy (multiclass — 3+ classes)
nn.MSELoss()             # mean squared error (regression)
```

Each loss applies its activation internally — this is why `forward` returns raw logits and you don't add sigmoid or softmax to the output layer.

They also expect different y shapes:
```python
# BCEWithLogitsLoss — binary
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # shape (batch, 1)

# CrossEntropyLoss — multiclass
y = torch.tensor(labels, dtype=torch.long)                  # shape (batch,) — class indices, no unsqueeze
```

### Optimizers

The optimizer is the algorithm that updates the model's weights after each backward pass. You pass it `model.parameters()` so it knows which weights to update, and `lr` (learning rate) to control step size.

```python
optim.SGD(model.parameters(), lr=0.01)               # basic gradient descent
optim.Adam(model.parameters(), lr=0.001)             # adaptive learning rate (default choice)
optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # SGD with momentum
```

Adam is the default for most problems — it adapts the learning rate per parameter, so you don't need to tune it as carefully.

---

## §5 — model.train() vs model.eval()

PyTorch models have two modes that affect how certain layers behave:

**`model.train()`** — training mode:
- Gradients are tracked (needed for backprop)
- Layers like Dropout randomly zero out neurons to prevent overfitting
- BatchNorm computes statistics from the current batch

**`model.eval()`** — evaluation mode:
- Dropout is turned off — all neurons are active, giving deterministic predictions
- BatchNorm uses statistics collected during training instead of the current batch
- Combined with `torch.no_grad()`, no computation graph is built — faster and uses less memory

You need `model.eval()` whenever you are measuring validation/test loss or making predictions — so the measurement is honest and consistent. Without it, layers like Dropout are still randomly zeroing neurons, which means every forward pass gives a slightly different result and your loss measurement is noisy and unreliable.

For the networks in this week (no Dropout or BatchNorm) the difference is minimal, but it's still good practice to always switch modes correctly — it matters a lot in week 12+.

```python
# training
model.train()
loss.backward()
optimizer.step()

# evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
```

---

## §6 — DataLoader

For large datasets you don't load everything at once — you use a DataLoader to batch the data:

```python
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_tensor, y_tensor)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

for X_batch, y_batch in loader:
    optimizer.zero_grad()
    y_pred = model(X_batch)
    loss = criterion(y_pred, y_batch)
    loss.backward()
    optimizer.step()
```

`shuffle=True` randomizes the order each epoch — prevents the model from memorizing the order of samples.

---

## Putting It Together

```
week 9 numpy:
    manually compute forward pass
    manually compute every gradient
    manually update weights

week 9 sklearn:
    fit() does everything — black box

week 10 pytorch:
    you write the forward pass (via nn.Module)
    autograd computes gradients automatically
    you write the training loop
    optimizer updates weights
```

PyTorch is the balance: you control the architecture and loop, autograd handles the math.

---

## Further Watching

- [PyTorch in 100 Seconds](https://www.youtube.com/watch?v=ORMx45xqWkA) (2 min) — fastest possible overview
- [Andrej Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0) (2h 25min) — builds autograd from scratch, deepens understanding of what PyTorch does under the hood
- [PyTorch Official — Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) — official tutorial series, good reference

---

## Checklist

- [ ] Can create tensors and perform basic operations
- [ ] Understand what `requires_grad=True` does and why
- [ ] Know the 5-step training loop: zero_grad → forward → loss → backward → step
- [ ] Can define a model using `nn.Module` or `nn.Sequential`
- [ ] Know the difference between `model.train()` and `model.eval()`
- [ ] Can use `DataLoader` to batch data
- [ ] Understand what Adam does differently from plain SGD
