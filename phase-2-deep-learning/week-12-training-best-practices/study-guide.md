# Week 12 Study Guide — Training Best Practices

Work through each section, then immediately do that section in `tutorial.ipynb` before moving on.

---

## The Big Shift

In week 11 you built a CNN that hit 93.8% on EMNIST letters. You also saw overfitting — train loss kept dropping while val loss flattened and crept back up. You couldn't do much about it at the time.

This week you fix that. The techniques here are the difference between a model that barely trains and one that converges fast and generalizes well. Every professional training run uses most or all of these.

---

## §1 — Why Training Goes Wrong

Before fixing problems, you need to recognize them. There are two common failure modes:

### Overfitting

Train loss keeps dropping, val loss stops improving or goes back up. The model is memorizing training data instead of learning general patterns.

```
epoch 1:  train=0.50  val=0.29   good — both going down
epoch 5:  train=0.15  val=0.20   good — converging
epoch 10: train=0.09  val=0.23   bad  — gap growing, overfitting
```

You saw exactly this in the EMNIST capstone.

### Unstable training

Loss jumps around instead of decreasing smoothly. Often caused by learning rate being too high, or activations growing too large as they pass through layers.

```
epoch 1:  train=0.80
epoch 2:  train=0.45
epoch 3:  train=0.91   <- jumped back up
epoch 4:  train=0.38
```

Both problems have specific fixes:
```
overfitting       -> Dropout (§2), Early Stopping (§5)
unstable training -> Batch Normalization (§3), Learning Rate Scheduling (§4)
```

---

## §2 — Dropout

Dropout randomly sets a fraction of neurons to zero during each forward pass in training. Each neuron has a probability `p` of being zeroed out.

```
without dropout:   all neurons active every forward pass
with dropout(p=0.5): each neuron has 50% chance of being zeroed each forward pass
```

### Why it prevents overfitting

When neurons are randomly dropped, the network can't rely on any single neuron to always be there. It's forced to learn redundant representations — multiple neurons learning the same pattern from different angles. This makes the model more robust and less likely to memorize specific training examples.

### How to use it in PyTorch

```python
nn.Dropout(p=0.5)   # p = probability of zeroing a neuron
```

Dropout is placed after activation functions, typically between Linear layers:

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1     = nn.Linear(1600, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2     = nn.Linear(256, 10)
        self.relu    = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)        # randomly zero 50% of neurons
        x = self.fc2(x)
        return x
```

### Dropout and model.train() / model.eval()

This is why `model.train()` and `model.eval()` matter:

- `model.train()` — Dropout is active, neurons are randomly dropped
- `model.eval()` — Dropout is turned off, all neurons are active

During inference you want all neurons active so predictions are deterministic. This is why you must always call `model.eval()` before evaluating or predicting.

### Choosing p

```
p=0.5   most common for fully connected layers
p=0.25  lighter dropout, use when model is not heavily overfitting
p=0.1   very light, sometimes used after conv layers
```

Don't use Dropout after the output layer — that would randomly zero your predictions.

---

## §3 — Batch Normalization

Batch Normalization (BatchNorm) normalizes the activations within each mini-batch — it keeps values in a consistent range throughout training.

### The problem it solves

As data passes through many layers, the activations can grow very large or very small. This makes training unstable — gradients either explode or vanish. BatchNorm fixes this by normalizing activations at each layer.

### What it does mathematically

For each mini-batch, BatchNorm computes:

```
1. mean of the batch:      mu = mean(x)
2. variance of the batch:  var = variance(x)
3. normalize:              x_norm = (x - mu) / sqrt(var + epsilon)
4. scale and shift:        output = gamma * x_norm + beta
```

`gamma` and `beta` are learned parameters — the network learns how much to scale and shift after normalization. `epsilon` is a small number (1e-5) added to prevent division by zero.

### How to use it in PyTorch

```python
nn.BatchNorm2d(num_features)   # after Conv2d layers — num_features = out_channels
nn.BatchNorm1d(num_features)   # after Linear layers — num_features = layer output size
```

BatchNorm goes between the linear/conv operation and the activation function:

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(32)    # 32 = out_channels of conv1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2   = nn.BatchNorm2d(64)    # 64 = out_channels of conv2
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))   # conv -> batchnorm -> relu
        x = self.relu(self.bn2(self.conv2(x)))
        return x
```

### BatchNorm and model.train() / model.eval()

Like Dropout, BatchNorm behaves differently in train vs eval mode:

- `model.train()` — BatchNorm computes mean and variance from the current batch
- `model.eval()` — BatchNorm uses running statistics collected during training (more stable)

This is another reason to always switch modes correctly.

### Benefits

```
faster training     — larger learning rates become stable
acts as regularizer — mild effect similar to Dropout
more stable         — less sensitive to weight initialization
```

---

## §4 — Learning Rate Scheduling

A fixed learning rate is rarely optimal. Early in training you want a larger learning rate to make fast progress. Later you want a smaller learning rate to fine-tune without overshooting.

Learning rate scheduling automatically adjusts the learning rate during training.

### ReduceLROnPlateau

Reduces learning rate when val loss stops improving. Most practical choice — it reacts to what's actually happening in training.

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # reduce when monitored metric stops decreasing
    factor=0.5,       # multiply lr by this factor when reducing (lr = lr * 0.5)
    patience=3        # wait this many epochs before reducing
)
```

Call it after computing val loss each epoch:

```python
val_loss = ...          # compute val loss
scheduler.step(val_loss)  # scheduler checks if val loss improved
```

### StepLR

Reduces learning rate by a fixed factor every N epochs, regardless of performance.

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,   # reduce every 5 epochs
    gamma=0.5      # multiply lr by 0.5
)

# call at end of each epoch (no argument needed)
scheduler.step()
```

### CosineAnnealingLR

Learning rate follows a cosine curve from max to near zero over T_max epochs. Smooth decay.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=10   # number of epochs for one cosine cycle
)

scheduler.step()   # call at end of each epoch
```

### Which to use

```
ReduceLROnPlateau  — default choice, adapts to your training
StepLR             — simple, predictable, good when you know how many epochs to train
CosineAnnealingLR  — smooth decay, popular in research
```

---

## §5 — Early Stopping

Early stopping stops training when val loss stops improving, preventing overfitting and saving compute.

PyTorch doesn't have built-in early stopping — you implement it manually by tracking val loss:

```python
best_val_loss = float('inf')   # start with infinity so any loss is better
patience      = 5              # stop if no improvement for 5 epochs
no_improve    = 0              # counter for epochs without improvement

for epoch in range(100):
    # ... train ...
    val_loss = ...   # compute val loss

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve    = 0
        torch.save(model.state_dict(), 'best_model.pt')   # save best weights
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f'Early stopping at epoch {epoch}')
        break

# load best weights after stopping
model.load_state_dict(torch.load('best_model.pt'))
```

### torch.save and torch.load

```python
torch.save(model.state_dict(), 'best_model.pt')     # save model weights to file
model.load_state_dict(torch.load('best_model.pt'))  # load weights back into model
```

`state_dict()` returns a dictionary of all the model's learned parameters (weights and biases). Saving it lets you restore the model to its best state after early stopping triggers.

### Choosing patience

```
patience=3   aggressive — stops quickly, good when compute is limited
patience=5   balanced   — default choice
patience=10  lenient    — allows more time to escape local minima
```

---

## §6 — Weights & Biases

Weights & Biases (W&B) is a tool for tracking training experiments. Instead of printing loss each epoch and losing it when the notebook closes, W&B logs everything to a dashboard you can view in a browser.

### Setup

```bash
pip install wandb
```

```python
import wandb
wandb.login()   # opens browser to log in first time
```

### Starting a run

```python
wandb.init(
    project='mnist-cnn',   # project name — groups related runs together
    config={               # hyperparameters to log
        'learning_rate': 0.001,
        'epochs': 10,
        'batch_size': 64,
        'dropout': 0.5
    }
)
```

### Logging metrics

```python
# inside training loop, after each epoch
wandb.log({
    'train_loss': train_loss,
    'val_loss':   val_loss,
    'epoch':      epoch
})
```

### Finishing a run

```python
wandb.finish()   # call after training is done
```

### Full pattern

```python
wandb.init(project='my-project', config={'lr': 0.001, 'epochs': 10})

for epoch in range(num_epochs):
    # train...
    # evaluate...
    wandb.log({'train_loss': train_loss, 'val_loss': val_loss})

wandb.finish()
```

W&B automatically creates charts, lets you compare multiple runs side by side, and stores all your results permanently.

---

## §7 — Putting It All Together

A production-quality training loop combines all of the above:

```python
class BestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1     = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2     = nn.BatchNorm2d(64)
        self.pool    = nn.MaxPool2d(2, 2)
        self.fc1     = nn.Linear(64*5*5, 256)
        self.bn3     = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2     = nn.Linear(256, 10)
        self.relu    = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model     = BestNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

best_val_loss = float('inf')
patience      = 5
no_improve    = 0

wandb.init(project='week12', config={'lr': 0.001, 'dropout': 0.5})

for epoch in range(50):
    model.train()
    # ... training loop ...

    model.eval()
    with torch.no_grad():
        # ... val loss ...

    scheduler.step(val_loss)

    wandb.log({'train_loss': train_loss, 'val_loss': val_loss})

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve    = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

model.load_state_dict(torch.load('best_model.pt'))
wandb.finish()
```

---

## Quick Reference

```
# Dropout
nn.Dropout(p=0.5)         # place after relu, between linear layers
active in model.train()   # randomly zeros neurons
off in model.eval()       # all neurons active

# Batch Normalization
nn.BatchNorm2d(out_channels)   # after Conv2d
nn.BatchNorm1d(out_features)   # after Linear
order: conv/linear -> batchnorm -> relu

# Learning Rate Schedulers
ReduceLROnPlateau: scheduler.step(val_loss)   # call with val loss each epoch
StepLR / CosineAnnealingLR: scheduler.step()  # call with no argument each epoch

# Early Stopping
torch.save(model.state_dict(), 'best.pt')        # save when val loss improves
model.load_state_dict(torch.load('best.pt'))     # restore best weights after stopping

# W&B
wandb.init(project='name', config={...})
wandb.log({'train_loss': ..., 'val_loss': ...})
wandb.finish()
```

---

## Further Watching

- [Andrej Karpathy — Batch Normalization](https://www.youtube.com/watch?v=P4_rJfHpr7k) — intuition behind why it works
- [W&B Quickstart](https://docs.wandb.ai/quickstart) — official docs, 5 min setup

---

## Checklist

- [ ] Can explain what overfitting looks like on a loss curve
- [ ] Know what Dropout does and why it prevents overfitting
- [ ] Know where to place Dropout and BatchNorm in a model
- [ ] Understand why model.train() and model.eval() matter for Dropout and BatchNorm
- [ ] Can implement early stopping with torch.save and torch.load
- [ ] Can add a learning rate scheduler to a training loop
- [ ] Can log metrics to W&B and view them in the dashboard
