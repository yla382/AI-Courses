# Week 11 Study Guide — Convolutional Neural Networks

Work through each section, then immediately do that section in `tutorial.ipynb` before moving on.

---

## The Big Shift

In weeks 9-10 you used fully connected networks (MLPs) — every neuron connects to every input. That works fine for tabular data like wine or breast cancer where the features don't have spatial relationships.

For images it breaks down. This week you'll learn why, and how CNNs fix it.

---

## §1 — Why MLPs Fail on Images

### The parameter explosion problem

A small grayscale image is 28×28 = 784 pixels. If your first hidden layer has 512 neurons, the weight matrix is:

```
784 × 512 = 401,408 parameters — just for one layer
```

A real image is 224×224×3 (color) = 150,528 pixels. First layer with 512 neurons:

```
150,528 × 512 = 77,070,336 parameters — just for one layer
```

That's 77 million parameters before you've even done anything useful. This overfits immediately on any reasonable dataset.

### The spatial awareness problem

MLPs treat every pixel as an independent feature — they have no built-in concept of which pixels are neighbors. Pixel 0 and pixel 783 (opposite corners) are treated as equally related as pixel 0 and pixel 1 (right next to each other).

This matters because edges, shapes, and textures are defined by neighboring pixels. An MLP can only learn "pixel (3,4) tends to have a certain value" — it can't efficiently learn "there's a horizontal edge here" because that requires knowing that pixels (3,3), (3,4), and (3,5) are spatially adjacent.

### What CNNs do differently

Instead of connecting every neuron to every pixel, CNNs use small **filters** (e.g. 3×3) that slide across the image. Each filter looks at a small local patch at a time and detects a specific pattern (an edge, a corner, a texture). This gives two advantages:

```
parameter sharing:   one filter is reused across the entire image → far fewer parameters
spatial awareness:   filter looks at neighboring pixels together → can detect local patterns
```

### Counting parameters

**Linear layer** — each neuron connects to every input, plus one bias:
```
nn.Linear(in, out)  →  (in × out) + out  parameters
                                     ↑ one bias per output neuron

example: nn.Linear(784, 512)  →  (784 × 512) + 512  =  401,920
```

**Conv2d layer** — each filter has `kernel_size × kernel_size × in_channels` weights, plus one bias per filter. The filter is reused across the entire image, so the spatial size of the image doesn't affect parameter count at all:
```
nn.Conv2d(in_channels, out_channels, kernel_size)
  →  (kernel_size × kernel_size × in_channels + 1) × out_channels
                                                  ↑ +1 for bias

example: nn.Conv2d(1, 32, kernel_size=3)
  →  (3 × 3 × 1 + 1) × 32  =  10 × 32  =  320
```

Compare: 401,920 vs 320. The Conv layer has 1,250× fewer parameters than the Linear layer, and it works on the full 28×28 image.

---

## §2 — The Convolution Operation

A convolution is a small filter (also called a kernel) sliding across an image, computing a dot product at each position.

**Filter, kernel, and weights all mean the same thing** — a small grid of numbers that the network learns during training. In week 9/10 you called them weights because they connected neurons in a matrix. In CNNs the same concept is called a filter or kernel because it slides across the image like a window. The name changes, the idea doesn't.

### Example: 5×5 image, 3×3 filter

```
image (5×5):          filter (3×3):
1 2 3 0 1             1  0 -1
0 1 2 1 0             1  0 -1
1 0 1 0 1             1  0 -1
0 1 0 1 0
1 0 1 0 1
```

At position (0,0), the filter covers the top-left 3×3 patch of the image:
```
patch:        filter:       multiply + sum:
1 2 3         1  0 -1       1*1 + 2*0 + 3*(-1)
0 1 2    ×    1  0 -1   =   0*1 + 1*0 + 2*(-1)  = 1+0-3+0+0-2+1+0-1 = -4
1 0 1         1  0 -1       1*1 + 0*0 + 1*(-1)
```

The filter slides one step right, computes again. Then moves down, and so on across the whole image.

The result is a **feature map** — a 2D grid of values showing where the filter detected its pattern strongly (high value) or weakly (low value).

### Output size formula

```
output_size = (input_size - kernel_size + 2×padding) / stride + 1
```

Example: 5×5 image, 3×3 filter, stride=1, padding=0 (default):
```
(5 - 3 + 0) / 1 + 1 = 3   →   output is 3×3
```

With `padding=1`: the `+2` cancels the `-2` from the kernel, so output size equals input size:
```
(5 - 3 + 2) / 1 + 1 = 5   →   output is 5×5 (same as input)
```

### What filters learn

The filter weights are **learned during training** — you don't design them. Early layers tend to learn:
```
Layer 1: edges (horizontal, vertical, diagonal)
Layer 2: corners, curves, textures
Layer 3+: higher-level patterns (eyes, wheels, etc.)
```

The filter in the example above (`1 0 -1` repeated) is a vertical edge detector — it produces high values where there's a sharp left-to-right intensity change.

---

## §3 — Multiple Filters and Feature Maps

A single filter produces one feature map. In practice you use many filters at once — each learns to detect a different pattern.

```python
nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
```

This applies 32 different 3×3 filters to the input, producing 32 feature maps. The output has shape:

```
(batch_size, 32, height, width)
```

### Channels

**in_channels**: how many channels the input has
- Grayscale image: 1 channel
- Color (RGB) image: 3 channels
- Output of previous conv layer: whatever `out_channels` that layer used

**out_channels**: how many filters to apply = how many feature maps to produce

```python
# grayscale image input
nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=3)  # 1 → 32 feature maps
nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)  # 32 → 64 feature maps
```

---

## §4 — Pooling

After a convolution, the feature maps are still large. Pooling **downsamples** them — reduces spatial size while keeping the most important information.

### MaxPool2d

Takes the maximum value in each non-overlapping window:

```
input (4×4):          2×2 blocks:        max of each block:
1 3 2 4               [1 3] [2 4]        6 4
5 6 1 2    →          [5 6] [1 2]   →    3 4
3 2 4 1               [3 2] [4 1]
2 1 3 2               [2 1] [3 2]
```

Each 2×2 block → one value (the max). Output is 2×2.

```python
nn.MaxPool2d(kernel_size=2, stride=2)   # halves both height and width
```

**Why max?** The filter detected a pattern at some position in the patch. We care that the pattern exists, not exactly where. Taking the max keeps the strongest detection.

---

## §5 — Full CNN Architecture

A typical CNN stacks: Conv → ReLU → Pool, repeated, then Flatten → Linear for classification.

```
Input image (1, 28, 28)
    ↓
Conv2d(1, 32, 3)    →  (32, 26, 26)   # 32 filters, each 3×3
ReLU
MaxPool2d(2, 2)     →  (32, 13, 13)   # halved
    ↓
Conv2d(32, 64, 3)   →  (64, 11, 11)   # 64 filters
ReLU
MaxPool2d(2, 2)     →  (64, 5, 5)     # halved
    ↓
Flatten             →  (64×5×5,) = (1600,)
Linear(1600, 128)   →  (128,)
ReLU
Linear(128, 10)     →  (10,)           # 10 classes for MNIST digits
```

The Conv layers extract spatial features. The Linear layers at the end classify based on those features.

---

## §6 — PyTorch Building Blocks

### nn.Conv2d

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

- `in_channels`: channels of input (1 for grayscale, 3 for RGB, or previous layer's out_channels)
- `out_channels`: number of filters = number of output feature maps
- `kernel_size`: filter size (3 means 3×3)
- `padding`: adds zeros around the border so output size matches input size (`padding=1` with `kernel_size=3` keeps size the same)

### nn.MaxPool2d

```python
nn.MaxPool2d(kernel_size=2, stride=2)   # most common — halves spatial dimensions
```

### nn.Flatten

Converts the 3D feature maps `(channels, height, width)` into a 1D vector so it can feed into a Linear layer:

```python
nn.Flatten()   # (batch, 64, 5, 5) → (batch, 1600)
```

You must know the size after flattening to set the first Linear layer's input size. Calculate it by tracking shapes through each layer (see §7).

---

## §7 — Shape Tracking

Shape tracking is the most important skill for debugging CNNs. You need to know the output shape after every layer to set the next layer's input size correctly.

### Rules

```
Conv2d(in_c, out_c, k):            (batch, in_c, H, W)  →  (batch, out_c, H-k+1,   W-k+1)
Conv2d(in_c, out_c, k, padding=1): (batch, in_c, H, W)  →  (batch, out_c, H,       W)      # same size
MaxPool2d(2, 2):          (batch, C,     H,   W)  →  (batch, C,    H//2,  W//2)
Flatten:                  (batch, C,     H,   W)  →  (batch, C*H*W)
Linear(in, out):          (batch, in)             →  (batch, out)
```

### How to check shapes in code

```python
# pass a dummy input through the model and print shapes at each step
x = torch.randn(1, 1, 28, 28)   # batch=1, channels=1, 28×28
print(x.shape)                   # torch.Size([1, 1, 28, 28])

x = conv1(x)
print(x.shape)                   # torch.Size([1, 32, 26, 26])

x = pool(x)
print(x.shape)                   # torch.Size([1, 32, 13, 13])
```

Always do this when building a new CNN — it prevents the most common error (wrong input size to Linear after Flatten).

---

## §8 — Training a CNN

The training loop is **identical to week 10** — nothing changes. CNNs are just a different model architecture; the train/eval pattern, DataLoader, loss functions, and optimizer are the same.

```python
# same as week 10
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)
```

### MNIST specifics

MNIST is a dataset of 28×28 grayscale handwritten digit images, 10 classes (0-9).

```python
# loading MNIST with torchvision
from torchvision import datasets, transforms

transform = transforms.ToTensor()   # converts PIL image to tensor, scales to [0,1]

train_data = datasets.MNIST(root='data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)
```

Input shape from DataLoader: `(batch_size, 1, 28, 28)` — batch of grayscale 28×28 images.

Loss function: `nn.CrossEntropyLoss()` — 10 classes, same as multiclass from week 10.
Labels: `dtype=torch.long` — class indices 0-9, no unsqueeze needed.

---

## Quick Reference

```
# dtype rules
X:  dtype=torch.float32   (image tensors from ToTensor() are already float32)
y:  dtype=torch.long      (class indices for CrossEntropyLoss)

# loss + output layer
10 classes → CrossEntropyLoss → no softmax on output layer → argmax for prediction

# shape after each layer (MNIST example)
input:          (batch, 1,  28, 28)
Conv2d(1,32,3): (batch, 32, 26, 26)
MaxPool2d(2,2): (batch, 32, 13, 13)
Conv2d(32,64,3):(batch, 64, 11, 11)
MaxPool2d(2,2): (batch, 64,  5,  5)
Flatten:        (batch, 1600)
Linear(1600,10):(batch, 10)
```

---

## Further Watching

- [3Blue1Brown — But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) (23 min) — best visual explanation of what convolution actually computes
- [Andrej Karpathy — Building makemore (CNNs)](https://www.youtube.com/watch?v=P6sfmUTpUmc) — builds a CNN from scratch

---

## Checklist

- [ ] Can explain why MLPs fail on images (parameter count, no spatial awareness)
- [ ] Can describe what a convolution does — filter sliding over image, dot product at each position
- [ ] Know what in_channels, out_channels, and kernel_size mean in nn.Conv2d
- [ ] Know what MaxPool2d does and why
- [ ] Can track shapes through a CNN: Conv → Pool → Flatten → Linear
- [ ] Can load MNIST and train a CNN using the week 10 training loop pattern
