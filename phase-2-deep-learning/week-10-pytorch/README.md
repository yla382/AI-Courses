# Week 10 — PyTorch Fundamentals

**Topic:** Move from sklearn's black box to PyTorch — full control over architecture, training loop, and data.
**Math Unlocked:** Tensor operations, automatic differentiation (autograd)

## What You'll Build
- Tensors: creation, operations, GPU/CPU transfer
- Autograd: let PyTorch compute gradients automatically
- A neural network in PyTorch from scratch (same network as week 9, now in PyTorch)
- Custom training loop: forward pass, loss, backward, optimizer step

## Concepts Covered
- `torch.Tensor` vs numpy array
- `requires_grad=True` and autograd
- `nn.Module` — base class for all PyTorch models
- `nn.Linear`, `nn.ReLU`, `nn.Sequential`
- Optimizers: `optim.SGD`, `optim.Adam`
- DataLoader and Dataset for batching
- `model.train()` vs `model.eval()`

## Why PyTorch?
- sklearn MLP is a black box — you can't customize the architecture
- PyTorch gives you full control: custom layers, custom losses, custom training loops
- Industry standard for research and production deep learning
- Everything in weeks 11-16 requires PyTorch

## What's Next
Week 11: Convolutional Neural Networks — PyTorch applied to image data.
