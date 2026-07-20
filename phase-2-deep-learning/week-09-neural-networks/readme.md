# Week 9 — Neural Networks

**Topic:** The bridge from phase 1 to deep learning. Everything you know from phase 1 still applies — you're just stacking more layers.
**Math Unlocked:** Backpropagation, vanishing gradients, activation functions

## What You'll Build
- Forward pass from scratch in numpy
- Backpropagation from scratch — watch loss decrease
- MLPClassifier in sklearn — quick experiments
- Compare neural network vs classical ML on tabular data

## Concepts Covered
- From logistic regression to neural network: same math, more layers
- Activation functions: sigmoid, ReLU, leaky ReLU, softmax
- Forward pass: data flows left to right through layers
- Backpropagation: chain rule applied backwards to compute gradients
- Vanishing gradients: why sigmoid fails in deep networks
- Hidden layer size and depth: how architecture affects capacity
- Batch size and mini-batch gradient descent

## Connections to Phase 1
```
Week 4 gradient descent  ->  same update rule, now applied to millions of weights
Week 4 chain rule        ->  backpropagation IS the chain rule
Week 5 log loss          ->  same loss function used in output layer
Week 6 bias-variance     ->  same tradeoff, controlled by architecture and regularization
```

## Key Insight
Neural networks don't require new math — they require the same math applied many times in sequence. The complexity comes from scale, not from new concepts.

## What's Next
Week 10: PyTorch — full control over architecture and training loop. Required for everything in weeks 11-16.
