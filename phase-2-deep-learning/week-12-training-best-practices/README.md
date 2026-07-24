# Week 12 — Training Best Practices + Weights & Biases

**Topic:** The difference between a model that barely trains and one that converges fast and generalizes well.
**Math Unlocked:** Learning rate schedules, batch normalization math, dropout probability

## What You'll Build
- Experiment tracking with Weights & Biases (W&B)
- Dropout regularization to prevent overfitting
- Batch normalization for stable training
- Learning rate scheduling (step decay, cosine annealing)
- Early stopping

## Concepts Covered

### Dropout
Randomly zero out neurons during training — forces the network not to rely on any single neuron. At inference, all neurons are active (scaled by dropout probability).

### Batch Normalization
Normalize activations within each mini-batch — keeps values in a good range, speeds up training, acts as mild regularization.

```
BN(x) = (x - mean) / sqrt(var + epsilon) * gamma + beta
```

### Learning Rate Scheduling
- Step decay: reduce lr by factor every N epochs
- Cosine annealing: lr follows a cosine curve from max to min
- ReduceLROnPlateau: reduce lr when validation loss stops improving

### Weights & Biases
Log metrics (loss, accuracy) to W&B dashboard. Compare runs, visualize training curves, share results.

## What's Next
Week 13: Transfer learning — use pretrained models instead of training from scratch.
