# Week 11 — Convolutional Neural Networks (CNNs)

**Topic:** CNNs are the architecture behind image recognition. Learn why they work and build one in PyTorch.
**Math Unlocked:** Convolution operation, receptive field, pooling

## What You'll Build
- A CNN from scratch in PyTorch on MNIST (handwritten digits)
- Conv2d + MaxPool2d + Linear layers
- Visualize what filters learn (edges, shapes)
- Compare CNN vs MLP on image data

## Concepts Covered
- Why fully connected networks fail on images (too many parameters, no spatial awareness)
- Convolution: sliding filter over image, detecting local patterns
- Filter / kernel: learned weights that detect specific features
- Feature maps: output of a convolution layer
- Pooling: downsampling to reduce spatial dimensions
- Receptive field: how much of the input each neuron sees
- Flatten: connecting conv layers to fully connected layers

## Math: Convolution
```
output[i,j] = sum(input[i+di, j+dj] * filter[di, dj])
```
Filter slides over the image, computing dot products at each position.

Output size: (input_size - filter_size + 1) / stride

## Architecture Pattern
```
Input image
    -> Conv2d (learn local features)
    -> ReLU
    -> MaxPool2d (downsample)
    -> Conv2d (learn higher-level features)
    -> ReLU
    -> MaxPool2d
    -> Flatten
    -> Linear (classify)
    -> Softmax
```

## What's Next
Week 12: Training best practices — batch norm, dropout, learning rate scheduling.
