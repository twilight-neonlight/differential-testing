# Differential Testing with DeepXplore (CIFAR-10)

Assignment #2 — Reliable and Trustworthy AI

## Overview

This project implements differential testing of neural networks following the DeepXplore methodology (Pei et al., SOSP 2017). Two ResNet50 models trained on CIFAR-10 with different seeds and augmentation strategies are tested to find inputs on which the models disagree.

## Note on Implementation

The original DeepXplore repository (https://github.com/peikexin9/deepxplore) is implemented in Keras/TensorFlow. Since this project uses PyTorch throughout (model definition, training, inference), directly integrating the original code would require maintaining two separate deep learning frameworks, adding significant complexity and dependency overhead.

Instead, the core DeepXplore algorithm — neuron coverage measurement and gradient-based test input generation — has been **reimplemented in PyTorch** under `deepxplore/`, faithfully following the original paper's methodology. The algorithm logic (joint loss combining disagreement objective and neuron coverage, gradient ascent on the input) is equivalent to the original.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the two models

```bash
python train.py
```

This produces `model_a.pth` (no augmentation, seed=0) and `model_b.pth` (with augmentation, seed=42).

### 2. Run DeepXplore

```bash
python test.py
```

Disagreement-inducing inputs are saved as PNG files under `results/`.

## Project Structure

```
.
├── model.py          # ResNet50 architecture for CIFAR-10
├── train.py          # Trains model_a and model_b
├── test.py           # Runs DeepXplore and saves results
├── deepxplore/
│   ├── coverage.py   # Neuron coverage tracker
│   └── generator.py  # Gradient-based test input generator
├── results/          # Output visualizations
└── requirements.txt
```

## Code Explanation

### `model.py`

Defines a ResNet50 adapted for CIFAR-10's 32×32 input size. Two modifications are applied to the standard torchvision ResNet50:

- `conv1` is replaced with a 3×3 convolution (stride 1, no bias) instead of the original 7×7 (stride 2), to avoid excessive downsampling on the small CIFAR-10 images.
- `maxpool` is replaced with `nn.Identity()` for the same reason.

The final fully-connected layer is resized to output 10 classes.

### `train.py`

Trains two independent ResNet50 models on CIFAR-10 to use as the DNN pair for differential testing. The two models are intentionally differentiated to make disagreement-based testing meaningful:

- **Model A** (`model_a.pth`): trained with seed=0, no data augmentation.
- **Model B** (`model_b.pth`): trained with seed=42, RandomCrop and RandomHorizontalFlip augmentation.

Both models are trained with Adam (lr=1e-3) and CrossEntropyLoss for 10 epochs.

### `deepxplore/coverage.py`

Implements `NeuronCoverageTracker`, which measures the fraction of neurons activated above a threshold across all forward passes. It registers PyTorch forward hooks on every `nn.ReLU` module in the model. For convolutional layers, spatial dimensions are averaged before comparing against the threshold.

**Modification from naive implementation:** ResNet50's `Bottleneck` block reuses a single `self.relu` instance for three activations of different channel sizes (e.g., 64, 64, and 256). A naive implementation keyed only by module name would set `total` on the first call (64 neurons) but accumulate indices up to 256, yielding coverage values above 100%. This is fixed by keying each tracked group by `f"{name}_{neuron_count}"`, separating activations of different sizes into distinct entries.

### `deepxplore/generator.py`

Implements `generate_test_inputs`, which reproduces Algorithm 1 from the DeepXplore paper. Starting from seed inputs taken from the CIFAR-10 test set, it performs gradient ascent to jointly maximize two objectives:

- **L_diff**: encourages model disagreement — increases model 0's confidence on its predicted class while decreasing other models' confidence on the same class.
- **L_cov**: encourages neuron coverage — sums activations of neurons currently below the threshold, pushing them to activate.

The combined loss is `L = (1 - λ) * L_diff + λ * L_cov`. After `steps` iterations of signed gradient ascent, inputs on which the models produce different predicted classes are recorded as disagreements.

### `test.py`

Orchestrates the full pipeline:

1. Loads `model_a.pth` and `model_b.pth`.
2. Samples 100 seed inputs from the CIFAR-10 test set.
3. Attaches a `NeuronCoverageTracker` to model A; coverage accumulates automatically during generation.
4. Calls `generate_test_inputs` to produce disagreement-inducing inputs.
5. Prints the number of disagreements and neuron coverage.
6. Saves individual PNG files (`results/disagreement_01.png` … ) and a combined grid (`results/disagreements.png`), each showing the generated input and the predicted class from each model.

## Results

Evaluated on 100 CIFAR-10 test images as seeds (50 gradient ascent steps per input):

| Metric | Value |
|---|---|
| Seed inputs tested | 100 |
| Disagreement-inducing inputs | 87 / 100 |
| Neuron coverage (Model A) | 98.66% |

Visualizations of disagreement-inducing inputs are saved in `results/`.
