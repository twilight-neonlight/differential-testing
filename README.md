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

## Results

See `results/` for visualizations of disagreement-inducing inputs and model predictions.
