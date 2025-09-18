# Handwritten Digit Recognition — Two-Layer Neural Network from Scratch

## Overview
This project implements a **two-layer feedforward neural network** from scratch in Python to classify handwritten digits from the **MNIST dataset**.  
The goal is to understand the **inner workings of deep learning models** by manually coding forward propagation, loss calculation, and backpropagation, rather than relying on frameworks like TensorFlow or PyTorch.

## Repository Contents
- `assignment5.py` — Full implementation of the neural network.
- `Assignment-5.docx` — Problem statement (not required for execution).

## What’s Implemented

### 1. Dataset
- **MNIST**: 70,000 grayscale images of handwritten digits (0–9).
  - Training set: 60,000 images
  - Test set: 10,000 images
- Each image: `28 × 28` pixels, flattened into a 784-dimensional input vector.

### 2. Neural Network Architecture
- **Input Layer**: 784 units (one for each pixel)
- **Hidden Layer**: User-defined (commonly 64–128 neurons, ReLU or sigmoid activation)
- **Output Layer**: 10 units (digit classes 0–9, softmax activation)

### 3. Forward & Backpropagation
- **Forward pass**:
  - Compute hidden activations
  - Compute output predictions via softmax
- **Loss Function**:
  - Cross-entropy loss
- **Backward pass**:
  - Derivatives for weights and biases computed manually
  - Parameters updated via gradient descent

### 4. Training & Evaluation
- Mini-batch gradient descent
- Hyperparameters: learning rate, batch size, number of epochs
- Metrics: training loss, test accuracy
- Final accuracy on MNIST test set typically **85–92%** (depending on hyperparameters)

## Running the Code

### 1) Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install numpy matplotlib

2) Running the Network
python assignment5.py


This will:

Load the MNIST dataset

Train the neural network

Print training progress

Report test accuracy

Sample Snippets

Initializing Parameters

def initialize_parameters(input_dim, hidden_dim, output_dim):
    W1 = np.random.randn(hidden_dim, input_dim) * 0.01
    b1 = np.zeros((hidden_dim, 1))
    W2 = np.random.randn(output_dim, hidden_dim) * 0.01
    b2 = np.zeros((output_dim, 1))
    return W1, b1, W2, b2


Forward Propagation

def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return A1, A2


Backpropagation & Updates

def backward(X, Y, A1, A2, W2):
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

Key Learnings

Coding a neural network from scratch clarifies:

How inputs flow through hidden layers

How gradients propagate backward

Why learning rates and initialization matter

This implementation serves as a foundation for understanding frameworks like TensorFlow/PyTorch.
