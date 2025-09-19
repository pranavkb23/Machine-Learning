# Neural Networks with TensorFlow, PyTorch & Custom Implementation

## Overview
This project explores training neural networks on the **MNIST dataset** using three different approaches:
1. A **from-scratch Python implementation** (manual backpropagation).
2. **TensorFlow** framework implementation.
3. **PyTorch** framework implementation.

By comparing these, the project highlights the trade-offs between **control and abstraction**, helping understand what deep learning frameworks do under the hood.

## Repository Contents
- `neural_networks.py` — Pure Python/NumPy implementation (manual forward/backprop).
- `NN_tensorflow.ipynb` — Neural network training in TensorFlow.
- `NN_PyTorch.ipynb` — Neural network training in PyTorch.

## What’s Implemented

### 1. Dataset
- **MNIST Handwritten Digits**
  - 70,000 grayscale images (28×28 pixels).
  - 60,000 training, 10,000 test images.
  - 10 classes (digits 0–9).

### 2. Models

#### Custom Python Implementation (`assignment6.py`)
- Two-layer feedforward NN
- Forward propagation, softmax activation
- Cross-entropy loss
- Manual backpropagation and parameter updates with gradient descent

#### TensorFlow Implementation (`NN_tensorflow.ipynb`)
- Dense layers with ReLU activation
- Softmax output layer
- Adam optimizer
- TensorBoard logs and training curves

#### PyTorch Implementation (`NN_PyTorch.ipynb`)
- `torch.nn.Sequential` model
- ReLU activations, softmax output
- CrossEntropyLoss and Adam optimizer
- GPU acceleration (if CUDA available)

### 3. Evaluation
- Training & test accuracy
- Confusion matrix
- Loss vs. epoch plots
- Comparison across the three approaches

## Running the Code

### 1) Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install numpy matplotlib
pip install tensorflow torch torchvision


2) Running Custom Python Implementation
python assignment6.py

3) Running TensorFlow Notebook

Open in Jupyter or VSCode:

jupyter notebook NN_tensorflow.ipynb

4) Running PyTorch Notebook
jupyter notebook NN_PyTorch.ipynb

Sample Snippets

TensorFlow

import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


PyTorch

import torch.nn as nn
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)


Custom Implementation

# Forward pass
Z1 = np.dot(W1, X) + b1
A1 = np.tanh(Z1)
Z2 = np.dot(W2, A1) + b2
A2 = softmax(Z2)

Key Learnings

The manual NN teaches the mechanics of forward/backpropagation.

TensorFlow abstracts away much of the math but offers strong tooling (TensorBoard, Keras API).

PyTorch provides flexibility, dynamic computation graphs, and strong research adoption.

Running all three provides a holistic view of deep learning model development.
