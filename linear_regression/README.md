# Linear Regression with Gradient Descent & Stochastic Gradient Descent

## Overview
This project demonstrates how to implement **Linear Regression** from scratch using **NumPy**.  
Both **Batch Gradient Descent (GD)** and **Stochastic Gradient Descent (SGD)** approaches are covered.  
The goal is to build intuition for optimization algorithms by avoiding pre-built ML libraries (e.g., scikit-learn).

## Repository Contents
- `linear_regression.py` — Core implementation of Linear Regression using **batch gradient descent**.
- `linear_regression2.py` — Extended version implementing **stochastic gradient descent**.
- `assignment2_data.csv` — Dataset used for training and evaluation.

## What’s Implemented

### 1. Data Handling
- Load tabular data from CSV using NumPy.
- Split into **features (X)** and **target (y)** variables.
- Apply normalization to stabilize gradient updates.

### 2. Linear Regression via Gradient Descent
- Initialize weights and bias.
- Iteratively update parameters using the gradient of the loss function (Mean Squared Error).
- Track convergence by observing the cost function decreasing across epochs.

### 3. Linear Regression via Stochastic Gradient Descent
- Randomly shuffle and pick one sample at a time (or mini-batches).
- Update parameters more frequently, leading to faster (though noisier) convergence.
- Compare performance against batch GD.

### 4. Evaluation & Visualization
- Plot the cost function over iterations to visualize convergence.
- Compare predicted vs. actual target values.
- Explore how learning rate and iterations affect accuracy.

## Running the Code

### 1) Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

2) Running Batch Gradient Descent
python assignment_2.py

3) Running Stochastic Gradient Descent
python assignment_2_2nd.py

Sample Usage (Interactive)
import numpy as np
from assignment_2 import LinearRegressionGD
from assignment_2_2nd import LinearRegressionSGD

# Load dataset
data = np.loadtxt("assignment2_data.csv", delimiter=",", skiprows=1)
X, y = data[:, :-1], data[:, -1]

# Batch GD
model_gd = LinearRegressionGD(lr=0.01, epochs=1000)
model_gd.fit(X, y)
print("Batch GD weights:", model_gd.weights)

# SGD
model_sgd = LinearRegressionSGD(lr=0.01, epochs=1000)
model_sgd.fit(X, y)
print("SGD weights:", model_sgd.weights)

Key Learnings

Gradient Descent provides stable convergence but can be slow for large datasets.

Stochastic Gradient Descent updates faster and is better suited for big data, though it introduces variance in updates.

Parameter tuning (learning rate, iterations) is crucial to model performance.

pip install numpy matplotlib
