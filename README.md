# Machine Learning & Deep Learning Projects Portfolio

## Overview
This repository is a collection of projects exploring **machine learning, deep learning, and model deployment**.  
Each project builds on core concepts — from Python basics and data handling, to implementing algorithms from scratch, to modern deep learning frameworks and deployment pipelines.

The repo is structured to showcase:
- **Hands-on implementations** (from scratch and with libraries)
- **Exploratory data analysis**
- **Model training and tuning**
- **Interpretability and visualization**
- **Real-world deployment workflows**

## Repository Contents

### 📘 Project 1 — Python Basics, Data Structures & Intro to Pandas/Numpy
- Core Python practice with lists, tuples, dicts, sorting
- Mutable vs. immutable types
- Small exploratory data analysis using Pandas and NumPy
- Dataset: `bank_additional_full.csv`

### 📗 Project 2 — Linear Regression with Gradient Descent & Stochastic Gradient Descent
- Implement linear regression **from scratch** using NumPy
- Gradient Descent (batch) and Stochastic Gradient Descent (SGD)
- Comparison of convergence speed and stability
- Dataset: `assignment2_data.csv`

### 📙 Project 3 — Sonar Signal Classification (Decision Trees, Random Forests & SVC)
- Train classifiers to distinguish sonar signals from rocks vs. metal cylinders
- Models: Decision Tree, Random Forest, SVC
- Hyperparameter tuning & performance comparison
- Dataset: `sonar.all-data`

### 📕 Project 4 — Credit Card Default Prediction & Model Interpretability
- Predict default risk using tree-based classifiers: Decision Tree, Random Forest, XGBoost
- Interpretability with **LIME, Partial Dependence Plots, Surrogate Models**
- Emphasis on explainability for financial applications
- Dataset: `default_of_credit_card_clients.csv`

### 📒 Project 5 — Handwritten Digit Recognition (Two-Layer Neural Network from Scratch)
- Build a feedforward neural network using **only Python & NumPy**
- Forward pass, backpropagation, gradient descent coded manually
- Trained on **MNIST** dataset
- Accuracy ~85–92% depending on hyperparameters

### 📓 Project 6 — Neural Networks with TensorFlow, PyTorch & Custom Implementation
- Compare three approaches to MNIST digit classification:
  1. Custom NumPy implementation
  2. TensorFlow implementation
  3. PyTorch implementation
- Highlights the trade-off between **manual control** and **framework abstraction**

### 📔 Project 7 — TensorFlow Model Deployment (TF Serving & Flask)
- Train and export a TensorFlow model
- Deploy with **TensorFlow Serving (Docker)** as a REST API
- Flask server for local predictions
- Demonstrates **end-to-end deployment pipeline**

### 📒 Project 8 — CNN with TensorFlow Functional API & Hyperparameter Tuning
- Build a CNN using the **Functional API**
- Hyperparameter tuning on filters, layers, learning rate, dropout, optimizer
- Validation accuracy ~95% achievable
- Training monitored with **TensorBoard** (loss/accuracy graphs)

---

## Skills Demonstrated
- **Programming & Data Handling**: Python, NumPy, Pandas
- **Machine Learning Algorithms**: Linear Regression, Decision Trees, Random Forests, SVM
- **Deep Learning**: Neural Networks (scratch + frameworks), CNNs, TensorFlow, PyTorch
- **Model Interpretability**: LIME, PD Plots, Feature Importance
- **Deployment**: TensorFlow Serving, Flask, REST APIs, Docker
- **Experiment Tracking**: TensorBoard
