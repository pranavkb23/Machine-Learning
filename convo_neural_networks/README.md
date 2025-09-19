# Convolutional Neural Network with TensorFlow Functional API & Hyperparameter Tuning

## Overview
This project implements a **Convolutional Neural Network (CNN)** using the **TensorFlow Functional API** to classify image data.  
The primary focus is not just on model training but also on **hyperparameter tuning** and **visualization with TensorBoard** to understand how tuning affects performance.  

The objective: reach a **validation accuracy of ~95%** while experimenting with different hyperparameters.

## Repository Contents
- `convolutional_NN.py` — CNN model built using the TensorFlow Functional API.
- `aconvolutional_NN.py` — Additional experiments with hyperparameter tuning and training refinements.

## What’s Implemented

### 1. Dataset
- Image dataset (MNIST or similar, depending on configuration).
- Preprocessing: normalization, reshaping into 4D tensors for CNN input.
- Train/validation split for model evaluation.

### 2. CNN Architecture (Functional API)
- Input layer
- Multiple Conv2D + MaxPooling layers (tuned for depth/width)
- Dropout layers for regularization
- Fully connected Dense layers
- Softmax output layer for classification

### 3. Hyperparameter Tuning
Parameters explored:
- **Filter Size (Kernel Size)**
- **Number of Filters**
- **Number of Conv/Pooling Layers**
- **Learning Rate**
- **Batch Size**
- **Regularization (L2 penalty, Dropout rates)**
- **Optimizer choice** (SGD, Adam, RMSprop)
- **Learning Rate Scheduling** (step decay, exponential decay)

### 4. Training & Evaluation
- Cross-entropy loss
- Optimizer with tuned learning rates
- Training monitored with TensorBoard
- Validation metrics logged per epoch
- Final validation accuracy reported

### 5. Visualization
- **TensorBoard graphs** for:
  - Validation accuracy
  - Validation loss
- Logs exported (`events.out.tfevents`) for local TensorBoard execution
- Example screenshots (to be included in repo/wiki if desired)

## Running the Code

### 1) Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install tensorflow numpy matplotlib

2) Training the Model
python assignment8.py

3) Launching TensorBoard
tensorboard --logdir=logs/

Open browser at http://localhost:6006
 to view accuracy/loss curves.

Sample Snippets

Defining the Model (Functional API)

from tensorflow.keras import layers, models, Input

inputs = Input(shape=(28, 28, 1))
x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Dropout(0.25)(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs, outputs)


Compiling and Training

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=64)


TensorBoard Callback

from tensorflow.keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir="logs/run1")
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=64, callbacks=[tensorboard])

Key Learnings

Functional API allows for flexible, non-sequential CNN designs.

Hyperparameter tuning drastically impacts performance — especially filter sizes, dropout rates, and optimizers.

TensorBoard is invaluable for tracking training dynamics and diagnosing overfitting.

With proper tuning, CNNs can achieve >95% validation accuracy on MNIST-scale datasets.
