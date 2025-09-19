# TensorFlow Model Deployment — TF Serving & Flask REST API

## Overview
This project demonstrates how to **deploy a trained TensorFlow model** using **TensorFlow Serving** and expose it as a **REST API**.  
The focus is on **deployment workflow** rather than model performance. The example model is trained on the **MNIST dataset** for digit recognition.

## Repository Contents
- `train_export.py` — Trains the model and exports it in TensorFlow `SavedModel` format.
- `model.keras` — Saved Keras model (lightweight format, recommended for version control).
- `model.h5` — Legacy HDF5 model (optional, included for reference).
- `flask_API.py` — Flask server for serving predictions locally.
- `predict.py` — Base script to send requests to the deployed model.
- `predict_h5.py` / `predict_keras.py` — Demonstrates loading predictions from saved `.h5` or `.keras` models.
- `pred_local.py` / `pred_local_saved.py` — Local prediction scripts (without TF Serving).

## What’s Implemented

### 1. Model Training & Export
- Dataset: MNIST handwritten digits
- Simple feedforward neural network (Dense layers with ReLU + softmax output)
- Exported with `SavedModel` format for TensorFlow Serving compatibility

### 2. Serving with TensorFlow Serving (Docker)
- Model served using official TensorFlow Serving Docker image
- REST API exposed at port `8501`
- Example Docker run command:

```bash
docker run -p 8501:8501 \
  --name=tf_serving_mnist \
  --mount type=bind,source=$(pwd)/saved_model/my_model,target=/models/mnist_model \
  -e MODEL_NAME=mnist_model -t tensorflow/serving

3. REST API Endpoints

Predict Endpoint

POST http://localhost:8501/v1/models/mnist_model:predict


Input: JSON with image data (flattened 28×28 pixels, normalized).

Output: Predicted class probabilities.

4. Local Prediction with Flask

A lightweight Flask API (flask_API.py) mimics serving setup for local testing.

Run:

python flask_API.py


Flask API will be available at http://127.0.0.1:5000/predict.

5. Prediction Scripts

predict.py — Sends JSON requests to TF Serving REST API.

pred_local.py / pred_local_saved.py — Predicts directly from locally loaded models.

predict_h5.py / predict_keras.py — Supports predictions from .h5 and .keras formats.

Running the Code
1) Environment Setup
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install tensorflow flask requests numpy

2) Training and Export
python train_export.py

3) Deploy with TensorFlow Serving (Docker)
docker run -p 8501:8501 \
  --name=tf_serving_mnist \
  --mount type=bind,source=$(pwd)/saved_model/my_model,target=/models/mnist_model \
  -e MODEL_NAME=mnist_model -t tensorflow/serving

4) Making Predictions
python predict.py


Example JSON payload:

{
  "instances": [[0, 0, 0, ..., 0, 0, 0]]
}


Response:

{
  "predictions": [[0.01, 0.00, 0.95, 0.03, ...]]
}

Key Learnings

TensorFlow Serving is a production-ready tool for exposing trained models as REST APIs.

Flask provides a simple way to test serving locally.

SavedModel format is the preferred way to deploy TensorFlow models.

REST APIs make it easy to integrate ML models into applications.
