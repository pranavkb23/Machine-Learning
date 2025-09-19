# predict_saved_model_keras.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer

def load_test_data():
  # Load test data
  test_data = pd.read_csv('/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/MNIST_data_csv/mnist_test.csv')
  x_test = test_data.iloc[:, 1:].values / 255.0  # Normalize pixel values

  # Reshape the input data to 28x28
  x_test = x_test.reshape(-1, 28, 28)

  return x_test

def make_prediction():
  # Load the model saved in SavedModel format
  model_keras = TFSMLayer('/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment7/saved_model_keras/1', call_endpoint='serving_default')
  model_h5 = TFSMLayer('/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment7/saved_model_h5/1', call_endpoint='serving_default')

  x_test = load_test_data()

  # Prepare data for prediction (single example)
  data = np.expand_dims(x_test[0], axis=0)

  # Make the prediction
  predictions_keras = model_keras(data)
  print("Prediction Keras:", predictions_keras)

  # Make the prediction
  predictions_h5 = model_h5(data)
  print("Prediction h5:", predictions_h5)

if __name__ == "__main__":
  make_prediction()
