# predict_keras.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def load_test_data():
  # Load test data
  test_data = pd.read_csv('/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/MNIST_data_csv/mnist_test.csv')
  x_test = test_data.iloc[:, 1:].values / 255.0  # Normalize pixel values

  # Reshape the input data to 28x28
  x_test = x_test.reshape(-1, 28, 28)

  # instead of loading test data, take user input (MNIST image). 
  # how to take image input in flask (how to make flask app).
  # POSTMAN to test the API
  # make the whole API

  # to build an API of a tensorflow trained model using Flask in Python
  # 

  return x_test

def make_prediction():
  # Load the model saved in .keras format
  model_keras = load_model('/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment7/model.keras')

  # Load the model saved in .h5 format
  model_h5 = load_model('/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment7/model.h5')

# Recompile the model with its original settings
  model_keras.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  
  # Recompile the model with its original settings
  model_h5.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  x_test = load_test_data()

  # Prepare data for prediction (single example)
  data = np.expand_dims(x_test[0], axis=0)

  # Make the prediction
  predictions_keras = model_keras.predict(data)
  print("Prediction Keras:", predictions_keras)

  # Make the prediction
  predictions_h5 = model_h5.predict(data)
  print("Prediction h5:", predictions_h5)

# deploy, file should be able to run continuously, not stop


if __name__ == "__main__":
  make_prediction()