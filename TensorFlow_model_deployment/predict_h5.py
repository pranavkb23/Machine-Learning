# predict_h5_rest.py

import requests
import numpy as np
import pandas as pd

def load_test_data():
  # Load test data
  test_data = pd.read_csv('/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/MNIST_data_csv/mnist_test.csv')
  x_test = test_data.iloc[:, 1:].values / 255.0  # Normalize pixel values

  # Reshape the input data to 28x28
  x_test = x_test.reshape(-1, 28, 28)

  return x_test

def make_prediction():
  x_test = load_test_data()

  # Prepare data for prediction (single example)
  data = x_test[0].reshape(1, 28, 28).tolist()

  # Define the URL and request headers
  url = "http://localhost:8502/v1/models/mymodel:predict"
  headers = {"content-type": "application/json"}

  # Make the request
  response = requests.post(url, json={"instances": data}, headers=headers)

  # Parse the response
  if response.status_code == 200:
      predictions = response.json()
      print("Predictions:", predictions)
  else:
      print("Error:", response.status_code, response.text)

if __name__ == "__main__":
  make_prediction()
