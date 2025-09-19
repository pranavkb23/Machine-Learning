# predict_request.py

import requests
import pandas as pd

def load_test_data():
  # Load test data
  test_data = pd.read_csv('/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/MNIST_data_csv/mnist_test.csv')
  x_test = test_data.iloc[:, 1:].values / 255.0  # Normalize pixel values

  return x_test

def make_prediction():
  x_test = load_test_data()

  # Prepare data for prediction (single example)
  data = x_test[0].reshape(1, 28, 28).tolist()  # Reshape to match model input

  # Define the URL and request headers
  url = "http://localhost:8501/v1/models/mymodel:predict"
  headers = {"content-type": "application/json"}

  # Make the request
  response = requests.post(url, json={"instances": data}, headers=headers)

  # Parse the response
  predictions = response.json()
  print("Predictions:", predictions)

if __name__ == "__main__":
  make_prediction()
