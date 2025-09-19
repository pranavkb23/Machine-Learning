# train_and_export.py

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os

def load_data():
  # Load training data
  train_path = "/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/MNIST_data_csv/mnist_train.csv"
  train_data = pd.read_csv(train_path)
  x_train = train_data.iloc[:, 1:].values / 255.0  # Normalize pixel values
  y_train = train_data.iloc[:, 0].values           # Labels are in the first column

  x_train = x_train.reshape(-1, 28, 28)            # Reshaping the input data to 28x28

  return x_train, y_train

def train_and_save_model():
  x_train, y_train = load_data()

  # Define the model
  model = Sequential([
    Flatten(input_shape=(28, 28)),  # Images are 28x28 pixels
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
  ])

  # Compile the model
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  # Train the model
  model.fit(x_train, y_train, epochs=5)

  print(model.summary())

  # Determine the directory of the current script
  script_dir = os.path.dirname(os.path.abspath(__file__))

  # # Save the model in .keras format
  # keras_path = os.path.join(script_dir, "model.keras")
  # model.save(keras_path)

  # Save the model in .h5 format
  h5_path = os.path.join(script_dir, "model.h5")
  model.save(h5_path)

  # # Load the models
  # keras_model = load_model('/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment7/model.keras')
  # h5_model = load_model('/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment7/model.h5')

  # print(keras_model.summary())
  # print(h5_model.summary())

  # # Export to SavedModel format for keras
  # saved_model_keras_path = "/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment7/saved_model_keras"
  # tf.saved_model.save(keras_model, saved_model_keras_path)

  # # Export to SavedModel format for h5
  # saved_model_h5_path = "/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment7/saved_model_h5"
  # tf.saved_model.save(h5_model, saved_model_h5_path)


  # Define the path to save the model
  # save_path = os.path.join(script_dir, "model/1")

  # # Save the model in SavedModel format for TensorFlow Serving
  # tf.saved_model.save(model, save_path)

if __name__ == "__main__":
  train_and_save_model()

# docker run -p 8501:8501 --name=tf_serving \ --mount type=bind,source=/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment7/model/1,target=/models/mymodel \ -e MODEL_NAME=mymodel -t tensorflow/serving
