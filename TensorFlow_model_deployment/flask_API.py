from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load the saved model
model = tf.keras.models.load_model('/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment7/model.h5')

print(model.summary())

app = Flask(__name__)

# how to take image input in flask and convert to compatible format for trained TF model. 

# Defining a route for prediction
@app.route('/predict', methods=['POST'])
def predict():

  if request.files.get("image"):
    file = request.files['image']

  # If the user does not select a file, the browser submits an empty file without a filename
  if file.filename == '':
    return jsonify({'error': 'No selected file'}), 400
  
  # Open the image file
  image = Image.open(file).convert('L')  # Convert to grayscale

  # Resize the image to 28x28 pixels
  image = image.resize((28, 28))

  # Convert the image to a numpy array and normalize pixel values
  image_data = np.array(image) / 255.0
  image_data = image_data.reshape(1, 28, 28)  # Reshape for model input

  # # Get the JSON data from the request
  # data = request.get_json(force=True)

  # # Converting data into numpy array and preprocess it
  # # Assuming data['image'] contains the pixel data
  # image_data = np.array(data['image']).reshape((1, 28, 28)) / 255.0

  # Make prediction
  predictions = model.predict(image_data)
  
  # Get the predicted class
  predicted_class = np.argmax(predictions, axis=1)[0]
  
  # Return the prediction as a JSON response
  return jsonify({'predicted_class': int(predicted_class)})

# Run the app
if __name__ == '__main__':
  app.run(debug=True)