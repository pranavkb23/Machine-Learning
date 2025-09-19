import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, confusion_matrix

# loading training and test data set
train_data = pd.read_csv("/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment5/MNIST_data_csv/mnist_train.csv")
test_data = pd.read_csv("/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment5/MNIST_data_csv/mnist_test.csv")

# separating features and label 
# dividing to normalize pixel values between 0 and 1, use floating point to ensure proper normalization
X_train = train_data.iloc[:, 1:].values.T / 255.0
Y_train = train_data.iloc[:, 0].values

X_test = test_data.iloc[:, 1:].values.T / 255.0 # dividing to normalize pixel values between 0 and 1
Y_test = test_data.iloc[:, 0].values

# one-hot encoding labels 
def one_hot_encode(labels, num_classes):
  one_hot = np.zeros((num_classes, labels.size))
  one_hot[labels, np.arange(labels.size)] = 1
  return one_hot

Y_train = one_hot_encode(Y_train, 10)
Y_test = one_hot_encode(Y_test, 10)

# defining network architecture i.e. how many neurons in each layer
input_dim = 784 # 28x28
hidden_dim = 64 
output_dim = 10 # no. of classes (here, digits)

# initializing weights and biases 
np.random.seed(1) # setting seed, ensuring that random numbers generated for iteration of the program will be the same. useful for debugging. 

# from input to hidden layer
W1 = np.random.randn(hidden_dim, input_dim) * 0.01 # small initial weights help prevent activation functions from saturating too early
b1 = np.zeros((hidden_dim, 1)) # creates a column vector of shape (hidden_dim, 1)


W2 = np.random.randn(output_dim, hidden_dim) * 0.01
b2 = np.zeros((output_dim, 1)) 

# Breaking Symmetry: If all weights were initialized to zero, every neuron in the layer would learn the same features during training. 
# Random initialization breaks this symmetry, allowing neurons to learn different features.

# activation functions (forward propagation) and derivatives (computing gradient in backward propagation)
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)

# rectified linear unit
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def initialize_parameters(input_dim, hidden_dim, output_dim):
  np.random.seed(1)
  W1 = np.random.randn(hidden_dim, input_dim) * 0.01
  b1 = np.zeros((hidden_dim, 1))
  W2 = np.random.randn(output_dim, hidden_dim) * 0.01
  b2 = np.zeros((output_dim, 1))
  return W1, b1, W2, b2

# forward propagation 
# X : Input data matrix of shape (input_dim, number_of_samples). Each column represents a single sample
# W1 : Weights matrix for the first layer (hidden layer) of shape (hidden_dim, input_dim)
# b1 : Bias vector for the first layer of shape (hidden_dim, 1)
# W2 : Weights matrix for the second layer (output layer) of shape (output_dim, hidden_dim)
# b2 : Bias vector for the second layer of shape (output_dim, 1)
def forward_propagation(X, W1, b1, W2, b2):
  # Compute the linear combination of inputs and weights for the first layer. matrix multiplication between the weights of the first layer and the input data
  # Z1 : matrix of shape (hidden_dim, number_of_samples) 
  Z1 = np.dot(W1, X) + b1 # broadcasting in NumPy adds b1 to each column of Z1
  A1 = relu(Z1) # introducing non-linearity -> learn more complex patterns

  # Compute the linear combination of activations from the first layer and weights for the second layer
  # Z2 : matrix of shape (output_dim, number_of_samples)
  Z2 = np.dot(W2, A1) + b2 
  # squashes output to range [0, 1], suitable for classification tasks (we want to interpret the output as probabilities)
  A2 = sigmoid(Z2) 

  # Return intermediate computations needed for backpropagation
  return Z1, A1, Z2, A2


# calculate the cross-entropy loss between the true labels Y and the predicted probabilities A2
# Y : true labels, represented as a one-hot encoded matrix, shape (output_dim, number_of_samples)
# A2 : predicted probabilities from output layer of the network, shape (output_dim, number_of_samples)
def compute_loss(Y, A2):
  m = Y.shape[1] # number of columns in Y, which is the total number of samples
  # Compute the average cross-entropy loss over all samples (formula for Cross-Entropy Loss for Binary Classification)
  # Y * np.log(A2): computes the log loss for the positive class (when Y = 1)
  # (1 - Y) * np.log(1 - A2): computes the log loss for the negative class (when Y = 0)
  # sum the losses across all classes and samples, average the loss (divide by m)
  loss = -1/m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) 
  return loss


# X: Input data matrix of shape (input_dim, number_of_samples).
# Y: True labels, one-hot encoded, of shape (output_dim, number_of_samples).
# Z1: Linear combination (pre-activation) from the first layer.
# A1: Activation from the first layer.
# Z2: Linear combination (pre-activation) from the second layer.
# A2: Activation from the second layer (output).
# W1: Weights matrix for the first layer.
# W2: Weights matrix for the second layer.
def backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2):
  m = X.shape[1] # number of columns in X, which is the total number of samples

  # Gradients for OUTPUT LAYER

  # Cross-entropy loss combined with a sigmoid activation function,
  # gradient of the loss with respect to Z2 is the difference A2 - Y
  dZ2 = A2 - Y

  # dZ2 has shape (output_dim, number_of_samples)
  # A1.T (transpose of A1) has shape (number_of_samples, hidden_dim)
  # dot product np.dot(dZ2, A1.T) results in a matrix of shape (output_dim, hidden_dim)
  # 1/m * ... averages the gradients over all samples
  dW2 = 1/m * np.dot(dZ2, A1.T)

  # np.sum(dZ2, axis=1, keepdims=True) sums the gradients across all samples, resulting in a vector of shape (output_dim, 1)
  # averaging gradients by m
  db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

  # Gradients for HIDDEN LAYER
  
  # gradient of loss w.r.t. A1 
  # W2.T (transpose of W2) has shape (hidden_dim, output_dim).
  # dZ2 has shape (output_dim, number_of_samples).
  # The dot product np.dot(W2.T, dZ2) results in a matrix of shape (hidden_dim, number_of_samples).
  dA1 = np.dot(W2.T, dZ2)

  # gradient of loss w.r.t. Z1 
  # dA1 is the gradient of the loss with respect to the activations from the first layer.
  # relu_derivative(Z1) computes the derivative of the ReLU activation function applied to Z1.
  # Element-wise multiplication * results in dZ1, which has shape (hidden_dim, number_of_samples)
  dZ1 = dA1 * relu_derivative(Z1)

  # gradient of loss w.r.t. W1
  # dZ1 has shape (hidden_dim, number_of_samples).
  # X.T (transpose of X) has shape (number_of_samples, input_dim).
  # The dot product np.dot(dZ1, X.T) results in a matrix of shape (hidden_dim, input_dim).
  # 1/m * ... averages the gradients over all samples
  dW1 = 1/m * np.dot(dZ1, X.T)

  # gradient of loss w.r.t. b1
  # dZ1 has shape (hidden_dim, number_of_samples).
  # np.sum(dZ1, axis=1, keepdims=True) sums the gradients across all samples, resulting in a vector of shape (hidden_dim, 1).
  # 1/m * ... averages the gradients over all samples
  db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
  
  return dW1, db1, dW2, db2

# Update weights and biases of the neural network using gradients computed from backward propagation.
# W1: Weights matrix for the first layer (hidden layer).
# b1: Bias vector for the first layer.
# W2: Weights matrix for the second layer (output layer).
# b2: Bias vector for the second layer.
# dW1: Gradient of the loss with respect to W1.
# db1: Gradient of the loss with respect to b1.
# dW2: Gradient of the loss with respect to W2.
# db2: Gradient of the loss with respect to b2.
# learning_rate: Scalar value that determines the step size for updating the parameters.
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    # the scaled gradient (learning_rate * gradient) is subtracted from the current
    # bias/weight, moving it in the direction that reduces loss
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Train the neural network by iteratively updating the weights and biases using the gradients computed from backpropagation
# X: Input data matrix of shape (input_dim, number_of_samples).
# Y: True labels, one-hot encoded, of shape (output_dim, number_of_samples).
# input_dim: Number of input features (e.g., 784 for MNIST).
# hidden_dim: Number of neurons in the hidden layer.
# output_dim: Number of output classes (e.g., 10 for MNIST digits).
# num_iterations: Number of training iterations.
# learning_rate: Learning rate for gradient descent.
def train_neural_network(X, Y, input_dim, hidden_dim, output_dim, num_iterations, learning_rate):
  # sets up the initial weights and biases for hidden and output layers 
  # with small random values for weights and zeros for biases
  W1, b1, W2, b2 = initialize_parameters(input_dim, hidden_dim, output_dim)
  

  for i in range(num_iterations):
    # computes the output (linear combination and activation) of each layer in the network.
    Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
    # Calculate cross-entropy loss between predicted output A2 and true labels Y, measuring networks performance
    loss = compute_loss(Y, A2)
    # Compute the gradients of the loss with respect to the weights and biases
    dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2)
    # adjust weights and biases based on the gradients and the learning rate.
    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

    if i % 100 == 0:
      print(f"Iteration {i}, Loss: {loss}")

  return W1, b1, W2, b2

# Prediction function, generate predictions using the trained neural network
# takes the input data X and the trained weights and biases (W1, b1, W2, b2)
# calls forward_propagation to compute the network's output (predictions).
# A2, the output of the forward propagation, represents the predicted probabilities for each class
def predict(X, W1, b1, W2, b2):
  _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
  return A2

# Evaluation function
def evaluate_predictions(Y_true, Y_pred):
  Y_true_classes = np.argmax(Y_true, axis=0) # Converts the one-hot encoded true labels to class labels
  Y_pred_classes = np.argmax(Y_pred, axis=0) # Converts the predicted probabilities to class labels
  
  mse = mean_squared_error(Y_true_classes, Y_pred_classes)
  mae = mean_absolute_error(Y_true_classes, Y_pred_classes)

  # Avoid division by zero by masking zeros
  mask = Y_true_classes != 0
  mpe = np.mean((Y_true_classes[mask] - Y_pred_classes[mask]) / Y_true_classes[mask]) * 100  # Mean Percentage Error

  class_report = classification_report(Y_true_classes, Y_pred_classes)
  conf_matrix = confusion_matrix(Y_true_classes, Y_pred_classes)
  
  return mse, mae, mpe, class_report, conf_matrix

num_iterations = 1000
learning_rate = 0.01

W1, b1, W2, b2 = train_neural_network(X_train, Y_train, input_dim, hidden_dim, output_dim, num_iterations, learning_rate)
  
# Prediction for training data
train_predictions = predict(X_train, W1, b1, W2, b2)

# Evaluate on training data
mse_train, mae_train, mpe_train, class_report_train, conf_matrix_train = evaluate_predictions(Y_train, train_predictions)
print("Training Metrics:")
print(f"MSE: {mse_train}")
print(f"MAE: {mae_train}")
print(f"MPE: {mpe_train}")
print("Classification Report:\n", class_report_train)
print("Confusion Matrix:\n", conf_matrix_train)

# prediction for test data
test_predictions = predict(X_test, W1, b1, W2, b2)
# Evaluate on test data
mse_test, mae_test, mpe_test, class_report_test, conf_matrix_test = evaluate_predictions(Y_test, test_predictions)
print("Testing Metrics:")
print(f"MSE: {mse_test}")
print(f"MAE: {mae_test}")
print(f"MPE: {mpe_test}")
print("Classification Report:\n", class_report_test)
print("Confusion Matrix:\n", conf_matrix_test)

# Precision: Ratio of true positives to total predicted positives (how many of the predicted digits are actually correct)
# Recall: Ratio of true positives to total actual positives (how many of the actual digits are correctly predicted)
#
# F1-score: The harmonic mean of precision and recall, providing a single measure of the model's performance. 
# Measure of a test's accuracy in the context of binary classification (and can be extended to multi-class classification)
# particularly useful when you need to account for both false positives and false negatives, especially in scenarios where class distribution is imbalanced.
# range : (0, 1) (from 0 correct identifications to 100% correct)
# 
# Class Imbalance: In cases where certain classes are significantly less frequent than others, the F1-score becomes crucial. 
# Accuracy alone can be misleading because a model can achieve high accuracy by simply predicting the majority class.
# Balancing Precision and Recall: The F1-score helps ensure that the model performs well in identifying the minority class, 
# maintaining a balance between precision (minimizing false positives) and recall (minimizing false negatives).

# Training Metrics:
# MSE: 4.1668666666666665
# MAE: 0.9013333333333333
# MPE: 1.8982595657984453
# Classification Report:
#                precision    recall  f1-score   support

#            0       0.83      0.96      0.89      5923
#            1       0.77      0.97      0.86      6742
#            2       0.83      0.74      0.78      5958
#            3       0.61      0.84      0.70      6131
#            4       0.70      0.80      0.75      5842
#            5       0.90      0.25      0.39      5421
#            6       0.82      0.91      0.87      5918
#            7       0.75      0.87      0.80      6265
#            8       0.74      0.61      0.67      5851
#            9       0.69      0.47      0.56      5949

#     accuracy                           0.75     60000
#    macro avg       0.76      0.74      0.73     60000
# weighted avg       0.76      0.75      0.73     60000


# Testing Metrics:
# MSE: 3.9137
# MAE: 0.8563
# MPE: 2.5128198359905682
# Classification Report:
#                precision    recall  f1-score   support

#            0       0.82      0.98      0.89       980
#            1       0.81      0.99      0.89      1135
#            2       0.84      0.72      0.78      1032
#            3       0.61      0.87      0.72      1010
#            4       0.69      0.81      0.75       982
#            5       0.91      0.26      0.40       892
#            6       0.83      0.90      0.86       958
#            7       0.75      0.86      0.80      1028
#            8       0.76      0.62      0.69       974
#            9       0.71      0.50      0.59      1009

#     accuracy                           0.76     10000
#    macro avg       0.77      0.75      0.74     10000
# weighted avg       0.77      0.76      0.74     10000

