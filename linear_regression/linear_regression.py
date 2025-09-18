import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegressionGD:
  def __init__(self, learning_rate, n_iterations):
    self.learning_rate = learning_rate
    self.n_iterations = n_iterations
    self.theta = None

  def fit(self, X, y):
    # adding the bias term
    X_b = np.c_[np.ones((X.shape[0], 1)), X.reshape(-1,1)]
    m = X_b.shape[0]
    self.theta = np.zeros((2,1))

    # reshaping y to be a column vector 
    y = y.reshape(-1,1)

    # gradient Descent 
    for i in range(self.n_iterations):
      gradients = 2/m * X_b.T.dot(X_b.dot(self.theta) - y)
      self.theta = self.theta - (self.learning_rate * gradients)

  def predict(self, X):
    X_b = np.c_[np.ones((X.shape[0], 1)), X.reshape(-1, 1)]  
    return X_b.dot(self.theta)
  

def main():
  # loading the data set 
  data = pd.read_csv("/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment 2/assignment2_data.csv")

  # extracting feature and target variables 
  X = data['X'].values
  y = data['Y'].values
  X = (X - np.mean(X)) / np.std(X) # standardizing the data

  # initializing and fitting the model 
  model_lr = LinearRegressionGD(0.01, 1000)
  model_lr.fit(X, y)

  # making the predictions 
  y_pred = model_lr.predict(X)

  # printing the final parameters 
  print("Parameters (Theta) : ", model_lr.theta)

  # print(data.isnull().sum())
  # print(np.isfinite(data).all())

  # evaluation metrics
  # mean squared error
  mse = np.mean((y_pred - y) ** 2)  

  # R-squared
  ss_total = np.sum((y - np.mean(y)) ** 2) # total sum of squares
  ss_residual = np.sum((y - y_pred) ** 2) # residual sum of squares
  r_squared = 1 - (ss_residual / ss_total) 

  print("Mean squared error : ", mse)
  print("\n")
  print("ss_total : ", ss_total)
  print("\n")
  print("ss_residual : ", ss_residual)
  print("\n")
  print("R-squared : ", r_squared)
  print("\n")

  # calculations for line of best fit
  a, b = np.polyfit(X, y, 1)

  # plotting the results 
  plt.plot(X, y, "b.")
  # plt.plot(X, y_pred, "r-")
  plt.plot(X, a*X+b) # for line of best fit

  plt.xlabel("Feature variable")
  plt.ylabel("Target variable")
  plt.title("Linear Regression using Gradient Descent")
  plt.show()

if __name__ == "__main__":
  main()
