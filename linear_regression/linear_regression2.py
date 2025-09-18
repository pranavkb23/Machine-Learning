import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegressionGD: 
  def __init__(self, learning_rate, n_iterations):
    self.learning_rate = learning_rate
    self.n_iterations = n_iterations
    self.theta0 = 0   # Intercept
    self.theta1 = 0   # Slope

  def fit(self, X, y):
    m = len(y)
    print(f"Initial Theta: {self.theta0}, {self.theta1}")
    for i in range(self.n_iterations):
      y_pred = self.theta0 + (self.theta1 * X)
      d_theta0 = (1/m) * np.sum(y_pred - y)
      d_theta1 = (1/m) * np.sum((y_pred - y) * X)
      self.theta0 -= self.learning_rate * d_theta0
      self.theta1 -= self.learning_rate * d_theta1
      if i % 100 == 0:  # Print every 100 iterations
        print(f"Iteration {i}: Theta0 = {self.theta0}, Theta1 = {self.theta1}")

  def predict(self, X):
    return self.theta0 + (self.theta1 * X)
  
  def eval_stats(self, X, y):
    y_pred = self.predict(X)
    mse = np.mean((y_pred - y) ** 2)
    mae = np.mean(np.abs(y - y_pred))
    mape = np.mean(np.abs((y - y_pred) / y) * 100)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    
    return mse, mae, mape, ss_total, ss_residual, r_squared
  
def main():
  # loading data
  data = pd.read_csv("/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment2/assignment2_data.csv")
  print("Data Loaded")

  # cleaning null values if any
  if data.isnull().sum().any():
    data = data.dropna()
  print("Data Cleaned")

  # print(f"Feature and Target Extracted: X.shape = {X.shape}, y.shape = {y.shape}")

  # extracting independent and dependent variables
  X = data['X'].values
  y = data['Y'].values

  # Normalizing X to prevent nan in evaluation statistics
  X = (X - np.mean(X)) / np.std(X)
  print("Feature Normalized")

  model = LinearRegressionGD(0.01, 1000)
  model.fit(X, y)
  print("Model Fitted")

  y_pred = model.predict(X)
  print("Predictions Made")

  # printing final paremeters
  print('Theta0 (Intercept) : ', model.theta0)
  print('Theta1 (Slope) : ', model.theta1)

  mse, mae, mape, ss_total, ss_residual, r_squared = model.eval_stats(X, y)
  print("MSE : ", mse)
  print("MAE : ", mae)
  print("MAPE : ", mape)
  print("ss_total : ", ss_total)
  print("ss_residual : ", ss_residual)
  print("r_squared : ", r_squared)

  # calculations for line of best fit
  a, b = np.polyfit(X, y, 1)

  # plotting results 
  plt.plot(X, y, "b.")
  # plt.plot(X, y_pred, "r-")
  plt.plot(X, a*X+b) # for line of best fit
  plt.xlabel("X")
  plt.ylabel("y")
  plt.title("Linear Regression using Gradient Descent")
  plt.show()

if __name__ == "__main__":
  main()

