import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def sanity_checks():
  # loading data 
  path = "/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment3/sonar.all-data"
  data = pd.read_csv(path, header=None)
  print("\nSANITY CHECKS\n")
  print(data.info())
  print(data.describe())
  print(data.isnull().sum())
  print(data.iloc[:, -1].value_counts())
  return data
  # class imbalance, duplicates


def loading_preprocessing(data):
  # splitting feature and target variables 
  X = data.iloc[:, :-1] # all rows, all but last column
  y = data.iloc[:, -1] # all rows, only last column

  # encoding target variable (converting categorical to numerical)
  le = LabelEncoder()
  y = le.fit_transform(y)

  # splitting data into training and testing data 
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)

  # transforming feature variables to standardized values so that they contribute 
  # equally to the ML algorithm. Each feature now has mean of 0, SD of 1. 
  # doing this after splitting the data is ideal. 
  # scaler = StandardScaler()
  # X_train_scaled = scaler.fit_transform(X_train)
  # X_test_scaled = scaler.transform(X_test)
  return X_train, X_test, y_train, y_test
  # return X_train_scaled, X_test_scaled, y_train, y_test

def tuning_training(X_train_scaled, y_train):
  models = {}
  # decision tree hyperparameters
  dt_params = {
    'criterion' : ['gini', 'entropy'], # function to measure the quality of a split. gini — impurity, entropy — information gain
    # these next 3 hyperparameters help prevent overfitting
    'max_depth' : [None, 10, 20, 30], 
    'min_samples_split' : [2, 5, 10],  # minimum number of samples required to split an internal node.
    'min_samples_leaf' : [1, 2, 4]     # minimum number of samples required to be in a leaf node. 
  }

  # (base model to optimize, Hyperparameter list, number of cross-validation folds, testing metric, use all available CPU cores)
  dt_grid = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5, scoring='accuracy', n_jobs=-1)

  # fitting GridSearchCV object to training data — trains DT classifier on the training data using the best-found hyperparameters.
  dt_grid.fit(X_train_scaled, y_train)

  models['Decision Tree'] = dt_grid.best_estimator_ # best model found during Grid Search 

  # random forest hyperparameters 
  rf_params = {
    'n_estimators': [100, 200, 300], # number of trees in the forest 
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
  }

  rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring='accuracy', n_jobs=-1)
  rf_grid.fit(X_train_scaled, y_train)

  models['Random Forest'] = rf_grid.best_estimator_

  # SVC hyperparameters
  svc_params = {
    # regularization parameter. strength of the regularization is inversely proportional to C
    'C': [0.1, 1, 10, 100],      
    # kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. defines how far the influence of a single training example reaches.
    'gamma': [1, 0.1, 0.01, 0.001],     
    # specifies the kernel type to be used in the algorithm. Options - 'rbf', 'poly', and 'sigmoid'.
    'kernel': ['linear', 'poly', 'sigmoid']
  }

  svc_grid = GridSearchCV(SVC(), svc_params, cv=5, scoring='accuracy', n_jobs=-1)
  svc_grid.fit(X_train_scaled, y_train)

  models['SVC'] = svc_grid.best_estimator_

  return models

def testing(X_test_scaled, y_test, models):
  # Decision Tree : Accuracy = 0.66, many false positives 
  # Random Forest : Accuracy = 0.71
  # SVM : Accuracy = 0.83

  for name, model in models.items():
    y_pred = model.predict(X_test_scaled) 
    print(f'MODEL : {name}')
    print(f'ACCURACY : {accuracy_score(y_test, y_pred)}')
    print('CLASSIFICATION REPORT : ')
    print(classification_report(y_test, y_pred))
    print('CONFUSION MATRIX : ')
    print(confusion_matrix(y_test, y_pred))
    print('\n\n')

def run():
  df = sanity_checks()
  X_train_scaled, X_test_scaled, y_train, y_test = loading_preprocessing(df)
  models = tuning_training(X_train_scaled, y_train)
  testing(X_test_scaled, y_test, models)
  
run()