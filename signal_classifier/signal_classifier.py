import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SonarSignalClassifier:
  
  def __init__(self, path):
    self.path = path
    # self.data = None
    # self.X_train, self.X_test = None
    # self.y_train, self.y_test = None
    self.models = {}


  def sanity_checks(self):
    print("\nSANITY CHECKS\n")
    print(self.data.info())
    print(self.data.describe())
    print(self.data.isnull().sum())
    # class imbalance, duplicates


  def loading_preprocessing(self):
    # loading data 
    self.data = pd.read_csv(self.path)

    # splitting feature and target variables 
    X = self.data.iloc[:, :-1] # all rows, all but last column
    y = self.data.iloc[:, -1] # all rows, only last column

    # encoding target variable (converting categorical to numerical)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # splitting data into training and testing data 
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)

    # transforming feature variables to standardized values so that they contribute 
    # equally to the ML algorithm. Each feature now has mean of 0, SD of 1. 
    # doing this after splitting the data is ideal. 
    scaler = StandardScaler()
    self.X_train_scaled = scaler.fit_transform(self.X_train)
    self.X_test_scaled = scaler.transform(self.X_test)


  def tuning_training(self):
    # decision tree hyperparameters
    dt_params = {
      'criterion' : ['gini', 'entropy'], # function to measure the quality of a split. gini — impurity, entropy — information gain
      'max_depth' : [None, 10, 20, 30], 
      'min_samples_split' : [2, 5, 10],  # minimum number of samples required to split an internal node.
      'min_samples_leaf' : [1, 2, 4]     # minimum number of samples required to be at a leaf node. 
    }

    # (base model to optimize, Hyperparameter list, number of cross-validation folds, testing metric, use all available CPU cores)
    dt_grid = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5, scoring='accuracy', n_jobs=-1)

    # fitting GridSearchCV object to training data — trains DT classifier on the training data using the best-found hyperparameters.
    dt_grid.fit(self.X_train_scaled, self.y_train)

    self.models['Decision Tree'] = dt_grid.best_estimator_ # best model found during Grid Search 

    # random forest hyperparameters 
    rf_params = {
      'n_estimators': [100, 200, 300], # number of trees in the forest 
      'criterion': ['gini', 'entropy'],
      'max_depth': [None, 10, 20, 30],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4]
    }

    rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(self.X_train_scaled, self.y_train)

    self.models['Random Forest'] = rf_grid.best_estimator_

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
    svc_grid.fit(self.X_train_scaled, self.y_train)

    self.models['SVC'] = svc_grid.best_estimator_


  def testing(self):
    for name, model in self.models.values():
      y_pred = model.predict(self.X_test) 
      print(f'MODEL : {name}')
      print(f'ACCURACY : {accuracy_score(self.y_test, y_pred)}')
      print('CLASSIFICATION REPORT : ')
      print(classification_report(self.y_test, y_pred))
      print('CONFUSION MATRIX : ')
      print(confusion_matrix(self.y_test, y_pred))
      print('\n\n')


  def run(self):
    # self.sanity_checks()
    self.loading_preprocessing()
    self.testing()


if __name__ == "__main__":
  classifier = SonarSignalClassifier("/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment3/sonar.all-data")
  classifier.run()
  