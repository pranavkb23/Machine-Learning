import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import lime 
import lime.lime_tabular 
import matplotlib.pyplot as plt
import shap

# loading data 
path = "/Users/pranavkbhandari/Desktop/Pranav/Courses/ComputationalLinguistics/Internships/BluCocoon/Assignment4/default_of_credit_card_clients.csv"
data = pd.read_csv(path)

print("\nSANITY CHECKS\n")
print(data.info())
print(data.describe())
print(data.isnull().sum())
print(data.iloc[:, -1].value_counts())

X = data.drop(columns=['ID','default payment next month'])
y = data['default payment next month']

print("Point 1")

# splitting data into training and testing data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Point 2")

# decision tree hyperparameters
dt_params = {
  'criterion' : ['gini', 'entropy'], # function to measure the quality of a split. gini — impurity, entropy — information gain
  # these next 3 hyperparameters help prevent overfitting
  'max_depth' : [None, 10, 20, 30], 
  'min_samples_split' : [2, 5, 10],  # minimum number of samples required to split an internal node.
  'min_samples_leaf' : [1, 2, 4]     # minimum number of samples required to be in a leaf node. 
}

print("Point 3")

# (base model to optimize, Hyperparameter list, number of cross-validation folds, testing metric, use all available CPU cores)
dt_grid = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5, scoring='accuracy', n_jobs=-1)
print("Point 4")
# fitting GridSearchCV object to training data — trains DT classifier on the training data using the best-found hyperparameters.
dt_grid.fit(X_train, y_train)
print("Point 5")
dt_best = dt_grid.best_estimator_ # best model found during Grid Search
print("Point 6")


# random forest hyperparameters 
rf_params = {
  'n_estimators': [10, 20], # number of trees in the forest 
  'criterion': ['gini', 'entropy'],
  'max_depth': [None, 10, 20],
  'min_samples_split': [2, 5, 10],
  'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring='accuracy', n_jobs=-1)
print("Point 7")
rf_grid.fit(X_train, y_train)
print("Point 8")

rf_best = rf_grid.best_estimator_

print("Point 9")

# Making predictions and evaluating the models
dt_predictions = dt_best.predict(X_test)
rf_predictions = rf_best.predict(X_test)
print("Point 8")

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_predictions))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))

# Generating classification reports
print("Decision Tree Report:\n", classification_report(y_test, dt_predictions))
print("Random Forest Report:\n", classification_report(y_test, rf_predictions))

# LIME explainer for a single instance using DT
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    class_names=['No Default', 'Default'],
    discretize_continuous=True
)
print("Point 10")

# Choose an instance to explain
instance_to_explain = X_test.iloc[0].values

# Explaining the instance 
explanation = explainer.explain_instance(
  data_row=instance_to_explain,
  predict_fn=dt_best.predict_proba,
  num_features=10
)

print("Point 11")

explanation.as_pyplot_figure()
plt.tight_layout()
##########################################################
print(explanation.as_list())

# # Display the explanation
# explainer.show_in_notebook(show_table=True, show_all=False)

# Partial Dependence Plots for RF
# Initialize SHAP explainer
shap_explainer = shap.TreeExplainer(rf_best)

# Compute SHAP values
shap_values = shap_explainer.shap_values(X_train)
print("Point 12")

# Plot Partial Dependence for a specific feature
shap.summary_plot(shap_values, X_train)

# Surrogate model to explain Random Forest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Generate predictions from the Random Forest model
rf_predictions_train = rf_best.predict_proba(X_train)[:, 1]

# Train a surrogate model
surrogate_model = LogisticRegression(random_state=42)
surrogate_model.fit(X_train, rf_predictions_train)

# Evaluate the surrogate model
surrogate_predictions = surrogate_model.predict_proba(X_test)[:, 1]
print("Surrogate Model AUC:", roc_auc_score(y_test, surrogate_predictions))

# Visualize feature importances for surrogate model
importances = pd.Series(surrogate_model.coef_[0], index=X_train.columns)
importances.sort_values().plot(kind='barh', title='Surrogate Model Feature Importances')
plt.show()




