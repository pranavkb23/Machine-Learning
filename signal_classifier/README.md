# Sonar Signal Classification — Decision Trees, Random Forests & SVC

## Overview
This project builds a **binary classifier** to distinguish between sonar signals bounced off a **metal cylinder** and those bounced off a **rock**.  
The dataset used is the classic **Sonar dataset** (60 features, continuous values).  

The work involves:
- Preprocessing the sonar dataset
- Training **Decision Tree**, **Random Forest**, and **Support Vector Classifier (SVC)** models
- Performing **hyperparameter tuning**
- Comparing classification performance across models

## Repository Contents
- `signal_classifier.py` — Implementation using Decision Trees and Random Forests.
- `signal_classifier2.py` — Implementation using SVC and extended comparisons.
- `sonar.all-data` — Dataset (UCI Sonar Dataset).

## What’s Implemented

### 1. Dataset
- Input: 208 samples, each with 60 sonar frequency energy features.
- Target: `M` (metal cylinder) or `R` (rock).
- Preprocessing:
  - Convert categorical labels into numerical form.
  - Split into training and test sets.
  - Normalize features for SVC compatibility.

### 2. Models
- **Decision Tree Classifier**
  - Explores criteria (`gini`, `entropy`)
  - Hyperparameter tuning: max depth, min samples split
- **Random Forest Classifier**
  - Combines multiple trees for stability
  - Tuned via number of estimators, max features
- **Support Vector Classifier (SVC)**
  - Tested with linear, polynomial, and RBF kernels
  - Tuned via C, gamma, kernel choice

### 3. Evaluation
- Train/test split evaluation
- Accuracy, precision, recall, F1-score
- Confusion matrix visualization
- Comparison table of model performance
- Notes on overfitting vs. generalization

## Running the Code

### 1) Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install numpy pandas scikit-learn matplotlib seaborn

2) Running Decision Trees & Random Forest
python assignment_3.py

3) Running SVC & Extended Comparisons
python assignment_3_2nd.py

Sample Snippets

Decision Tree Training

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5, criterion="entropy")
clf.fit(X_train, y_train)


Random Forest Training

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_features="sqrt")
rf.fit(X_train, y_train)


SVC Training

from sklearn.svm import SVC
svc = SVC(kernel="rbf", C=1.0, gamma="scale")
svc.fit(X_train, y_train)


Performance Evaluation

from sklearn.metrics import classification_report, confusion_matrix
y_pred = svc.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

Key Learnings

Decision Trees are interpretable but prone to overfitting.

Random Forests improve stability and often outperform a single tree.

SVC works well with normalized features, especially with non-linear kernels.

Hyperparameter tuning is critical to balancing bias and variance.

Comparing models provides insights into trade-offs between interpretability and accuracy.
