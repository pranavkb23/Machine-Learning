# Credit Card Default Prediction — Model Interpretability with Tree-Based Algorithms

## Overview
This project focuses on building a **credit default prediction model** using **tree-based algorithms** and exploring **model interpretability techniques**.  
The dataset contains anonymized credit card client information, and the objective is to predict whether a client will **default on their next payment**.

The project covers:
- Training Decision Tree, Random Forest, and XGBoost classifiers
- Evaluating classification performance
- Using **model interpretability tools** (LIME, Partial Dependence plots, surrogate models) for both **global** and **local** explanations

## Repository Contents
- `assignment_4.py` — Implementation of models and interpretability experiments.
- `default_of_credit_card_clients.csv` — Credit default dataset (UCI dataset).

## What’s Implemented

### 1. Dataset
- Source: UCI Default of Credit Card Clients Dataset
- Features include:
  - Demographics (age, sex, education, marital status)
  - Payment history
  - Bill statements
  - Previous payments
- Target: `default.payment.next.month` (binary classification)

### 2. Models
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **XGBoost Classifier**  
All tuned with cross-validation and hyperparameters.

### 3. Interpretability Methods
- **LIME (Local Interpretable Model-agnostic Explanations)**
  - Explains individual predictions by approximating locally with simpler models
- **Partial Dependence (PD) Plots**
  - Visualizes the marginal effect of selected features on predictions
- **Surrogate Models**
  - Train a simpler model (e.g., decision tree) to mimic complex models for global interpretability
- **Feature Importance Analysis**
  - Compares model-level insights across tree-based classifiers

### 4. Evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion matrix visualization
- ROC/AUC curves
- Side-by-side comparison of interpretability outputs

## Running the Code

### 1) Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install numpy pandas scikit-learn matplotlib seaborn lime xgboost

2) Running the Script
python assignment_4.py


This will:

Train models

Output evaluation metrics

Generate interpretability plots (saved to disk or displayed)

Sample Snippets

Training Random Forest

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)


Using LIME

import lime.lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                   feature_names=X_train.columns,
                                                   class_names=["No Default", "Default"],
                                                   discretize_continuous=True)
exp = explainer.explain_instance(X_test.iloc[0].values, rf.predict_proba, num_features=10)
exp.show_in_notebook(show_table=True)


Partial Dependence Plot

from sklearn.inspection import plot_partial_dependence
plot_partial_dependence(rf, X_train, ["LIMIT_BAL", "AGE"])

Key Learnings

Tree-based models offer high predictive performance, but can be opaque.

LIME provides instance-level explanations (why a specific prediction was made).

PD plots and surrogate models give global interpretability.

Combining these methods helps balance accuracy and transparency in financial decision-making.
