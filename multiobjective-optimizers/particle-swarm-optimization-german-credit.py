import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import pyswarms as ps
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.metrics import recall_score, precision_score, accuracy_score, classification_report, f1_score
from sklearn.metrics import precision_recall_curve
import xgboost as xgb

def compute_statistical_parity(xgboost_y_pred, X_test):
  privileged = xgboost_y_pred[(X_test['statussex_A91'] == 1) | (X_test['statussex_A93'] == 1) | (X_test['statussex_A94'] == 1) & (X_test['age'] > 18)]
  unprivileged = xgboost_y_pred[(X_test['statussex_A92'] == 1) & (X_test['age'] > 18)]
  
  privileged_proportion = np.mean(privileged)
  unprivileged_proportion = np.mean(unprivileged)

  statistical_parity_difference = privileged_proportion - unprivileged_proportion

  return statistical_parity_difference

def compute_counterfactual_fairness(xgboost_model, X_test):
    original_predictions = xgboost_model.predict(X_test)
    
    # Create counterfactual instances by flipping the sensitive attribute
    counterfactual_X_test = X_test.copy()
    counterfactual_X_test['statussex_A92'] = 1 - counterfactual_X_test['statussex_A92']

    # Predict probabilities for original and counterfactual instances
    original_probabilities = xgboost_model.predict_proba(X_test)[:, 1]
    counterfactual_probabilities = xgboost_model.predict_proba(counterfactual_X_test)[:, 1]

    # Calculate counterfactual fairness metrics for original instances
    counterfactual_fairness_original = np.abs(original_probabilities - counterfactual_probabilities)

    return counterfactual_fairness_original

# Load the data
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                   header=None, sep=' ')

# Define feature names
feature_names = ['status', 'duration', 'credit_history', 'purpose', 'amount',
                 'savings', 'employment_duration', 'installment_rate', 'statussex',
                 'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
                 'housing', 'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
                 'credit_risk']

# Assign the feature names to the dataset
data.columns = feature_names

# Converted categorical variables to numerical using one-hot encoding
categorical_cols = ['status', 'credit_history', 'purpose', 'savings', 'employment_duration',
                    'statussex', 'other_debtors', 'property', 'other_installment_plans',
                    'housing', 'job', 'telephone', 'foreign_worker']
data = pd.get_dummies(data, columns=categorical_cols)
data.credit_risk.replace([1,2], [1,0], inplace=True)

# Defined the target variable and the features
target_col = 'credit_risk'
features = data.drop(target_col, axis=1)
target = data[target_col]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

#sensitive features
sensitive_features = ['statussex_A91', 'statussex_A92', 'statussex_A93', 'statussex_A94', 'age']


# Defined the XGBoost model
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'seed': 42,
}


def objective_function(self):
    model = XGBClassifier(**params)
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)

    individual_fairness = compute_statistical_parity(y_pred, X_test)
    counterfactual_fairness_original = compute_counterfactual_fairness(model, X_test)
    print(accuracy*100, "%")
    print("Statistical Parity (Individual Fairness):", individual_fairness)
    print("Counterfactual Fairness (Individual Fairness):", np.mean(counterfactual_fairness_original))

    return [accuracy, individual_fairness]

# Defined the search space
min_bound = [0.001, 1, 10]
max_bound = [0.5, 10, 200]
bounds = (min_bound, max_bound)

# Initialized the particle swarm
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
num_particles = 2
dimensions = len(min_bound)
optimizer = ps.single.GlobalBestPSO(n_particles=num_particles, dimensions=dimensions, options=options, bounds=bounds)

cost, pos = optimizer.optimize(objective_function, iters=10)
print("cost ", cost)
print("pos ", pos)
