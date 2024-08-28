import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTETomek
from scipy.optimize import differential_evolution

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                 header=None, sep=' ')


feature_names = ['status', 'duration', 'credit_history', 'purpose', 'amount',
                 'savings', 'employment_duration', 'installment_rate', 'statussex',
                 'other_debtors', 'residence_since', 'property',
                 'age', 'other_installment_plans', 'housing', 'number_credits',
                 'job', 'people_liable', 'telephone', 'foreign_worker',
                 'credit_risk']

df.columns = feature_names


categorical_cols = ['status', 'credit_history', 'purpose', 'savings', 'employment_duration',
                    'other_debtors', 'property', 'other_installment_plans',
                    'housing', 'job', 'telephone', 'statussex', 'foreign_worker']

df = pd.get_dummies(df, columns=categorical_cols)
df.credit_risk.replace([1, 2], [1, 0], inplace=True)

target_col = 'credit_risk'
sensitive_features = ['statussex_A92', 'age']

X = df.drop([target_col], axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

# Define functions for fairness metrics
def compute_statistical_parity(y_pred, X_test):
    privileged = y_pred[(X_test['statussex_A92'] == 1)]
    unprivileged = y_pred[(X_test['statussex_A92'] == 0)]

    privileged_proportion = np.mean(privileged)
    unprivileged_proportion = np.mean(unprivileged)

    statistical_parity_difference = privileged_proportion - unprivileged_proportion

    return statistical_parity_difference

def compute_counterfactual_fairness(xgboost_model, X_test):
    original_probabilities = xgboost_model.predict_proba(X_test)[:, 1]

    counterfactual_X_test = X_test.copy()
    counterfactual_X_test['statussex_A92'] = 1 - counterfactual_X_test['statussex_A92']

    counterfactual_probabilities = xgboost_model.predict_proba(counterfactual_X_test)[:, 1]

    counterfactual_fairness_original = np.abs(original_probabilities - counterfactual_probabilities)

    return counterfactual_fairness_original

def objective_function(x):
    learning_rate, max_depth, n_estimators = x

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'seed': 42
    }

    model = XGBClassifier(**params)
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    individual_fairness = compute_statistical_parity(y_pred, X_test)

    weight_accuracy = 0.7
    weight_fairness = 0.3

    fairness_mean = np.mean(individual_fairness)

    objective_value = -(weight_accuracy * accuracy) + (weight_fairness * fairness_mean)

    return objective_value

bounds = [(0.001, 0.5), (1, 10), (10, 200)]

result = differential_evolution(objective_function, bounds, maxiter=10)

best_params = result.x
best_cost = result.fun

print("Best Parameters:", best_params)
print("Best Cost:", best_cost)

best_learning_rate, best_max_depth, best_n_estimators = best_params

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'max_depth': int(best_max_depth),
    'learning_rate': best_learning_rate,
    'n_estimators': int(best_n_estimators),
    'seed': 42
}

model = XGBClassifier(**params)
model.fit(X_train_balanced, y_train_balanced)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
individual_fairness = compute_statistical_parity(y_pred, X_test)
counterfactual_fairness_original = compute_counterfactual_fairness(model, X_test)

print("Optimized Accuracy: {:.4f}".format(accuracy))
print("Optimized Statistical Parity (Group Fairness): {:.4f}".format(individual_fairness))
print("Optimized Counterfactual Fairness (Individual Fairness): {:.4f}".format(np.mean(counterfactual_fairness_original)))
