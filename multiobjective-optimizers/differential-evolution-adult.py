import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTETomek
from scipy.optimize import differential_evolution


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv(url, names=column_names, na_values=' ?', skipinitialspace=True)

df = df.dropna()

df = df.drop(columns=['age', 'education', 'education-num', 'marital-status', 'occupation'])

features = ['workclass', 'relationship', 'race', 'sex', 'hours-per-week', 'native-country']
target = 'income'


categorical_cols = features
df = pd.get_dummies(df, columns=categorical_cols)


df[target] = df[target].apply(lambda x: 1 if x == '>50K' else 0)

X = df.drop([target], axis=1)
y = df[target]

print(X.columns)  

sensitive_feature = 'relationship_Male'  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

def compute_statistical_parity(y_pred, X_test):
    privileged = y_pred[(X_test[sensitive_feature] == 1)]
    unprivileged = y_pred[(X_test[sensitive_feature] == 0)]

    privileged_proportion = np.mean(privileged)
    unprivileged_proportion = np.mean(unprivileged)

    statistical_parity_difference = privileged_proportion - unprivileged_proportion

    return statistical_parity_difference

def compute_counterfactual_fairness(xgboost_model, X_test):
    original_probabilities = xgboost_model.predict_proba(X_test)[:, 1]

    counterfactual_X_test = X_test.copy()
    counterfactual_X_test[sensitive_feature] = 1 - counterfactual_X_test[sensitive_feature]

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
    statistical_parity = compute_statistical_parity(y_pred, X_test)

  
    weight_accuracy = 0.7
    weight_fairness = 0.3

    fairness_mean = np.mean(statistical_parity)

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
statistical_parity = compute_statistical_parity(y_pred, X_test)
counterfactual_fairness_original = compute_counterfactual_fairness(model, X_test)

print("Optimized Accuracy: {:.4f}".format(accuracy))
print("Optimized Statistical Parity (Group Fairness): {:.4f}".format(statistical_parity))
print("Optimized Counterfactual Fairness (Individual Fairness): {:.4f}".format(np.mean(counterfactual_fairness_original)))
