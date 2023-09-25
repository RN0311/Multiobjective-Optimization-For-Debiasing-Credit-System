import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.combine import SMOTETomek


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                 header=None, sep=' ')

feature_names = ['status', 'duration', 'credit_history', 'purpose', 'amount',
                 'savings', 'employment_duration', 'installment_rate', 'statussex',
                 'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
                 'housing', 'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
                 'credit_risk']

df.columns = feature_names


categorical_cols = ['status', 'credit_history', 'purpose', 'savings', 'employment_duration',
                    'other_debtors', 'property', 'other_installment_plans',
                    'housing', 'job', 'telephone', 'statussex', 'foreign_worker']

df = pd.get_dummies(df, columns=categorical_cols)
df.credit_risk.replace([1, 2], [1, 0], inplace=True)

target_col = 'credit_risk'
sensitive_features = ['statussex', 'age']

X = df.drop([target_col], axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

counterfactual_X_test = X_test.copy()

# Flipped the sensitive attribute for females ('statussex_A92')
counterfactual_X_test['statussex_A92'] = 1 - counterfactual_X_test['statussex_A92']

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'seed': 42
}

xgb = XGBClassifier(**params)
xgb.fit(X_train_balanced, y_train_balanced)

y_prob = xgb.predict_proba(X_test)[:, 1]

alpha = 0.65

y_prob_sorted = np.sort(y_prob)
reject_threshold = y_prob_sorted[int(alpha * len(y_prob))]

accepted_mask = (y_prob >= reject_threshold)
rejected_mask = (y_prob < reject_threshold)

accuracy_accepted = accuracy_score(y_test[accepted_mask], xgb.predict(X_test[accepted_mask]))
accuracy_rejected = accuracy_score(y_test[rejected_mask], xgb.predict(X_test[rejected_mask]))

# Calculated fairness (Statistical Parity) for accepted and rejected instances
group_fairness_accepted = (y_test[accepted_mask] == 1).mean() - (xgb.predict(X_test[accepted_mask]) == 1).mean()
group_fairness_rejected = (y_test[rejected_mask] == 1).mean() - (xgb.predict(X_test[rejected_mask]) == 1).mean()

print('Accuracy on Accepted Instances: {:.4f}'.format(accuracy_accepted))
print('Accuracy on Rejected Instances: {:.4f}'.format(accuracy_rejected))


print('Group Fairness on Accepted Instances (Statistical Parity): {:.4f}'.format(group_fairness_accepted))
print('Group Fairness on Rejected Instances (Statistical Parity): {:.4f}'.format(group_fairness_accepted))


y_prob_counterfactual = xgb.predict_proba(counterfactual_X_test)[:, 1]

counterfactual_fairness_original = np.abs(y_prob - y_prob_counterfactual)

alpha = 0.65

counterfactual_fairness_sorted = np.sort(counterfactual_fairness_original)
reject_threshold = counterfactual_fairness_sorted[int(alpha * len(counterfactual_fairness_original))]

accepted_mask = (counterfactual_fairness_original >= reject_threshold)
rejected_mask = (counterfactual_fairness_original < reject_threshold)

# Calculate counterfactual fairness metrics for accepted and rejected instances
counterfactual_fairness_accepted = counterfactual_fairness_original[accepted_mask]
counterfactual_fairness_rejected = counterfactual_fairness_original[rejected_mask]

accuracy_accepted = accuracy_score(y_test[accepted_mask], xgb.predict(X_test[accepted_mask]))
accuracy_rejected = accuracy_score(y_test[rejected_mask], xgb.predict(X_test[rejected_mask]))

print('Individual Fairness on Accepted Instances (Counterfactual Fairness): {:.4f}'.format(counterfactual_fairness_accepted.mean()))
print('Individual Fairness on Rejected Instances (Counterfactual Fairness): {:.4f}'.format(counterfactual_fairness_rejected.mean()))
