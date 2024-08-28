import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTETomek

# Load the COMPAS dataset
url = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
df = pd.read_csv(url)

# Data preprocessing
df = df[(df.days_b_screening_arrest <= 30) &
        (df.days_b_screening_arrest >= -30) &
        (df.is_recid != -1) &
        (df.c_charge_degree != 'O') &
        (df.score_text != 'N/A')]

df['length_of_stay'] = (pd.to_datetime(df['c_jail_out']) - pd.to_datetime(df['c_jail_in'])).dt.days

# Select features and target
features = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 'length_of_stay']
target = 'two_year_recid'

df = df[features + [target]]

# One-hot encode categorical columns
categorical_cols = ['c_charge_degree', 'race', 'age_cat', 'score_text', 'sex']
df = pd.get_dummies(df, columns=categorical_cols)

# Encode target variable
df[target] = df[target].astype(int)

# Define features and target
X = df.drop([target], axis=1)
y = df[target]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balance the dataset using SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

# Counterfactual Test: Flipping the protected attribute 'race'
counterfactual_X_test = X_test.copy()
# Assuming 'race_African-American' is the protected attribute to flip
if 'race_African-American' in counterfactual_X_test.columns:
    counterfactual_X_test['race_African-American'] = 1 - counterfactual_X_test['race_African-American']

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'seed': 42
}

xgb_model = XGBClassifier(**params)
xgb_model.fit(X_train_balanced, y_train_balanced)

y_prob = xgb_model.predict_proba(X_test)[:, 1]

alpha = 0.65
y_prob_sorted = np.sort(y_prob)
reject_threshold = y_prob_sorted[int(alpha * len(y_prob))]

accepted_mask = (y_prob >= reject_threshold)
rejected_mask = (y_prob < reject_threshold)

accuracy_accepted = accuracy_score(y_test[accepted_mask], xgb_model.predict(X_test[accepted_mask]))
accuracy_rejected = accuracy_score(y_test[rejected_mask], xgb_model.predict(X_test[rejected_mask]))

# Calculate fairness (Statistical Parity) for accepted and rejected instances
group_fairness_accepted = (y_test[accepted_mask] == 1).mean() - (xgb_model.predict(X_test[accepted_mask]) == 1).mean()
group_fairness_rejected = (y_test[rejected_mask] == 1).mean() - (xgb_model.predict(X_test[rejected_mask]) == 1).mean()

print('Accuracy on Accepted Instances: {:.4f}'.format(accuracy_accepted))
print('Accuracy on Rejected Instances: {:.4f}'.format(accuracy_rejected))
print('Group Fairness on Accepted Instances (Statistical Parity): {:.4f}'.format(group_fairness_accepted))
print('Group Fairness on Rejected Instances (Statistical Parity): {:.4f}'.format(group_fairness_rejected))

# Counterfactual fairness
y_prob_counterfactual = xgb_model.predict_proba(counterfactual_X_test)[:, 1]
counterfactual_fairness_original = np.abs(y_prob - y_prob_counterfactual)

alpha = 0.50
counterfactual_fairness_sorted = np.sort(counterfactual_fairness_original)
reject_threshold = counterfactual_fairness_sorted[int(alpha * len(counterfactual_fairness_original))]

accepted_mask = (counterfactual_fairness_original >= reject_threshold)
rejected_mask = (counterfactual_fairness_original < reject_threshold)

counterfactual_fairness_accepted = counterfactual_fairness_original[accepted_mask]
counterfactual_fairness_rejected = counterfactual_fairness_original[rejected_mask]

accuracy_accepted = accuracy_score(y_test[accepted_mask], xgb_model.predict(X_test[accepted_mask]))
accuracy_rejected = accuracy_score(y_test[rejected_mask], xgb_model.predict(X_test[rejected_mask]))

print('Individual Fairness on Accepted Instances (Counterfactual Fairness): {:.4f}'.format(counterfactual_fairness_accepted.mean()))
print('Individual Fairness on Rejected Instances (Counterfactual Fairness): {:.4f}'.format(counterfactual_fairness_rejected.mean()))
