import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTETomek

# Load and preprocess the Adult Income dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']

df = pd.read_csv(url, header=None, names=columns, na_values=' ?')

# Drop missing values
df.dropna(inplace=True)

# Convert categorical columns to one-hot encoding
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'native-country']
df = pd.get_dummies(df, columns=categorical_cols)

# Encode the target variable
df['income'] = df['income'].apply(lambda x: 1 if x == ' >50K' else 0)

target_col = 'income'
sensitive_features = ['race', 'sex']

X = df.drop([target_col], axis=1)
y = df[target_col]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTETomek for balancing
start_time = time.time()
smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
balance_time = time.time() - start_time
print(f"Balancing Time: {balance_time:.2f} seconds")

# Create a counterfactual test set by flipping the 'sex' attribute
counterfactual_X_test = X_test.copy()
if 'sex_ Male' in counterfactual_X_test.columns:
    counterfactual_X_test['sex_ Male'] = 1 - counterfactual_X_test['sex_ Male']

# Define model parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'seed': 42
}

# Train the model
start_time = time.time()
xgb = XGBClassifier(**params)
xgb.fit(X_train_balanced, y_train_balanced)
train_time = time.time() - start_time
print(f"Training Time: {train_time:.2f} seconds")

# Make predictions
start_time = time.time()
y_prob = xgb.predict_proba(X_test)[:, 1]
predict_time = time.time() - start_time
print(f"Prediction Time: {predict_time:.2f} seconds")

# Reject Option Classification
start_time = time.time()
alpha = 0.65
y_prob_sorted = np.sort(y_prob)
reject_threshold = y_prob_sorted[int(alpha * len(y_prob))]

accepted_mask = (y_prob >= reject_threshold)
rejected_mask = (y_prob < reject_threshold)

accuracy_accepted = accuracy_score(y_test[accepted_mask], xgb.predict(X_test[accepted_mask]))
accuracy_rejected = accuracy_score(y_test[rejected_mask], xgb.predict(X_test[rejected_mask]))

# Calculate fairness (Statistical Parity)
group_fairness_accepted = (y_test[accepted_mask] == 1).mean() - (xgb.predict(X_test[accepted_mask]) == 1).mean()
group_fairness_rejected = (y_test[rejected_mask] == 1).mean() - (xgb.predict(X_test[rejected_mask]) == 1).mean()
roc_time = time.time() - start_time
print(f"Group Fairness Calculation Time: {roc_time:.2f} seconds")

print('Accuracy on Accepted Instances: {:.4f}'.format(accuracy_accepted))
print('Accuracy on Rejected Instances: {:.4f}'.format(accuracy_rejected))
print('Group Fairness on Accepted Instances (Statistical Parity): {:.4f}'.format(group_fairness_accepted))
print('Group Fairness on Rejected Instances (Statistical Parity): {:.4f}'.format(group_fairness_rejected))

# Counterfactual Fairness
start_time = time.time()
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
cf_time = time.time() - start_time
print(f"Individual Fairness Calculation Time: {cf_time:.2f} seconds")

accuracy_accepted = accuracy_score(y_test[accepted_mask], xgb.predict(X_test[accepted_mask]))
accuracy_rejected = accuracy_score(y_test[rejected_mask], xgb.predict(X_test[rejected_mask]))

print('Individual Fairness on Accepted Instances (Counterfactual Fairness): {:.4f}'.format(counterfactual_fairness_accepted.mean()))
print('Individual Fairness on Rejected Instances (Counterfactual Fairness): {:.4f}'.format(counterfactual_fairness_rejected.mean()))
