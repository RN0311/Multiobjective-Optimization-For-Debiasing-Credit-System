import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.combine import SMOTETomek

# Load and preprocess the data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                 header=None, sep=' ')

feature_names = ['status', 'duration', 'credit_history', 'purpose', 'amount',
                 'savings', 'employment_duration', 'installment_rate', 'statussex',
                 'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
                 'housing', 'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
                 'credit_risk']

df.columns = feature_names

# Drop unnecessary columns for this example
df = df.drop(['age', 'statussex'], axis=1)

# Perform one-hot encoding for categorical columns
categorical_cols = ['status', 'credit_history', 'purpose', 'savings', 'employment_duration',
                    'other_debtors', 'property', 'other_installment_plans',
                    'housing', 'job', 'telephone', 'foreign_worker']

df = pd.get_dummies(df, columns=categorical_cols)
df.credit_risk.replace([1, 2], [1, 0], inplace=True)

# Defined target and sensitive features
target_col = 'credit_risk'
sensitive_features = ['statussex', 'age']

X = df.drop([target_col], axis=1)
y = df[target_col]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applied SMOTE-Tomek to balance the training data
smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

# Defined the XGBoost model
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'seed': 42
}

# Train the XGBoost model
xgb = XGBClassifier(**params)
xgb.fit(X_train_balanced, y_train_balanced)

# Predict probabilities for the test set
y_prob = xgb.predict_proba(X_test)[:, 1]

# Set the alpha (rejection rate) based on your fairness requirements
alpha = 0.65

# Sort the probabilities to find the rejection threshold
y_prob_sorted = np.sort(y_prob)
reject_threshold = y_prob_sorted[int(alpha * len(y_prob))]

# Create masks for accepted and rejected instances
accepted_mask = (y_prob >= reject_threshold)
rejected_mask = (y_prob < reject_threshold)

# Calculated accuracy for accepted and rejected instances
accuracy_accepted = accuracy_score(y_test[accepted_mask], xgb.predict(X_test[accepted_mask]))
accuracy_rejected = accuracy_score(y_test[rejected_mask], xgb.predict(X_test[rejected_mask]))

# Calculated fairness (Statistical Parity) for accepted and rejected instances
fairness_accepted = (y_test[accepted_mask] == 1).mean() - (xgb.predict(X_test[accepted_mask]) == 1).mean()
fairness_rejected = (y_test[rejected_mask] == 1).mean() - (xgb.predict(X_test[rejected_mask]) == 1).mean()

print('Accuracy on Accepted Instances: {:.4f}'.format(accuracy_accepted))
print('Fairness on Accepted Instances (Statistical Parity): {:.4f}'.format(fairness_accepted))
