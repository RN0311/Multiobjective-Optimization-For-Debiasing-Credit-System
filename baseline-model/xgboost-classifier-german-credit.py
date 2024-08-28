
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, classification_report, f1_score
import xgboost as xgb
from imblearn.combine import SMOTETomek


data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                   header=None, sep=' ')

feature_names = ['status', 'duration', 'credit_history', 'purpose', 'amount',
                 'savings', 'employment_duration', 'installment_rate', 'statussex',
                 'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
                 'housing', 'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
                 'credit_risk']

data.columns = feature_names

categorical_cols = ['status', 'credit_history', 'purpose', 'savings', 'employment_duration',
                    'statussex', 'other_debtors', 'property', 'other_installment_plans',
                    'housing', 'job', 'telephone', 'foreign_worker']

data = pd.get_dummies(data, columns=categorical_cols)
data.credit_risk.replace([1,2], [1,0], inplace=True)

target_col = 'credit_risk'
features = data.drop(target_col, axis=1)
target = data[target_col]
sensitive_features = ['statussex', 'age']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)


params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'seed': 42
}

xg_model = xgb.XGBClassifier(**params)

xg_model.fit(X_train_balanced, y_train_balanced)

xgboost_y_pred = xg_model.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, xgboost_y_pred):.4f}')
print(classification_report(y_test, xgboost_y_pred))

precision = precision_score(y_test, xgboost_y_pred)
recall = recall_score(y_test, xgboost_y_pred)
f1 = f1_score(y_test, xgboost_y_pred)
accuracy = accuracy_score(y_test, xgboost_y_pred)

xgb_metrics_dict = {'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'Accuracy': accuracy}

xgb_metrics_df = pd.DataFrame.from_dict(xgb_metrics_dict, orient='index', columns=['Score'])

print(xgb_metrics_df)
