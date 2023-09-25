import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
import matplotlib.pyplot as plt
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.operators.mutation.bitflip import BitflipMutation
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTETomek


def compute_statistical_parity(xgboost_y_pred, X_test):
    privileged = xgboost_y_pred[(X_test['statussex_A91'] == 1) | (X_test['statussex_A93'] == 1) | (X_test['statussex_A94'] == 1) & (X_test['age'] > 18)]
    unprivileged = xgboost_y_pred[(X_test['statussex_A92'] == 1) & (X_test['age'] > 18)]
    
    privileged_proportion = np.mean(privileged)
    unprivileged_proportion = np.mean(unprivileged)
    
    statistical_parity_difference = privileged_proportion - unprivileged_proportion
    return statistical_parity_difference

def calculate_equalized_odds(xgboost_y_pred, X_test, privileged_group, unprivileged_group):

    privileged_labels = xgboost_y_pred[(X_test['age'] > 18) & (X_test['statussex_A91'].isin(privileged_group) | X_test['statussex_A93'].isin(privileged_group) | X_test['statussex_A94'].isin(privileged_group))]
    unprivileged_labels = xgboost_y_pred[(X_test['age'] > 18) & (X_test['statussex_A92'].isin(unprivileged_group))]


    privileged_true_labels = y_test[(X_test['age'] > 18) & (X_test['statussex_A91'].isin(privileged_group) | X_test['statussex_A93'].isin(privileged_group) | X_test['statussex_A94'].isin(privileged_group))]
    unprivileged_true_labels = y_test[(X_test['age'] > 18) & (X_test['statussex_A92'].isin(unprivileged_group))]

    privileged_confusion_matrix = confusion_matrix(privileged_true_labels, privileged_labels)
    unprivileged_confusion_matrix = confusion_matrix(unprivileged_true_labels, unprivileged_labels)

    if privileged_confusion_matrix.size > 0:
        privileged_tpr = privileged_confusion_matrix[1, 1] / (privileged_confusion_matrix[1, 0] + privileged_confusion_matrix[1, 1])
        privileged_fpr = privileged_confusion_matrix[0, 1] / (privileged_confusion_matrix[0, 0] + privileged_confusion_matrix[0, 1])
    else:
        privileged_tpr = 0.0
        privileged_fpr = 0.0

    if unprivileged_confusion_matrix.size > 0:
        unprivileged_tpr = unprivileged_confusion_matrix[1, 1] / (unprivileged_confusion_matrix[1, 0] + unprivileged_confusion_matrix[1, 1])
        unprivileged_fpr = unprivileged_confusion_matrix[0, 1] / (unprivileged_confusion_matrix[0, 0] + unprivileged_confusion_matrix[0, 1])
    else:
        unprivileged_tpr = 0.0
        unprivileged_fpr = 0.0

    print(f'Privileged Group - TPR: {privileged_tpr:.4f}, FPR: {privileged_fpr:.4f}')
    print(f'Unprivileged Group - TPR: {unprivileged_tpr:.4f}, FPR: {unprivileged_fpr:.4f}')
    
    tpr_difference = abs(privileged_tpr - unprivileged_tpr)
    fpr_difference = abs(privileged_fpr - unprivileged_fpr)
    
    return tpr_difference + fpr_difference


sensitive_features = ['statussex_A91', 'statussex_A92', 'statussex_A93', 'statussex_A94', 'age']

privileged_group = ['statussex_A91', 'statussex_A93', 'statussex_A94']  # Privileged group: all men above the age of 18
unprivileged_group = ['statussex_A92']  # Unprivileged group: all women above the age of 18

class CreditRiskProblem(Problem):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(n_var=X_train.shape[1],
                         n_obj=2,
                         n_constr=0,
                         xl=0,
                         xu=1)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        param_grid = {
            'max_depth': [2, 4, 8],
            'learning_rate': [0.1, 0.01, 0.001],
            'n_estimators': [50, 100, 200],
            'objective': ['binary:logistic'],
            'seed': [42]
        }

        xgb_classifier = xgb.XGBClassifier()

        grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)


        best_params = grid_search.best_params_
        print("Best Hyperparameters:", best_params)

        self.model = xgb.XGBClassifier( **best_params)


    def _evaluate(self, X, *args, **kwargs):
        out = {}
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print("accuracy\n", acc*100,"%")

        individual_fairness = compute_statistical_parity(y_pred, self.X_test)
        group_fairness = calculate_equalized_odds(y_pred, self.X_test, privileged_group, unprivileged_group)
        print("fairness\n", individual_fairness)

        fairness_metric = -individual_fairness

        # Created a trade-off between accuracy and fairness using a weighted sum
        weight_accuracy = 0.7  
        weight_fairness = 0.2  

        # Combine accuracy and fairness using the weighted sum
        combined_metric = weight_accuracy * acc + weight_fairness * fairness_metric
        
        
        out["F"] = [-combined_metric]  # Minimize the combined metric

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

problem = CreditRiskProblem(X_train_balanced, y_train_balanced, X_test, y_test)
algorithm = NSGA2(pop_size=1000,
                  n_offsprings=50,
                  sampling=BinaryRandomSampling(),
                  crossover=TwoPointCrossover(),
                  mutation=BitflipMutation(),
                  eliminate_duplicates=True)

# Performed Pareto optimization
res = minimize(problem, algorithm, ('n_evals', 1000), ('pop_size', 5000), verbose=True)

print(res.history)
