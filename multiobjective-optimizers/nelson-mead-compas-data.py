import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from scipy.optimize import minimize


url = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
df = pd.read_csv(url)


df = df[(df.days_b_screening_arrest <= 30) &
        (df.days_b_screening_arrest >= -30) &
        (df.is_recid != -1) &
        (df.c_charge_degree != 'O') &
        (df.score_text != 'N/A')]

df['length_of_stay'] = (pd.to_datetime(df['c_jail_out']) - pd.to_datetime(df['c_jail_in'])).dt.days

features = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 'length_of_stay']
target = 'two_year_recid'

df = df[features + [target]]

categorical_cols = ['c_charge_degree', 'race', 'age_cat', 'score_text', 'sex']
df = pd.get_dummies(df, columns=categorical_cols)

df[target] = df[target].astype(int)
df = df.fillna(df.median())

X = df.drop([target], axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

sensitive_feature = 'sex_Male'

def fitness_function(params):
    learning_rate, max_depth, n_estimators = params

    model = XGBClassifier(
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        n_estimators=int(n_estimators),
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist',
        seed=42
    )


    model.fit(X_train_balanced, y_train_balanced)


    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)


    privileged = y_pred[(X_test[sensitive_feature] == 1)]
    unprivileged = y_pred[(X_test[sensitive_feature] == 0)]

    privileged_proportion = np.mean(privileged)
    unprivileged_proportion = np.mean(unprivileged)

    statistical_parity = privileged_proportion - unprivileged_proportion


    def compute_counterfactual_fairness(xgboost_model, X_test):
        original_probabilities = xgboost_model.predict_proba(X_test)[:, 1]

        counterfactual_X_test = X_test.copy()
        counterfactual_X_test[sensitive_feature] = 1 - counterfactual_X_test[sensitive_feature]

        counterfactual_probabilities = xgboost_model.predict_proba(counterfactual_X_test)[:, 1]

        counterfactual_fairness = np.abs(original_probabilities - counterfactual_probabilities)
        return np.mean(counterfactual_fairness)

    counterfactual_fairness = compute_counterfactual_fairness(model, X_test)

    
    fitness = 0.6 * accuracy - 0.2 * np.abs(statistical_parity) - 0.2 * counterfactual_fairness

    return -fitness 


def bounds_transform(params):
    learning_rate = np.clip(params[0], 0.001, 0.5)
    max_depth = int(np.clip(params[1], 1, 10))
    n_estimators = int(np.clip(params[2], 10, 200))
    return [learning_rate, max_depth, n_estimators]


initial_guess = [0.1, 5, 50]


result = minimize(
    lambda params: fitness_function(bounds_transform(params)),
    initial_guess,
    method='Nelder-Mead',
    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 50}
)

best_solution = result.x
best_learning_rate, best_max_depth, best_n_estimators = bounds_transform(best_solution)

print(f"Best Parameters: learning_rate={best_learning_rate}, max_depth={best_max_depth}, n_estimators={best_n_estimators}")
print(f"Best Fitness: {-result.fun}")


model = XGBClassifier(
    learning_rate=best_learning_rate,
    max_depth=best_max_depth,
    n_estimators=best_n_estimators,
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    seed=42
)

model.fit(X_train_balanced, y_train_balanced)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
privileged = y_pred[(X_test[sensitive_feature] == 1)]
unprivileged = y_pred[(X_test[sensitive_feature] == 0)]
statistical_parity = np.mean(privileged) - np.mean(unprivileged)

counterfactual_fairness = compute_counterfactual_fairness(model, X_test)

print(f"Final Accuracy: {accuracy}")
print(f"Final Statistical Parity: {statistical_parity}")
print(f"Final Counterfactual Fairness: {counterfactual_fairness}")
