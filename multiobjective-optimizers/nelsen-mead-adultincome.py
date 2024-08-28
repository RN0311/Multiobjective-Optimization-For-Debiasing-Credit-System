import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTETomek
from scipy.optimize import minimize


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv(url, names=column_names, na_values=' ?', skipinitialspace=True)

df = df.dropna()


df = df.drop(columns=['age', 'education', 'education-num', 'marital-status', 'occupation'])


features = ['workclass', 'relationship', 'race', 'sex', 'hours-per-week', 'native-country']
target = 'income'


df = pd.get_dummies(df, columns=features)


df[target] = df[target].apply(lambda x: 1 if x == '>50K' else 0)

X = df.drop([target], axis=1)
y = df[target]


print(X.columns)  
sensitive_feature = 'relationship_Male' 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

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

    fitness = 0.7 * accuracy - 0.3 * np.abs(statistical_parity)

    return -fitness  # Negate the fitness to convert maximization to minimization


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

print(f"Final Accuracy: {accuracy}")
print(f"Final Statistical Parity: {statistical_parity}")
