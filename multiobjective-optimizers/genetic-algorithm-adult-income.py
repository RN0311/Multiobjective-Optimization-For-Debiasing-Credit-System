import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTETomek
import pygad

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

def fitness_function(ga_instance, solution, solution_idx):
    learning_rate, max_depth, n_estimators = solution


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

    return fitness

ga_instance = pygad.GA(
    num_generations=10,
    num_parents_mating=5,
    fitness_func=fitness_function,
    sol_per_pop=10,
    num_genes=3,
    gene_type=float,
    init_range_low=[0.001, 1, 10],  
    init_range_high=[0.5, 10, 200],
    mutation_percent_genes=10,
    crossover_type="single_point",
    mutation_type="random",
    mutation_by_replacement=True,
    random_mutation_min_val=[0.001, 1, 10],
    random_mutation_max_val=[0.5, 10, 200],
    stop_criteria=["saturate_100"]
)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
best_learning_rate, best_max_depth, best_n_estimators = solution

print(f"Best Parameters: learning_rate={best_learning_rate}, max_depth={int(best_max_depth)}, n_estimators={int(best_n_estimators)}")
print(f"Best Fitness: {solution_fitness}")

model = XGBClassifier(
    learning_rate=best_learning_rate,
    max_depth=int(best_max_depth),
    n_estimators=int(best_n_estimators),
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

def compute_counterfactual_fairness(xgboost_model, X_test):
    original_predictions = xgboost_model.predict(X_test)

    counterfactual_X_test = X_test.copy()
    counterfactual_X_test['relationship_Male'] = 1 - counterfactual_X_test['relationship_Male']

    original_probabilities = xgboost_model.predict_proba(X_test)[:, 1]
    counterfactual_probabilities = xgboost_model.predict_proba(counterfactual_X_test)[:, 1]

    counterfactual_fairness_original = np.abs(original_probabilities - counterfactual_probabilities)

    return counterfactual_fairness_original
  
counterfactual_fairness_original = compute_counterfactual_fairness(model, X_test)

print("Best Accuracy: {:.2f}%".format(accuracy * 100))
print("Best Statistical Parity (Group Fairness):", individual_fairness)
print("Best Counterfactual Fairness (Individual Fairness):", np.mean(counterfactual_fairness_original))
