import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.metrics import precision_score, recall_score, f1_score
from pygad import GA  

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

sensitive_features = ['sex_Male']

def compute_statistical_parity(xgboost_y_pred, X_test):
    privileged = xgboost_y_pred[(X_test['sex_Male'] == 1)]
    unprivileged = xgboost_y_pred[(X_test['sex_Male'] == 0)]

    privileged_proportion = np.mean(privileged)
    unprivileged_proportion = np.mean(unprivileged)

    statistical_parity_difference = privileged_proportion - unprivileged_proportion

    return statistical_parity_difference

def compute_counterfactual_fairness(xgboost_model, X_test):
    original_predictions = xgboost_model.predict(X_test)

    counterfactual_X_test = X_test.copy()
    counterfactual_X_test['sex_Male'] = 1 - counterfactual_X_test['sex_Male']

    original_probabilities = xgboost_model.predict_proba(X_test)[:, 1]
    counterfactual_probabilities = xgboost_model.predict_proba(counterfactual_X_test)[:, 1]

    counterfactual_fairness_original = np.abs(original_probabilities - counterfactual_probabilities)

    return counterfactual_fairness_original
  
  def fitness_function(ga_instance, solution, solution_idx):
    learning_rate, max_depth, n_estimators = solution

    params['learning_rate'] = learning_rate
    params['max_depth'] = int(max_depth)
    params['n_estimators'] = int(n_estimators)

    model = XGBClassifier(**params)
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    statistical_parity = compute_statistical_parity(y_pred, X_test)

    weight_accuracy = 0.7
    weight_fairness = 0.3

    fairness_mean = np.mean(statistical_parity)
    fitness_value = -(weight_accuracy * accuracy) + (weight_fairness * fairness_mean)

    return fitness_value
  
  import pygad


gene_space = [
    {'low': 0.001, 'high': 0.5},   # learning_rate
    {'low': 1, 'high': 10},        # max_depth
    {'low': 10, 'high': 200}       # n_estimators
]

ga_instance = pygad.GA(
    num_generations=10,
    num_parents_mating=5,
    fitness_func=fitness_function,
    sol_per_pop=10,
    num_genes=3,
    gene_space=gene_space,
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=10
)

ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution:", solution)
print("Best solution fitness:", solution_fitness)


params['learning_rate'] = best_learning_rate
params['max_depth'] = int(best_max_depth)
params['n_estimators'] = int(best_n_estimators)

model = XGBClassifier(**params)
model.fit(X_train_balanced, y_train_balanced)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
individual_fairness = compute_statistical_parity(y_pred, X_test)
counterfactual_fairness_original = compute_counterfactual_fairness(model, X_test)

print("Best Accuracy: {:.2f}%".format(accuracy * 100))
print("Best Statistical Parity (Group Fairness):", individual_fairness)
print("Best Counterfactual Fairness (Individual Fairness):", np.mean(counterfactual_fairness_original))
