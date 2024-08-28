from imblearn.combine import SMOTETomek
from scipy.optimize import minimize
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                 header=None, sep=' ')


feature_names = ['status', 'duration', 'credit_history', 'purpose', 'amount',
                 'savings', 'employment_duration', 'installment_rate', 'statussex',
                 'other_debtors', 'residence_since', 'property',
                 'age', 'other_installment_plans', 'housing', 'number_credits',
                 'job', 'people_liable', 'telephone', 'foreign_worker',
                 'credit_risk']

df.columns = feature_names

categorical_cols = ['status', 'credit_history', 'purpose', 'savings', 'employment_duration',
                    'other_debtors', 'property', 'other_installment_plans',
                    'housing', 'job', 'telephone', 'statussex', 'foreign_worker']

df = pd.get_dummies(df, columns=categorical_cols)
df.credit_risk.replace([1, 2], [1, 0], inplace=True)

target_col = 'credit_risk'
sensitive_features = ['statussex_A92', 'age']

X = df.drop([target_col], axis=1)
y = df[target_col]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)


def compute_statistical_parity(y_pred, X_test):
    privileged = y_pred[(X_test['statussex_A92'] == 1)]
    unprivileged = y_pred[(X_test['statussex_A92'] == 0)]

    privileged_proportion = np.mean(privileged)
    unprivileged_proportion = np.mean(unprivileged)

    statistical_parity_difference = privileged_proportion - unprivileged_proportion

    return statistical_parity_difference

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
    statistical_parity = compute_statistical_parity(y_pred, X_test)


    weight_accuracy = 0.7
    weight_fairness = 0.3

    fitness_value = -(weight_accuracy * accuracy) + (weight_fairness * statistical_parity)
    return fitness_value

bounds = {
    'learning_rate': (0.001, 0.5),
    'max_depth': (1, 10),
    'n_estimators': (10, 200)
}


initial_guess = [0.1, 5, 50]


def bounded_fitness(params):
    learning_rate = np.clip(params[0], bounds['learning_rate'][0], bounds['learning_rate'][1])
    max_depth = int(np.clip(params[1], bounds['max_depth'][0], bounds['max_depth'][1]))
    n_estimators = int(np.clip(params[2], bounds['n_estimators'][0], bounds['n_estimators'][1]))

    return fitness_function([learning_rate, max_depth, n_estimators])

result = minimize(
    bounded_fitness,
    initial_guess,
    method='Nelder-Mead',
    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 50}
)


best_solution = result.x
best_learning_rate, best_max_depth, best_n_estimators = best_solution

print("Best Solution Parameters: Learning Rate: {:.4f}, Max Depth: {:.0f}, Estimators: {:.0f}".format(
    best_learning_rate, best_max_depth, best_n_estimators))
print("Best Objective Value: {:.4f}".format(bounded_fitness(best_solution)))

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

def compute_counterfactual_fairness(xgboost_model, X_test):
    original_predictions = xgboost_model.predict(X_test)

    counterfactual_X_test = X_test.copy()
    counterfactual_X_test['statussex_A92'] = 1 - counterfactual_X_test['statussex_A92']

    original_probabilities = xgboost_model.predict_proba(X_test)[:, 1]
    counterfactual_probabilities = xgboost_model.predict_proba(counterfactual_X_test)[:, 1]

    counterfactual_fairness_original = np.abs(original_probabilities - counterfactual_probabilities)

    return counterfactual_fairness_original
  
counterfactual_fairness_original = compute_counterfactual_fairness(model, X_test)

print("Best Accuracy: {:.2f}%".format(accuracy * 100))
print("Best Statistical Parity (Group Fairness):", individual_fairness)
print("Best Counterfactual Fairness (Individual Fairness):", np.mean(counterfactual_fairness_original))
