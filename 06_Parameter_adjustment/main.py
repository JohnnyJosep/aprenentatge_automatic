import pandas as pd

# Llibreries que necessitarem
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

random_value = 33
# Carrega de dades i preparació de les dades emprant Pandas
data = pd.read_csv("data/day.csv")
datos = pd.DataFrame(data.iloc[:, 4:13])  # Seleccionam totes les files i les columnes per index
valors = pd.DataFrame(data.iloc[:, -1])  # Seleccionam totes les files i la columna objectiu

X = datos.to_numpy()
y = valors.to_numpy().ravel()
features_names = datos.columns


# 1 train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_value)


# idea de Random Forest Regressor
basic_idea_estimator = RandomForestRegressor(n_estimators=10, random_state=random_value)
basic_idea_estimator.fit(X_train, y_train)
basic_idea_predictions = basic_idea_estimator.predict(X_test)
basic_idea_errors = abs(basic_idea_predictions - y_test)
basic_idea_accuracy = 100 - (100 * np.mean(basic_idea_errors / y_test))
print('average error', np.mean(basic_idea_errors))
print('accuracy', basic_idea_accuracy)

# 2 params
params = {
    'n_estimators': [50, 100, 200, 1000],
    'max_depth': [None, 2, 10, 90, 100, 150],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'criterion': ['squared_error', 'absolute_error', 'poisson'],
}

# 3 grid search
gs = GridSearchCV(RandomForestRegressor(bootstrap=True, random_state=random_value), params, cv=None, verbose=3)
gs.fit(X_train, y_train)

# 4 use best estimator
print(gs.best_params_)

rfr = gs.best_estimator_
rfr.fit(X_train, y_train)

# 5 predict
predictions = rfr.predict(X_test)

errors = abs(predictions - y_test)
accuracy = 100 - (100 * np.mean(errors / y_test))
print('average error', np.mean(errors))
print('accuracy', accuracy)

# Output:
# {'criterion': 'squared_error', 'max_depth': 10, 'max_features': 3, 'min_samples_leaf': 3, 'min_samples_split': 10, 'n_estimators': 100}
# average error 1104.5968710719544
# accuracy 65.53609437303857

