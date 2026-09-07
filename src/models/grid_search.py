import pickle

import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

params = yaml.safe_load(open("params.yaml"))["gridsearch"]

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")

grid = {
    "n_estimators": params["n_estimators"],
    "learning_rate": params["learning_rate"],
    "max_depth": params["max_depth"],
}

model = GradientBoostingRegressor(random_state=params["random_state"])
grid_search = GridSearchCV(model, grid, cv=params["cv"], scoring="r2")
grid_search.fit(X_train, y_train.values.ravel())

print(grid_search.best_params_)

with open("models/best_params.pkl", "wb") as f:
    pickle.dump(grid_search.best_params_, f)
