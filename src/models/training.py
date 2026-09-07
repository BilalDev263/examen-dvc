import pickle

import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor

params = yaml.safe_load(open("params.yaml"))["gridsearch"]

X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

model = GradientBoostingRegressor(random_state=params["random_state"], **best_params)
model.fit(X_train, y_train.values.ravel())

with open("models/gbr_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modele entraine")
