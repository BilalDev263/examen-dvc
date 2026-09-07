import json
import pickle

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

with open("models/gbr_model.pkl", "rb") as f:
    model = pickle.load(f)

predictions = model.predict(X_test)

scores = {
    "mse": mean_squared_error(y_test, predictions),
    "mae": mean_absolute_error(y_test, predictions),
    "r2": r2_score(y_test, predictions),
}

print(scores)

with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=4)

df = pd.DataFrame({"prediction": predictions})
df.to_csv("data/prediction.csv", index=False)
