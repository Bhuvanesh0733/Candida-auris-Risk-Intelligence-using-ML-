import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

csv_path = "c_auris_parameter_dataset.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError("Dataset not found")

df = pd.read_csv(csv_path)

PU = 200
PC = 30
PI = 10
HC = 15
HU = 85

def c_auris_dynamics(row):
    Lambda = row["Lambda"]
    beta1 = row["beta1"]
    beta2 = row["beta2"]
    d1 = row["d1"]
    d2 = row["d2"]
    lam = row["lambda"]
    inv_sigma = row["1_over_sigma"]
    phi = row["phi"]
    mu = row["mu"]
    sigma = 1 / inv_sigma
    dPU_dt = Lambda - beta2 * HC * PU - d1 * PU + phi * (PC + PI)
    dPC_dt = beta2 * HC * PU - (d2 + phi + sigma) * PC
    dPI_dt = sigma * PC - (phi + mu) * PI
    dHC_dt = beta1 * PC * HU - lam * HC
    return pd.Series([dPU_dt, dPC_dt, dPI_dt, dHC_dt])

df[["dPU_dt", "dPC_dt", "dPI_dt", "dHC_dt"]] = df.apply(c_auris_dynamics, axis=1)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.mean(numeric_only=True))

features = [
    "Lambda", "beta1", "beta2",
    "d1", "d2", "lambda",
    "1_over_sigma", "phi", "mu",
    "dPU_dt", "dPC_dt", "dHC_dt"
]

X = df[features]
y = df["dPI_dt"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

joblib.dump(model, "candida_auris_dynamics_model.pkl")

print("Model saved as candida_auris_dynamics_model.pkl")
