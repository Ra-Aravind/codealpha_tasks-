# =========================
# Car Price Prediction (Regression)
# Dataset: car data.csv
# =========================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Optional: install xgboost if you have it; otherwise script skips it gracefully
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

RANDOM_STATE = 42

# -------------------------
# 1) Load & quick inspect
# -------------------------
csv_path = "car data.csv"  # adjust if needed
df = pd.read_csv(csv_path)

print("Head:\n", df.head(), "\n")
print("Info:")
print(df.info(), "\n")
print("Missing values:\n", df.isnull().sum(), "\n")

print(df.columns)

# -------------------------
# 2) Basic cleaning & feature engineering
# -------------------------
# Drop rows with critical nulls (dataset typically has none)
# Note: column names in your file: Driven_kms etc — adjust if different
df = df.dropna(subset=["Selling_Price", "Year", "Present_Price", "Driven_kms"])

# Car age instead of raw year
current_year = pd.Timestamp.today().year
df["Car_Age"] = current_year - df["Year"]

# Some versions have 'Owner' as numeric already; ensure int
if "Owner" in df.columns:
    df["Owner"] = pd.to_numeric(df["Owner"], errors="coerce").fillna(0).astype(int)

# Remove obvious outliers if desired (optional)
df = df[df["Present_Price"] < df["Present_Price"].quantile(0.995)]
df = df[df["Driven_kms"] < df["Driven_kms"].quantile(0.995)]
df = df[df["Selling_Price"] > 0]

# Drop Car_Name (high-cardinality text rarely helpful without NLP)
if "Car_Name" in df.columns:
    df = df.drop(columns=["Car_Name"])

# -------------------------
# 3) Features/target split
# -------------------------
target = "Selling_Price"
y = df[target]

# Choose features (exclude Year because we use Car_Age)
feature_cols = [c for c in df.columns if c != target and c != "Year"]
X = df[feature_cols]

# Identify categorical and numeric columns
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

print("Categorical columns:", cat_cols)
print("Numeric columns:", num_cols, "\n")

# -------------------------
# 4) Preprocess (OHE + Scaling numeric)
# -------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# -------------------------
# 5) Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# -------------------------
# 6) Define models
# -------------------------
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "RandomForest": RandomForestRegressor(
        n_estimators=400, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    )
}
if HAS_XGB:
    models["XGBRegressor"] = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="reg:squarederror"
    )

# -------------------------
# 7) Train, evaluate, compare
# -------------------------
def evaluate(true, pred, name=""):
    r2 = r2_score(true, pred)
    mae = mean_absolute_error(true, pred)
    # Compute RMSE in a version-compatible way (avoid `squared=` param)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    print(f"{name:>15} | R2: {r2:.3f} | MAE: {mae:.3f} | RMSE: {rmse:.3f}")
    return {"model": name, "r2": r2, "mae": mae, "rmse": rmse}

results = []

for name, base_model in models.items():
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", base_model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    metrics = evaluate(y_test, preds, name)
    metrics["pipeline"] = pipe
    results.append(metrics)

# Pick the best by R2
best = sorted(results, key=lambda d: d["r2"], reverse=True)[0]
best_model = best["pipeline"]
print("\nBest model:", best["model"])

# -------------------------
# 8) Cross-validation on best model (quick)
# -------------------------
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="r2", n_jobs=-1)
print("CV R2 scores:", np.round(cv_scores, 3))
print("CV R2 mean:", cv_scores.mean().round(3), "±", cv_scores.std().round(3))

# -------------------------
# 9) Visual checks
# -------------------------
plt.figure(figsize=(7,6))
y_pred_best = best_model.predict(X_test)
plt.scatter(y_test, y_pred_best, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], lw=2)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title(f"Predicted vs Actual ({best['model']})")
plt.tight_layout()
plt.show()

# Feature importance (tree-based models only)
def plot_feature_importance(pipeline, feature_names):
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    num = pipeline.named_steps["preprocess"].named_transformers_["num"]
    new_cat_names = []
    if hasattr(ohe, "get_feature_names_out"):
        new_cat_names = list(ohe.get_feature_names_out(cat_cols))
    processed_feature_names = list(num_cols) + new_cat_names

    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        k = min(len(importances), len(processed_feature_names))
        fi = pd.Series(importances[:k], index=processed_feature_names[:k]).sort_values(ascending=False).head(20)
        plt.figure(figsize=(8,6))
        fi.iloc[::-1].plot(kind="barh")
        plt.xlabel("Importance")
        plt.title(f"Top Feature Importances ({type(model).__name__})")
        plt.tight_layout()
        plt.show()
    else:
        print("Feature importances not available for this model.")

if best["model"] in ["RandomForest", "DecisionTree", "XGBRegressor"]:
    plot_feature_importance(best_model, X.columns)

# -------------------------
# 10) Save the best model
# -------------------------
from joblib import dump
os.makedirs("artifacts", exist_ok=True)
model_path = os.path.join("artifacts", f"best_car_price_model_{best['model']}.joblib")
dump(best_model, model_path)
print("Saved best model to:", model_path)

# -------------------------
# 11) Example: make a single prediction
# -------------------------
example = pd.DataFrame([{
    "Present_Price": 7.5,
    "Driven_kms": 35000,
    "Owner": 0,
    "Car_Age": 5,
    "Fuel_Type": "Petrol",
    "Selling_type": "Dealer",
    "Transmission": "Manual"
}])

for col in X.columns:
    if col not in example.columns:
        if col in cat_cols:
            example[col] = X[col].mode()[0]
        else:
            example[col] = X[col].median()

example = example[X.columns]
pred_price = best_model.predict(example)[0]
print(f"\nExample predicted selling price: {pred_price:.2f} (same unit as dataset)")
