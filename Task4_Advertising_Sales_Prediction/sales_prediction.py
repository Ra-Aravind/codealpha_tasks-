import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------
# Load dataset
# --------------------
df = pd.read_csv("Advertising.csv")

# 1. Missing values
print("Missing values:\n", df.isnull().sum(), "\n")

# 2. Data description
print("Data description:\n", df.describe(), "\n")

# 3. Correlation with Sales
correlations = df.corr()["Sales"].sort_values(ascending=False)
print("Correlation with Sales:\n", correlations, "\n")

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --------------------
# Feature & Target
# --------------------
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------
# Model Comparison
# --------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
    "Support Vector Regressor": SVR()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse, "Model": model}

# Show comparison
results_df = pd.DataFrame(results).T.sort_values(by="R2", ascending=False)
print("Model comparison:\n", results_df, "\n")

# --------------------
# Best Model
# --------------------
best_model_name = results_df.index[0]
best_model = results[best_model_name]["Model"]
print(f"Best Model: {best_model_name}\n")

# Predict with best model
y_pred_best = best_model.predict(X_test)

# Plot Actual vs Predicted for Best Model
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred_best, alpha=0.7, color="green")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.title(f"Actual vs Predicted Sales ({best_model_name})")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.show()
