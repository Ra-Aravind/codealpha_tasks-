# =========================
# 📌 Step 1: Import Libraries
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# 📌 Step 2: Load Dataset
# =========================
df = pd.read_csv("Iris.csv")


print(df.head())
print("\nDataset Info:")
print(df.info())

# Drop "Id" column if it exists
if "Id" in df.columns:
    df = df.drop(columns=["Id"])


X = df.drop("Species", axis=1)
y = df["Species"]


# Encode labels (setosa=0, versicolor=1, virginica=2)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print("Unique Classes:", le.classes_)

# =========================
# 📌 Step 3: Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 📌 Step 4: Feature Scaling
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 📌 Step 6: Train Model (KNN)
# =========================

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# =========================
# 📌 Step 6: Model Evaluation
# =========================
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues",
             xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()