# =========================
# 📌 Step 1: Import Libraries
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")
# =========================
# 📌 Step 2: Load Dataset
# =========================

df = pd.read_csv("Unemployment in India.csv")

print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())


# =========================
# 📌 Step 3: Data Cleaning
# =========================

# Standardize column names
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# Rename properly
df.rename(columns={
    "estimated_unemployment_rate_(%)": "unemployment_rate",
    "estimated_employed": "employed",
    "estimated_labour_participation_rate_(%)": "labour_participation_rate",
    "date": "date",
    "region": "region"
}, inplace=True)

# Convert date column to datetime if available
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
df.head()

# =========================
# 📌 Step 4: Data Exploration
# =========================

print("\nSummary Statistics:\n", df.describe())

# Unique states/regions
if "region" in df.columns:
    print("\nRegions:", df["region"].unique())


# =========================
# 📌 Step 5: Unemployment Trends (Line Plot)
# =========================

if "date" in df.columns and "unemployment_rate" in df.columns:
    plt.figure(figsize=(12,6))
    sns.lineplot(x="date", y="unemployment_rate", data=df, hue="region")
    plt.title("Unemployment Rate Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")
    plt.legend(title="Region", bbox_to_anchor=(1.05,1))
    plt.show()

# =========================
# 📌 Step 6: Impact of Covid-19
# =========================

# Assuming Covid-19 impact started in early 2020
if "date" in df.columns:
    df["year"] = df["date"].dt.year
    plt.figure(figsize=(10,5))
    sns.boxplot(x="year", y="unemployment_rate", data=df)
    plt.title("Covid-19 Impact on Unemployment Rates (Before vs After 2020)")
    plt.xlabel("Year")
    plt.ylabel("Unemployment Rate (%)")
    plt.show()

# =========================
# 📌 Step 7: Seasonal Trends
# =========================
if "date" in df.columns:
    df["month"] = df["date"].dt.month
    monthly_avg = df.groupby("month")["unemployment_rate"].mean()

    plt.figure(figsize=(8,5))
    monthly_avg.plot(marker="o")
    plt.title("Seasonal Trend of Unemployment Rate (Monthly Average)")
    plt.xlabel("Month")
    plt.ylabel("Avg Unemployment Rate (%)")
    plt.grid(True)
    plt.show()

# =========================
# 📌 Step 8: Regional Insights
# =========================
if "region" in df.columns:
    plt.figure(figsize=(12,6))
    sns.barplot(x="region", y="unemployment_rate", data=df, estimator=np.mean, ci=None)
    plt.xticks(rotation=90)
    plt.title("Average Unemployment Rate by Region")
    plt.ylabel("Unemployment Rate (%)")
    plt.show()

