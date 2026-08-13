"""
Module 2 — Analytics Pipeline
Part A: Profiling, Cleaning, and EDA Story
Loads Titanic via seaborn, profiles, cleans, and produces EDA charts.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for script mode
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")
OUT_DIR = os.path.dirname(__file__)  # save artefacts alongside this script

# ─────────────────────────────────────────────
# TASK 1 — Load, profile, and save offline fallback
# ─────────────────────────────────────────────
print("=" * 60)
print("  TASK 1 — Load & Profile")
print("=" * 60)

# Load from seaborn (requires internet on first run; cached thereafter)
try:
    df_raw = sns.load_dataset("titanic")
    print("  Loaded from seaborn cache/network.")
except Exception:
    print("  seaborn load failed — reading from saved titanic.csv")
    df_raw = pd.read_csv(os.path.join(OUT_DIR, "titanic.csv"))

# Save offline fallback immediately after the ONE load
csv_path = os.path.join(OUT_DIR, "titanic.csv")
df_raw.to_csv(csv_path, index=False)
print(f"  Offline fallback saved to: {csv_path}")

print("\n--- df.shape ---")
print(df_raw.shape)

print("\n--- df.info() ---")
df_raw.info()

print("\n--- df.describe() ---")
print(df_raw.describe(include="all"))

# Missing value percentages
print("\n--- Missing value % per column ---")
missing_pct = (df_raw.isnull().sum() / len(df_raw) * 100).round(2)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
print(missing_pct)

# ─────────────────────────────────────────────
# TASK 2 — Missing-value handling
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TASK 2 — Missing-value Handling")
print("=" * 60)

df = df_raw.copy()

"""
Threshold rule applied:
  < 5%  missing  → drop those rows
  5–30% missing  → impute
  > 30% missing  → drop column OR encode as own category

Column analysis (measured on the raw load):
  deck       ~77.2%  missing → DROP COLUMN (too sparse to impute reliably)
  age        ~19.9%  missing → IMPUTE with median (numeric; 5–30% band)
  embarked    ~0.2%  missing → DROP ROWS (< 5% band; only 2 rows)
  embark_town ~0.2%  missing → DROP ROWS (same 2 rows as embarked)
"""

# deck: 77.2% missing — drop column
print(f"  'deck' missing: {missing_pct.get('deck', 0):.1f}% → DROP COLUMN")
df.drop(columns=["deck"], inplace=True)

# age: ~19.9% missing — impute with median
age_missing_pct = missing_pct.get("age", 0)
print(f"  'age' missing: {age_missing_pct:.1f}% → IMPUTE with median")
age_median = df["age"].median()
df["age"].fillna(age_median, inplace=True)

# embarked / embark_town: < 5% — drop those rows
embarked_missing_pct = missing_pct.get("embarked", 0)
print(f"  'embarked' missing: {embarked_missing_pct:.1f}% → DROP ROWS (< 5%)")
df.dropna(subset=["embarked", "embark_town"], inplace=True)

print(f"  DataFrame shape after cleaning: {df.shape}")
print(f"  Remaining nulls:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ─────────────────────────────────────────────
# TASK 3 — Univariate analysis
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TASK 3 — Univariate Analysis")
print("=" * 60)

def iqr_outlier_count(series, label):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    print(f"  {label}: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
          f"bounds=[{lower:.2f}, {upper:.2f}], outliers={len(outliers)}")
    return len(outliers)


# IQR outlier counts
age_outliers = iqr_outlier_count(df["age"], "age")
fare_outliers = iqr_outlier_count(df["fare"], "fare")

# Skewness stats for fare
fare_mean   = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode   = df["fare"].mode()[0]
print(f"\n  fare — mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}")
print("  Skewness conclusion: fare is RIGHT-SKEWED.")
print("  (mean > median > mode, indicating a long tail towards high fares)")

# Histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
df["age"].hist(bins=30, ax=axes[0][0], color="steelblue", edgecolor="white")
axes[0][0].set_title("Age — Histogram"); axes[0][0].set_xlabel("Age")

df["fare"].hist(bins=40, ax=axes[0][1], color="coral", edgecolor="white")
axes[0][1].set_title("Fare — Histogram"); axes[0][1].set_xlabel("Fare (£)")

df.boxplot(column="age", ax=axes[1][0])
axes[1][0].set_title("Age — Box Plot")

df.boxplot(column="fare", ax=axes[1][1])
axes[1][1].set_title("Fare — Box Plot")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "univariate_plots.png"), dpi=100)
plt.close()
print("  Saved: univariate_plots.png")

# ─────────────────────────────────────────────
# TASK 4 — Bivariate analysis + Correlation matrix
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TASK 4 — Bivariate Analysis")
print("=" * 60)

# (a) Survival rate by sex
surv_sex = df.groupby("sex")["survived"].mean().round(4)
print("\n  Survival rate by sex:")
print(surv_sex)

# (b) Survival rate by pclass
surv_pclass = df.groupby("pclass")["survived"].mean().round(4)
print("\n  Survival rate by pclass:")
print(surv_pclass)

# (c) Survival rate by sex AND pclass (boolean masking)
print("\n  Survival rate by sex & pclass (boolean masking):")
for sex in ["male", "female"]:
    for pclass in [1, 2, 3]:
        mask = (df["sex"] == sex) & (df["pclass"] == pclass)
        rate = df.loc[mask, "survived"].mean()
        print(f"    {sex}, pclass {pclass}: {rate:.4f}")

# Correlation matrix — exactly the 6 specified columns
corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df[corr_cols].corr()
print("\n  Correlation matrix (6 columns):")
print(corr_matrix.round(3))

# Find two strongest off-diagonal correlations
corr_abs = corr_matrix.abs().copy()
for i in range(len(corr_abs)):
    corr_abs.iloc[i, i] = 0
top2 = (corr_abs
        .unstack()
        .sort_values(ascending=False)
        .drop_duplicates()
        .head(2))
print("\n  Two strongest correlations (by absolute value):")
print(top2)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, square=True)
plt.title("Correlation Matrix — Titanic (6 features)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=100)
plt.close()
print("  Saved: correlation_heatmap.png")

# ─────────────────────────────────────────────
# TASK 5 — Multivariate data story (≥ 4 charts)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TASK 5 — Multivariate Data Story")
print("=" * 60)

# Chart 1: Survival count by sex (bar)
fig, ax = plt.subplots(figsize=(7, 5))
df.groupby(["sex", "survived"]).size().unstack().plot(
    kind="bar", ax=ax, color=["#e07070", "#70a8e0"]
)
ax.set_title("Chart 1: Survival Count by Sex")
ax.set_xlabel("Sex"); ax.set_ylabel("Count")
ax.legend(["Did not survive", "Survived"])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "chart1_survival_by_sex.png"), dpi=100)
plt.close()

# Chart 2: Survival rate by pclass (bar)
fig, ax = plt.subplots(figsize=(7, 5))
df.groupby("pclass")["survived"].mean().plot(kind="bar", ax=ax, color="teal", rot=0)
ax.set_title("Chart 2: Survival Rate by Passenger Class")
ax.set_xlabel("Passenger Class"); ax.set_ylabel("Survival Rate")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "chart2_survival_by_pclass.png"), dpi=100)
plt.close()

# Chart 3: Age distribution by survival (box)
fig, ax = plt.subplots(figsize=(7, 5))
df.boxplot(column="age", by="survived", ax=ax)
ax.set_title("Chart 3: Age Distribution by Survival")
ax.set_xlabel("Survived (0=No, 1=Yes)"); ax.set_ylabel("Age")
plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "chart3_age_by_survival.png"), dpi=100)
plt.close()

# Chart 4: Fare by pclass and survival (box)
fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(data=df, x="pclass", y="fare", hue="survived", ax=ax,
            palette={0: "#e07070", 1: "#70a8e0"})
ax.set_title("Chart 4: Fare Distribution by Pclass and Survival")
ax.set_xlabel("Passenger Class"); ax.set_ylabel("Fare (£)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "chart4_fare_pclass_survival.png"), dpi=100)
plt.close()

# Chart 5 (bonus): Survival rate heatmap — sex × pclass
pivot = df.groupby(["sex", "pclass"])["survived"].mean().unstack()
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
            linewidths=0.5, vmin=0, vmax=1)
ax.set_title("Chart 5: Survival Rate — Sex × Pclass")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "chart5_survival_sex_pclass_heatmap.png"), dpi=100)
plt.close()

print("  Saved: chart1 – chart5 PNGs")

# ─────────────────────────────────────────────
# TASK 6 — Exploratory z-score standardisation (sanity check only)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TASK 6 — Z-score Standardisation (EDA sanity check)")
print("=" * 60)

from sklearn.preprocessing import StandardScaler

scaler_check = StandardScaler()
df_std = df.copy()
df_std[["age_z", "fare_z"]] = scaler_check.fit_transform(df[["age", "fare"]])

print("  Before standardisation:")
print(f"    age  — mean={df['age'].mean():.2f}, std={df['age'].std():.2f}")
print(f"    fare — mean={df['fare'].mean():.2f}, std={df['fare'].std():.2f}")

print("  After standardisation (z-scores):")
print(f"    age_z  — mean={df_std['age_z'].mean():.4f}, std={df_std['age_z'].std():.4f}")
print(f"    fare_z — mean={df_std['fare_z'].mean():.4f}, std={df_std['fare_z'].std():.4f}")

# Save cleaned dataset for modeling notebook
cleaned_path = os.path.join(OUT_DIR, "titanic_cleaned.csv")
df.to_csv(cleaned_path, index=False)
print(f"\n  Cleaned dataset saved to: {cleaned_path}")
print("\n[✓] EDA complete.")
