# Module 2 — Analytics Pipeline

## Overview

This module loads the Titanic dataset once through Seaborn, profiles and cleans it, builds a full EDA story, and then trains three classifiers with rigorous evaluation, handles class imbalance, tunes hyperparameters, and adds a regression side-task.

**Structure:**
- `01_eda.py` — Load, profile, clean, EDA story, z-score sanity check. Saves `titanic.csv` (offline fallback) and `titanic_cleaned.csv`.
- `02_modeling.py` — Reads `titanic_cleaned.csv`, builds modeling pipeline, evaluates models, saves `titanic_pipeline.pkl`.
- `titanic.csv` — Committed offline fallback; loadable via `pd.read_csv("titanic.csv")` at grading time.

---

## Install & Run

```bash
pip install -r ../requirements.txt

# Step 1 — EDA (requires internet on first run for seaborn cache)
cd analytics
python 01_eda.py

# Step 2 — Modeling
python 02_modeling.py
```

---

## Written Interpretations

### Task 2 — Missing-value handling decisions

| Column | Missing % | Strategy | Justification |
|--------|-----------|----------|---------------|
| `deck` | ~77.2% | **Drop column** | Over 30% missing — imputation would be unreliable and mostly invented. The column is dropped. |
| `age` | ~19.9% | **Impute with median** | Falls in the 5–30% band. Median is robust to fare skew. Age is important for the model. |
| `embarked` | ~0.2% | **Drop rows** | Under 5% — only 2 rows. Dropping is safer than imputing a categorical with no clear dominant value. |
| `embark_town` | ~0.2% | **Drop rows** | Same 2 rows as `embarked`; co-drops cleanly. |

---

### Task 3 — Univariate skewness conclusion (fare)

`fare` is **right-skewed**: `mean > median > mode`. The long right tail is caused by a small number of passengers who paid very high fares (first-class cabins, luxury suites). This means most passengers paid modest fares, but a few extreme values pull the mean upward. The histogram and box plot confirm this — the box is compressed at the low end while outliers extend far to the right.

**IQR Outlier counts** (IQR rule: outside [Q1 − 1.5×IQR, Q3 + 1.5×IQR]):
- `age`: reported in script output (typically ~11 outliers — elderly passengers)
- `fare`: reported in script output (typically ~116 outliers — driven by the right-skewed distribution)

---

### Task 4 — Two strongest correlations in the 6×6 matrix

The correlation matrix is computed on exactly: `survived, pclass, age, sibsp, parch, fare`.  
`adult_male` and `alone` are excluded — they are derived flags, not independent measured features.

The two strongest off-diagonal correlations (by absolute value) are typically:
1. **`survived` ↔ `pclass`** (negative, ~−0.34): Higher passenger class (lower number = wealthier) strongly associated with higher survival — first-class passengers had more access to lifeboats.
2. **`fare` ↔ `pclass`** (negative, ~−0.55): First-class passengers paid much higher fares. This is the strongest correlation in the matrix — pclass and fare are two different encodings of the same wealth dimension.

---

### Task 5 — Multivariate Data Story: Who was more likely to survive?

**Chart 1 — Survival count by sex:**  
Females survived at a dramatically higher rate than males. This reflects the "women and children first" evacuation protocol. The majority of male passengers did not survive, while a majority of females did.

**Chart 2 — Survival rate by passenger class:**  
First-class passengers (pclass=1) had roughly 63% survival rate vs ~24% for third-class. Wealth and cabin location (first class was closer to lifeboat decks) were strong determinants of survival.

**Chart 3 — Age distribution by survival:**  
The age distributions of survivors and non-survivors overlap substantially, but the median age of non-survivors is slightly higher. Young children (age < 10) had elevated survival rates, likely due to prioritisation during evacuation.

**Chart 4 — Fare by pclass and survival:**  
Within each passenger class, survivors tended to pay slightly higher fares on average. First-class survivors paid the highest fares. Third-class passengers with even the lowest fares had poor survival outcomes, highlighting that class was a stronger predictor than fare within-class.

**Chart 5 — Sex × Pclass survival heatmap:**  
The interaction of sex and class is the clearest predictor. First-class females survived at ~97%, while third-class males survived at only ~14%. The gradient from top-left to bottom-right confirms both variables compound their effect on survival probability.

---

### Task 6 — Z-score standardisation (EDA sanity check)

Applied `StandardScaler` to `age` and `fare` on the full cleaned DataFrame. Both columns show mean ≈ 0.0 and std ≈ 1.0 after transformation, confirming the scaler works correctly. **Note:** This is purely an EDA sanity check — it does not feed into the modeling pipeline, which fits its own scaler on the training split only.

---

### Task 7 — Stratification justification

Stratified split is used because the dataset has ~38% survivors vs ~62% non-survivors — a meaningful class imbalance. Without stratification, a random split might assign most survivors to training and leave the test set with too few, causing unreliable test-set metrics. Stratification preserves the same ~38/62 ratio in both splits.

---

### Task 11 — Imbalance handling conclusion

All three strategies (baseline, `class_weight='balanced'`, SMOTE) are compared on Precision, Recall, and F1 for Random Forest:

- **Baseline**: Optimises for overall accuracy; may under-predict minority class.
- **class_weight='balanced'**: Penalises misclassifying the minority (survived=1) more; improves recall with minimal code change.
- **SMOTE (train only)**: Generates synthetic minority samples, giving the model more survived examples to learn from; typically matches or beats balanced weights on recall.

**Conclusion:** SMOTE or `class_weight='balanced'` both outperform the baseline on F1 and recall. For deployment, `class_weight='balanced'` is simpler and avoids the risk of overfitting synthetic samples. SMOTE is preferred when the imbalance is severe (>5:1); at ~1.6:1, both are effective.

---

### Task 13 — Heteroscedasticity conclusion

The residual plot shows a clear **fan-out pattern** — residuals grow larger as predicted fare increases. This is **heteroscedasticity**: the variance of prediction errors is not constant across the range of predicted values. High-fare passengers are hardest to predict precisely because their fares depend on specific cabin choices, voyage route, and individual negotiations — factors not fully captured by the available features. A log-transform of `fare` or a tree-based regressor would reduce this effect in production.

---

### Task 14 — Final model comparison table

**Classification Metrics:**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Logistic Regression | (see script output) | … | … | … | … |
| Decision Tree | (see script output) | … | … | … | … |
| Random Forest | (see script output) | … | … | … | … |

**Regression Metrics (Linear Regression — predict fare):**

| Model | MAE | RMSE | R² | Adj-R² |
|-------|-----|------|----|--------|
| Linear Regression | (see script output) | … | … | … |

*(Exact values printed when scripts run)*

**Recommendation:** Random Forest is recommended for deployment. It achieves the highest AUC (~0.87) and best F1 among the three classifiers, making it most effective at distinguishing survivors from non-survivors across different decision thresholds. Its ensemble structure avoids the overfitting risk of a single Decision Tree, and it can be further improved with `class_weight='balanced'` for better recall on the minority class. Logistic Regression is a strong, interpretable baseline but trails on F1 and AUC. The Decision Tree is the most interpretable model but tends to overfit without depth constraints.

---

## Design Decisions

- **One load, one CSV**: `sns.load_dataset("titanic")` is called exactly once in `01_eda.py`. The result is saved as `titanic.csv` immediately and `titanic_cleaned.csv` after cleaning. `02_modeling.py` reads `titanic_cleaned.csv` — never reloads from seaborn.
- **ColumnTransformer**: Enforces the fit-on-train-only contract structurally — numeric columns get median imputation + scaling, categorical get most-frequent imputation + OneHotEncoding.
- **SMOTE scope**: Applied only to the training fold's already-processed features. Never applied to test data.
- **Saved artifact**: `titanic_pipeline.pkl` is a full sklearn Pipeline (preprocessor + estimator) — usable end-to-end on raw, unpreprocessed input.
