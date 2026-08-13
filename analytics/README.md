# Module 2 — Analytics Pipeline

## Overview

A comprehensive analytics workflow: profile the Titanic dataset, handle missing values defensibly, produce a visual data story, train and evaluate three classifiers with rigorous metrics, handle class imbalance, tune hyperparameters, and save a production-ready pipeline.

## How to Run

```bash
cd analytics

# Step 1: EDA (requires internet on first run for seaborn dataset download)
python 01_eda.py

# Step 2: Modeling (reads cleaned data from Step 1)
python 02_modeling.py
```

## Output Files

### From 01_eda.py
- `titanic.csv` — Offline fallback (committed to repo)
- `titanic_cleaned.csv` — Cleaned dataset used by modeling
- `univariate_plots.png` — Age & fare histograms + box plots
- `correlation_heatmap.png` — 6-feature correlation matrix
- `chart1_survival_by_sex.png` through `chart5_survival_sex_pclass_heatmap.png`

### From 02_modeling.py
- `decision_tree.png` — Visualized decision tree with feature/class labels
- `roc_curves.png` — ROC curves for all 3 classifiers
- `regression_residuals.png` — Residual plot for fare prediction
- `titanic_pipeline.pkl` — Complete sklearn Pipeline (preprocessor + classifier)

## Design Decisions

### Part A — EDA

#### Data Loading (Task 1)
- **One load only**: `sns.load_dataset("titanic")` called exactly once in `01_eda.py`
- **Offline fallback**: Immediately saved as `titanic.csv` for grading without network
- **Modeling continues from same data**: `02_modeling.py` reads `titanic_cleaned.csv`

#### Missing Value Handling (Task 2)

Applied threshold rule: < 5% → drop rows, 5-30% → impute, > 30% → drop column or encode as category

| Column | Missing % | Strategy | Justification |
|--------|-----------|----------|---------------|
| `deck` | 77.2% | Drop column | Too sparse to impute reliably; mostly missing across all passenger classes |
| `age` | 19.9% | Impute with median | Within 5-30% band; age is numeric and important for survival analysis |
| `embarked` | 0.2% | Drop rows | < 5% threshold; only 2 rows affected |
| `embark_town` | 0.2% | Drop rows | Same 2 rows as embarked |

**Result**: 889 rows × 14 columns (deck dropped, 2 rows dropped)

#### Univariate Analysis (Task 3)

**IQR Outlier Counts** (outliers outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]):
- `age`: 8 outliers  
- `fare`: 114 outliers

**Fare Distribution (Right-Skewed)**:
- Mean: 32.10  
- Median: 14.45  
- Mode: 8.05

**Conclusion**: Fare is **right-skewed** (mean > median > mode), indicating a long tail of high-fare passengers

#### Bivariate Analysis (Task 4)

**Survival Rates** (boolean masking with `&` operator):

*By sex:*
- Female: 74.0%
- Male: 18.9%

*By pclass:*
- 1st class: 62.6%
- 2nd class: 47.3%
- 3rd class: 24.2%

*By sex AND pclass:*
- Female, 1st: 96.7%
- Female, 2nd: 92.1%
- Female, 3rd: 50.0%
- Male, 1st: 36.9%
- Male, 2nd: 15.7%
- Male, 3rd: 13.5%

**Correlation Matrix** (6 columns: survived, pclass, age, sibsp, parch, fare):

Top 2 strongest correlations (by absolute value):
1. **fare ↔ pclass**: -0.548 (negative) — Higher passenger class (lower number) strongly correlates with higher fares
2. **parch ↔ sibsp**: 0.415 (positive) — Passengers with siblings/spouses aboard tend to travel with parents/children too (family groups)

*Note: adult_male and alone excluded as redundant derived flags*

#### Multivariate Data Story (Task 5)

**Chart 1 — Survival Count by Sex** (bar):  
Women vastly outnumber men among survivors. The "women and children first" evacuation protocol is clearly reflected in the data.

**Chart 2 — Survival Rate by Passenger Class** (bar):  
First-class passengers had a 63% survival rate vs. only 24% for third-class. Proximity to lifeboats and evacuation priority favored upper decks.

**Chart 3 — Age Distribution by Survival** (box):  
Survivors skew slightly younger (median ~28) vs. non-survivors (median ~30), but the distributions largely overlap—age alone was not a strong predictor.

**Chart 4 — Fare by Pclass and Survival** (box):  
Within each class, survivors paid slightly higher fares on average, suggesting cabin location (even within class) mattered. Third-class fares show the tightest range.

**Chart 5 — Survival Rate Heatmap: Sex × Pclass** (heatmap):  
First-class women had a 97% survival rate (dark green). Third-class men had only 14%—a 7× difference. This interaction effect (sex + class together) is the strongest survival predictor.

#### Z-score Standardization Check (Task 6)

**Before standardization:**
- age: mean=29.64, std=14.49  
- fare: mean=32.10, std=49.70

**After standardization (z-scores):**
- age_z: mean=0.0000, std=1.0007  
- fare_z: mean=0.0000, std=1.0006

✅ Confirms correct standardization (mean ≈ 0, std ≈ 1)

---

### Part B — Predictive Modeling

#### Stratified Train/Test Split (Task 7)

**Class balance**:
- Not survived (0): 549 (61.8%)  
- Survived (1): 340 (38.2%)

**Split**: 80% train, 20% test with `stratify=y`

**Justification**: Stratification preserves the ~38/62 class distribution in both train and test sets. Without stratification, a random split could yield a test set with disproportionately more/fewer survivors, leading to biased evaluation metrics. Critical when class imbalance exists (as here).

**Verification**:
- Train survival rate: 38.3%  
- Test survival rate: 38.2%  
✅ Stratification successful

#### Preprocessing Pipeline (Task 8)

**Implementation**: `ColumnTransformer` inside a `Pipeline`

**Numeric features** (age, fare, sibsp, parch, pclass):
1. `SimpleImputer(strategy="median")`  
2. `StandardScaler()`

**Categorical features** (sex, embarked):
1. `SimpleImputer(strategy="most_frequent")`  
2. `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

**Critical rule**: All preprocessing steps **fit on training data only**, then applied in **transform-only mode** to test data. No step is fit or refit on test data or full pre-split dataset.

#### Model Training & Evaluation (Tasks 9-10)

Three classifiers trained on identical train/test split:

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| **Logistic Regression** | 80.9% | 0.783 | 0.691 | 0.734 | **0.861** |
| **Decision Tree** | 80.9% | 0.815 | 0.647 | 0.721 | 0.856 |
| **Random Forest** | 82.0% | 0.781 | 0.735 | 0.758 | 0.821 |

**Decision Tree**: Visualized with `plot_tree(max_depth=4)` showing feature names (age, fare, sex_encoded, etc.) and class labels ("Not Survived", "Survived")

#### Imbalance Handling Comparison (Task 11)

**Class distribution**: 439 not survived : 272 survived (1.61:1 ratio)

Retrained Random Forest with 3 strategies:

| Strategy | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Baseline (no handling) | 0.781 | 0.735 | 0.758 |
| `class_weight='balanced'` | 0.754 | 0.765 | 0.759 |
| SMOTE (train only) | 0.791 | 0.779 | **0.785** |

**Conclusion**: SMOTE and balanced weights both improve recall by helping the model learn the minority (survived) class better. SMOTE synthesizes new minority samples so the RF sees a balanced training set; balanced weights penalize misclassifying the minority more heavily. The trade-off is slightly lower precision. For a life-safety context (e.g., identifying survivors), higher recall (catching more true positives) is preferable. **SMOTE achieved the best F1 score (0.785)** and is recommended.

#### Hyperparameter Tuning (Task 12)

**GridSearchCV** on Random Forest:
- `n_estimators`: [100, 200]
- `max_depth`: [None, 5, 10]
- `max_features`: ["sqrt", "log2"]

**Results**:
- Best parameters: `{'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 100}`
- Best CV F1 score: 0.7420
- Out-of-bag (OOB) score: 0.8087

#### Regression Side-Task (Task 13)

**Goal**: Predict `fare` from other features (pclass, sex, age, sibsp, parch, embarked)

**Model**: Linear Regression (multivariate)

**Metrics**:
- MAE: 21.14  
- RMSE: 41.75  
- R²: 0.3468  
- Adjusted R²: 0.3239

**Heteroscedasticity**: **YES** — The residual plot shows residuals "fanning out" as predicted fare increases. High-fare passengers have much larger prediction errors. This is typical for skewed financial data like ticket prices, where variance increases with the mean.

#### Final Recommendation (Task 14)

**Random Forest** is the recommended classifier for deployment. It consistently outperforms Logistic Regression and Decision Tree on **F1 (0.758)** and achieves the highest **accuracy (82.0%)**—the two most balanced metrics for an imbalanced binary classification problem. While Logistic Regression has a slightly higher AUC (0.861), Random Forest's ensemble nature makes it more robust to overfitting compared to a single Decision Tree, and its F1 score indicates better overall precision-recall balance. Combining Random Forest with **class_weight='balanced'** or **SMOTE** (Task 11) further improves recall to 77-78% without significant precision loss, making it the best production choice when correctly identifying survivors matters most.

#### Saved Pipeline (Task 15)

**File**: `titanic_pipeline.pkl`

**Contents**: Complete sklearn `Pipeline` object:
```python
Pipeline([
    ('preprocessor', ColumnTransformer(...)),  # imputer + scaler + encoder
    ('classifier', RandomForestClassifier(...))
])
```

**Usage**:
```python
loaded_pipeline = joblib.load("titanic_pipeline.pkl")
predictions = loaded_pipeline.predict(raw_dataframe)  # No preprocessing needed!
```

✅ Reloaded and verified: predicts correctly on raw, unprocessed input

---

## Acceptance Criteria Met

✅ Missing-value percentages reported with threshold-based strategies justified  
✅ One load (`sns.load_dataset`) + committed `titanic.csv` offline fallback  
✅ IQR outlier counts for age/fare + skewness analysis (mean/median/mode)  
✅ All three bivariate breakdowns + 6-column correlation matrix (adult_male/alone excluded)  
✅ ≥ 4 multivariate charts with written interpretations + z-score check  
✅ Stratified train/test split with justification  
✅ All preprocessing fit on train only, transform-only on test  
✅ 3 classifiers trained + decision tree visualized + full metric suite  
✅ 3-way imbalance comparison (baseline/balanced/SMOTE) with conclusion  
✅ GridSearchCV + OOB score for `RandomForestClassifier(oob_score=True)`  
✅ Regression task with all 4 metrics + heteroscedasticity conclusion  
✅ Model comparison table + 3-5 sentence recommendation  
✅ Complete pipeline saved with joblib, reloadable, usable on raw input

## Dependencies

```
numpy>=1.23.0
pandas>=1.5.0
seaborn>=0.12.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
imbalanced-learn>=0.11.0
joblib>=1.2.0
```
