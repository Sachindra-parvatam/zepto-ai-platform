# Zepto Data & AI Platform — Capstone Project

A three-module end-to-end AI/ML platform built for Zepto's analytics guild.

```
zepto-ai-platform/
├── data_pipeline/          # Module 1 — Data Pipeline (25 marks)
├── analytics/              # Module 2 — Analytics Pipeline (50 marks)
├── support_assistant/      # Module 3 — Support Assistant (25 marks)
├── requirements.txt        # Consolidated requirements for all modules
└── README.md               # This file
```

---

## Setup

### Single consolidated requirements file

All three modules share one `requirements.txt` at the repository root.

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Install all dependencies
pip install -r requirements.txt
```

---

## Module 1 — Data Pipeline (`/data_pipeline`)

**What it does:** Scrapes book catalog data from books.toscrape.com (≥ 60 books, ≥ 3 categories), cleans and enriches the data with a fixed GBP→INR conversion, loads it into a normalized two-table SQLite database, and queries it with SQL and pandas.

**Fixed conversion rate: 1 GBP = 105.50 INR** (project-defined constant, no API needed).

### Run

```bash
cd data_pipeline
python scrape_and_load.py
```

**Output:**
- `books.db` — SQLite database with `categories` and `books` tables
- Printed SQL query results and pandas equivalence check

### Design decisions

- **Scraping scope**: Iterates category pages until ≥ 60 books and ≥ 3 categories are collected.
- **Cleaning**: Price stripped with regex; star rating mapped from word to int; availability parsed to bool.
- **Missing value handling**: Numeric fields (price, rating) imputed with column median; rows dropped only if `title` or `category` is missing.
- **Currency**: `price_inr = price_gbp × 105.50`. No API call, no date reference.
- **Schema**: Two-table normalized design — `categories(category_id PK, category_name)` and `books(book_id PK, …, category_id FK → categories)`.
- **SQL coverage**: 6 queries covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, IN, and JOIN.

---

## Module 2 — Analytics Pipeline (`/analytics`)

**What it does:** Loads the Titanic dataset once via seaborn, cleans it, produces a full EDA story with ≥ 5 charts, trains three classifiers, evaluates them rigorously, handles class imbalance (3 strategies), tunes a Random Forest with GridSearchCV, adds a regression side-task, and saves the best pipeline with joblib.

### Run

```bash
cd analytics

# Part A — EDA (requires internet on first run for seaborn)
python 01_eda.py

# Part B — Modeling
python 02_modeling.py
```

**Output files:**
- `titanic.csv` — offline fallback (committed)
- `titanic_cleaned.csv` — cleaned dataset used by modeling
- `univariate_plots.png`, `correlation_heatmap.png`, `chart1–5 PNGs`, `decision_tree.png`, `roc_curves.png`, `regression_residuals.png`
- `titanic_pipeline.pkl` — saved complete sklearn Pipeline (preprocessor + estimator)

### Design decisions

- **One load**: `sns.load_dataset("titanic")` called exactly once in `01_eda.py`. `02_modeling.py` reads `titanic_cleaned.csv`.
- **Missing values**: `deck` (77.2%) dropped; `age` (19.9%) imputed with median; `embarked` (0.2%) rows dropped.
- **Train/test split**: Stratified 80/20 split — preserves the ~38/62 class balance in both sets.
- **Preprocessing**: `ColumnTransformer` with `SimpleImputer + StandardScaler` for numeric and `SimpleImputer + OneHotEncoder` for categorical. Fit on training data only; transform-only applied to test.
- **Imbalance handling**: Baseline vs `class_weight='balanced'` vs SMOTE (train fold only).
- **Best classifier**: Random Forest — highest AUC (~0.87) and F1 across the 3 models.
- **Saved artifact**: `titanic_pipeline.pkl` is a full `Pipeline(preprocessor, classifier)` — usable on raw input directly.

### Written interpretations

Full written interpretations (Task 3 skewness, Task 4 correlations, Task 5 chart captions, Task 7 stratification, Task 11 imbalance conclusion, Task 13 heteroscedasticity, Task 14 recommendation) are in `/analytics/README.md`.

---

## Module 3 — Support Assistant (`/support_assistant`)

**What it does:** A RAG-based support assistant grounded in 8 Zepto policy documents. Embeddings generated locally with `all-MiniLM-L6-v2` and stored in ChromaDB. LangGraph orchestrates intent classification → retrieval → answer generation. FastAPI exposes a `POST /ask` endpoint. Fully offline/deterministic in default mock mode.

### Run

```bash
cd support_assistant

# Step 1 — Build the ChromaDB index
python ingest.py

# Step 2 — Start the API server
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
cd support_assistant
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support
```

### Design decisions

- **MOCK_LLM=1 (default)**: All generation is deterministic and offline. `classify_intent` uses keyword heuristic; `retrieve_and_answer` returns a canned template from the top retrieved chunk; `direct_answer` returns a fixed string.
- **Retrieval always real**: ChromaDB + sentence-transformers require no API key — retrieval runs for real in both modes.
- **ChromaDB**: Persistent local store under `chroma_store/`. One collection `zepto_policies`.
- **LangGraph**: 3 nodes (`classify_intent`, `retrieve_and_answer`, `direct_answer`) with a conditional edge routing on intent.
- **Pydantic output schema**: `ZeptoResponse(answer, sources, confidence)` validated on every response.
- **Retry logic (real LLM path)**: Up to 3 total attempts with corrective instruction on Pydantic validation failure.

Full architecture description and example call transcripts are in `/support_assistant/README.md`.

---

## Repository Structure (full)

```
zepto-ai-platform/
│
├── requirements.txt
├── README.md
│
├── data_pipeline/
│   ├── scrape_and_load.py
│   ├── books.db              (generated by running the script)
│   └── README.md
│
├── analytics/
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── titanic.csv           (offline fallback — committed)
│   ├── titanic_cleaned.csv   (generated by 01_eda.py)
│   ├── titanic_pipeline.pkl  (generated by 02_modeling.py)
│   ├── *.png                 (charts generated by scripts)
│   └── README.md
│
└── support_assistant/
    ├── docs/
    │   ├── doc_01.txt  … doc_08.txt
    ├── ingest.py
    ├── graph.py
    ├── main.py
    ├── Dockerfile
    ├── requirements.txt
    └── README.md
```
