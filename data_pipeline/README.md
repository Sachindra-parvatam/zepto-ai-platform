# Module 1 — Data Pipeline

## Overview

This module scrapes book catalog data from [books.toscrape.com](https://books.toscrape.com), cleans and enriches it, and loads it into a normalized SQLite database. It then demonstrates querying via both raw SQL and pandas.

---

## Fixed Currency Conversion Rate

**1 GBP = 105.50 INR**

This is a project-defined constant for this assignment — not a live or historical market rate. No external API call is made. The `price_inr` column in the database is computed exclusively using this rate:

```python
price_inr = price_gbp * 105.50
```

---

## Install & Run

### Install dependencies

```bash
pip install -r ../requirements.txt
```

Or, if using the module-level requirements:

```bash
pip install requests beautifulsoup4 pandas
```

### Run the full pipeline

```bash
cd data_pipeline
python scrape_and_load.py
```

This will:
1. Scrape ≥ 60 books across ≥ 3 categories from books.toscrape.com
2. Clean the data (strip £, convert ratings, parse availability)
3. Convert GBP → INR using the fixed rate
4. Create a normalized SQLite schema (`books.db`)
5. Insert all records
6. Run 6 SQL queries with printed output
7. Show `pd.read_sql` vs `pd.merge` equivalence

---

## Database Schema

Two tables with a PK/FK relationship:

```sql
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT    NOT NULL UNIQUE
);

CREATE TABLE books (
    book_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT    NOT NULL,
    price_gbp    REAL    NOT NULL,
    price_inr    REAL    NOT NULL,
    rating       INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    in_stock     INTEGER NOT NULL,   -- stored as 0/1 boolean
    category_id  INTEGER NOT NULL REFERENCES categories(category_id)
);
```

---

## SQL Queries Covered

| # | Query | Clauses Demonstrated |
|---|-------|----------------------|
| Q1 | 10 cheapest in-stock books | SELECT, WHERE, ORDER BY, LIMIT |
| Q2 | All distinct category names | SELECT, DISTINCT, ORDER BY |
| Q3 | Books priced £10–£30 | WHERE BETWEEN |
| Q4 | Books rated 4 or 5 stars | WHERE IN, ORDER BY, LIMIT |
| Q5 | Top-rated books with category name | JOIN (books ↔ categories), ORDER BY |
| Q6 | Average price per category | JOIN, GROUP BY, aggregate functions |

---

## Cleaning Decisions

### `price_gbp`
- The `£` symbol (and any whitespace) is stripped using a regex `[^\d.]`.
- Converted to `float`. If parsing fails (unexpected format), the value becomes `None`.
- **Imputation**: `None` values are replaced with the **median** of all successfully parsed prices. Median is preferred over mean for prices because it is robust to skewed distributions and outliers. No rows were dropped for this field.

### `rating` (integer 1–5)
- The HTML class name encodes the rating as a word (e.g., `"Three"`).
- Mapped to integer using `{"one":1, …, "five":5}`.
- If the word is unrecognised, it parses to `None`.
- **Imputation**: `None` values are replaced with the **median integer rating**. Same reasoning as above — robust to outliers.

### `in_stock` (boolean)
- Derived from the availability text: `True` if `"in stock"` appears (case-insensitive), `False` otherwise.
- This is a binary classification — no imputation needed; ambiguous text maps to `False` by default.

### Row dropping
- A row is **dropped** only if `title` or `category` is missing, as these are non-recoverable identity fields with no meaningful imputation.
- In practice, books.toscrape.com has clean data and no rows were dropped in test runs.

---

## Design Decisions

- **Scope**: The scraper iterates through category pages in order until it has collected ≥ 60 books across ≥ 3 categories. This keeps the scrape fast and deterministic.
- **Pagination**: Each category's pages are followed via the `<li class="next">` link in the HTML, up to 5 pages per category.
- **No anti-scraping workarounds needed**: books.toscrape.com is built for scraping practice — no rate limiting, CAPTCHA, or login.
- **SQLite chosen** for portability — the `books.db` file is self-contained and requires no server.
- **`pd.read_sql` vs `pd.merge`**: Both approaches produce the same joined result, confirming that SQL joins and in-memory DataFrame merges are equivalent operations on this dataset.
