# Module 1 — Data Pipeline

## Overview

This module scrapes book catalog data from `books.toscrape.com`, cleans and enriches it with a fixed GBP→INR currency conversion, loads it into a normalized SQLite database, and demonstrates both SQL and pandas querying capabilities.

## Fixed Conversion Rate

**1 GBP = 105.50 INR**

This is a project-defined constant baseline rate. No API call or date reference is required.

## How to Run

```bash
cd data_pipeline
python scrape_and_load.py
```

## Output

- `books.db` — SQLite database with normalized schema
- Console output showing:
  - Scraping progress (categories and book counts)
  - Data cleaning statistics
  - SQL query results (6 queries)
  - pandas operations verification

## Database Schema

### Two-table normalized design:

**categories**
- `category_id` INTEGER PRIMARY KEY AUTOINCREMENT
- `category_name` TEXT NOT NULL UNIQUE

**books**
- `book_id` INTEGER PRIMARY KEY AUTOINCREMENT  
- `title` TEXT NOT NULL
- `price_gbp` REAL NOT NULL
- `price_inr` REAL NOT NULL
- `rating` INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5)
- `in_stock` INTEGER NOT NULL  (0/1 boolean)
- `category_id` INTEGER NOT NULL REFERENCES categories(category_id)

## Design Decisions

### Scraping Strategy
- **Scope**: Iterates through category pages until ≥ 60 books across ≥ 3 categories
- **Error handling**: Continues scraping if individual pages fail
- **Pagination**: Follows "next" links automatically

### Data Cleaning

**Price (GBP)**
- Strip currency symbols using regex: `re.sub(r"[^\d.]", "", raw_price)`
- Convert to float
- Median imputation for any parse failures

**Rating**
- Map text ratings ("One"..."Five") to integers (1-5) using dictionary
- Median imputation for invalid values

**Availability**
- Parse "In stock" text to boolean using `"in stock" in text.lower()`
- Default to False for ambiguous text

**Missing Values**
- Numeric fields (price, rating): Impute with column median
- Essential fields (title, category): Drop the row if missing
- **Justification**: Titles and categories are identity fields—a book without these cannot be meaningfully stored or queried

### Currency Conversion
- `price_inr = price_gbp × 105.50`
- Computed after cleaning, rounded to 2 decimal places
- No external API, no network dependency

### SQL Query Coverage

The script executes **6 SQL queries** demonstrating:

1. **Q1**: SELECT, WHERE, ORDER BY, LIMIT (top 10 cheapest in-stock books)
2. **Q2**: DISTINCT (unique category names)
3. **Q3**: BETWEEN (mid-range price books: £10-30)
4. **Q4**: IN (high-rated books with rating 4 or 5)
5. **Q5**: JOIN (top-rated books per category, books ⨝ categories)
6. **Q6**: JOIN + GROUP BY + Aggregate (average price per category)

### pandas Equivalence Check

- **`pd.read_sql`**: Load query results directly from SQL
- **`pd.merge`**: Reproduce JOIN query in-memory using pandas merge operation
- **Verification**: Both approaches produce identical DataFrames (validated with `.equals()`)

## Dependencies

```
requests>=2.28.0
beautifulsoup4>=4.12.0
pandas>=1.5.0
```

(sqlite3 is part of Python's standard library)

## Acceptance Criteria Met

✅ Scraped ≥ 60 books across ≥ 3 categories  
✅ Fixed conversion rate (1 GBP = 105.50 INR) applied and documented  
✅ Normalized two-table PK/FK schema implemented  
✅ ≥ 5 SQL queries with all required clauses + JOIN  
✅ `pd.read_sql` and `pd.merge` equivalence demonstrated  
✅ All data properly typed (price_gbp: float, rating: int, in_stock: bool, price_inr: float)
