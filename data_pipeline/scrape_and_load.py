"""
Module 1 — Data Pipeline
Scrapes books.toscrape.com, cleans the data, converts GBP to INR,
loads into a normalized SQLite database, and runs SQL queries.

Fixed baseline conversion rate: 1 GBP = 105.50 INR
(This is a project-defined constant, not a live market rate.)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_URL = "https://books.toscrape.com/catalogue/"
GBP_TO_INR = 105.50          # Fixed project-defined constant
DB_PATH = os.path.join(os.path.dirname(__file__), "books.db")

# Word → integer mapping for star ratings
RATING_MAP = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5
}

# ─────────────────────────────────────────────
# STEP 1: SCRAPE
# ─────────────────────────────────────────────

def get_category_urls():
    """Fetch all category links from the homepage sidebar."""
    r = requests.get("https://books.toscrape.com/index.html", timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    sidebar = soup.select("ul.nav-list > li > ul > li > a")
    categories = {}
    for a in sidebar:
        name = a.text.strip()
        href = a["href"]          # e.g. "catalogue/category/books/mystery_3/index.html"
        full_url = "https://books.toscrape.com/" + href
        categories[name] = full_url
    return categories


def scrape_books_from_page(page_url, category_name):
    """Scrape all books listed on a single listing page."""
    r = requests.get(page_url, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    articles = soup.select("article.product_pod")
    books = []
    for art in articles:
        # Title
        title = art.h3.a["title"]

        # Price — strip £ symbol
        price_text = art.select_one("p.price_color").text.strip()

        # Star rating — class name encodes the word
        rating_cls = art.p["class"]            # e.g. ['star-rating', 'Three']
        star_word = rating_cls[1].lower() if len(rating_cls) > 1 else "one"

        # Availability
        avail_text = art.select_one("p.availability").text.strip()

        books.append({
            "title": title,
            "price_raw": price_text,
            "star_word": star_word,
            "availability_raw": avail_text,
            "category": category_name
        })
    return books, soup


def scrape_category(category_name, category_url, max_pages=5):
    """Paginate through a category and collect all book rows."""
    all_books = []
    page_url = category_url
    for _ in range(max_pages):
        books, soup = scrape_books_from_page(page_url, category_name)
        all_books.extend(books)
        # Check for 'next' button
        next_btn = soup.select_one("li.next > a")
        if not next_btn:
            break
        # Build next page URL relative to current page
        current_dir = page_url.rsplit("/", 1)[0] + "/"
        page_url = current_dir + next_btn["href"]
    return all_books


def scrape_all(min_books=60, min_categories=3):
    """Scrape enough categories until we have ≥ min_books across ≥ min_categories."""
    category_urls = get_category_urls()
    all_rows = []
    categories_scraped = []
    for cat_name, cat_url in category_urls.items():
        print(f"  Scraping category: {cat_name} …")
        rows = scrape_category(cat_name, cat_url)
        all_rows.extend(rows)
        categories_scraped.append(cat_name)
        if len(all_rows) >= min_books and len(categories_scraped) >= min_categories:
            break
    print(f"  Scraped {len(all_rows)} books across {len(categories_scraped)} categories.")
    return all_rows
    # ─────────────────────────────────────────────
# STEP 2: CLEAN
# ─────────────────────────────────────────────

def parse_price(raw):
    """Strip currency symbols and convert to float."""
    cleaned = re.sub(r"[^\d.]", "", raw)
    return float(cleaned) if cleaned else None


def parse_rating(star_word):
    """Convert word rating to integer 1–5."""
    return RATING_MAP.get(star_word.lower(), None)


def parse_availability(avail_text):
    """Return True if 'In stock', False otherwise."""
    return "in stock" in avail_text.lower()


def clean(raw_rows):
    """Convert raw scraped rows into a clean DataFrame."""
    records = []
    dropped = 0
    for row in raw_rows:
        price_gbp = parse_price(row["price_raw"])
        rating = parse_rating(row["star_word"])
        in_stock = parse_availability(row["availability_raw"])

        # Median imputation for price_gbp — collect valids first; apply after
        # For rating, if parse fails (None), we will impute below
        records.append({
            "title": row["title"],
            "price_gbp": price_gbp,
            "rating": rating,
            "in_stock": in_stock,
            "category": row["category"]
        })

    df = pd.DataFrame(records)

    # Median imputation for numeric fields
    price_median = df["price_gbp"].median()
    rating_median = int(df["rating"].median()) if df["rating"].notna().any() else 3

    before = len(df)
    # Only drop rows where title or category is missing — these are unrecoverable
    df = df.dropna(subset=["title", "category"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing title/category.")

    # Impute numeric fields
    df["price_gbp"] = df["price_gbp"].fillna(price_median)
    df["rating"] = df["rating"].fillna(rating_median).astype(int)
    df["in_stock"] = df["in_stock"].fillna(False)

    # ── Currency conversion (fixed-rate baseline) ──
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    # Ensure correct dtypes
    df["price_gbp"] = df["price_gbp"].astype(float)
    df["price_inr"] = df["price_inr"].astype(float)
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    
    return df
# ─────────────────────────────────────────────
# STEP 3: LOAD INTO SQLITE
# ─────────────────────────────────────────────

def create_schema(conn):
    """Create normalized two-table schema."""
    conn.executescript("""
        DROP TABLE IF EXISTS books;
        DROP TABLE IF EXISTS categories;

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
            in_stock     INTEGER NOT NULL,   -- 0/1 boolean
            category_id  INTEGER NOT NULL REFERENCES categories(category_id)
        );
    """)
    conn.commit()


def insert_data(conn, df):
    """Insert categories then books, maintaining FK relationship."""
    # Insert unique categories
    categories = df["category"].unique()
    for cat in categories:
        conn.execute(
            "INSERT OR IGNORE INTO categories (category_name) VALUES (?)", (cat,)
        )
    conn.commit()

    # Build category_name → id map
    cat_map = {row[1]: row[0] for row in conn.execute(
        "SELECT category_id, category_name FROM categories"
    )}

    # Insert books
    for _, row in df.iterrows():
        conn.execute(
            """INSERT INTO books
               (title, price_gbp, price_inr, rating, in_stock, category_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(row["in_stock"]),
                cat_map[row["category"]]
            )
        )
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    print(f"  Inserted {total} books into SQLite.")
    print(f"  Clean DataFrame shape: {df.shape}")
    return df
    # ─────────────────────────────────────────────
# STEP 4: SQL QUERIES (≥ 5, covering all required clauses + JOIN)
# ─────────────────────────────────────────────

def run_queries(conn):
    """Run all required SQL queries, print output, and return DataFrames."""
    queries = {}

    # Q1 — SELECT / WHERE / ORDER BY / LIMIT
    queries["Q1_cheapest_in_stock"] = """
        SELECT title, price_gbp, price_inr, rating
        FROM   books
        WHERE  in_stock = 1
        ORDER  BY price_gbp ASC
        LIMIT  10;
    """

    # Q2 — DISTINCT
    queries["Q2_distinct_categories"] = """
        SELECT DISTINCT category_name
        FROM   categories
        ORDER  BY category_name;
    """

    # Q3 — BETWEEN
    queries["Q3_mid_range_price"] = """
        SELECT title, price_gbp, rating
        FROM   books
        WHERE  price_gbp BETWEEN 10.00 AND 30.00
        ORDER  BY price_gbp DESC;
    """

    # Q4 — IN
    queries["Q4_high_rated"] = """
        SELECT title, rating, price_gbp
        FROM   books
        WHERE  rating IN (4, 5)
        ORDER  BY rating DESC, price_gbp ASC
        LIMIT  15;
    """

    # Q5 — JOIN (top 10 highest-rated books per category)
    queries["Q5_join_top_rated_per_category"] = """
        SELECT c.category_name,
               b.title,
               b.rating,
               b.price_gbp,
               b.price_inr
        FROM   books b
        JOIN   categories c ON b.category_id = c.category_id
        ORDER  BY c.category_name, b.rating DESC, b.price_gbp ASC
        LIMIT  30;
    """

    # Q6 — Aggregate + JOIN (average price per category)
    queries["Q6_avg_price_per_category"] = """
        SELECT c.category_name,
               COUNT(b.book_id)       AS total_books,
               ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp,
               ROUND(AVG(b.price_inr), 2) AS avg_price_inr
        FROM   books b
        JOIN   categories c ON b.category_id = c.category_id
        GROUP  BY c.category_name
        ORDER  BY avg_price_gbp DESC;
    """

    results = {}
    for name, sql in queries.items():
        print(f"\n{'─'*60}")
        print(f"  {name}")
        print(f"  SQL: {sql.strip()}")
        df = pd.read_sql(sql, conn)
        print(df.to_string(index=False))
        results[name] = df

    return results

# ─────────────────────────────────────────────
# STEP 5: pandas pd.read_sql AND pd.merge
# ─────────────────────────────────────────────

def pandas_operations(conn, df_books):
    """
    Demonstrate pd.read_sql and pd.merge equivalence for the JOIN query.
    df_books: the full cleaned books DataFrame (in-memory, with 'category' column).
    """
    print("\n" + "="*60)
    print("  PANDAS OPERATIONS — pd.read_sql vs pd.merge")
    print("="*60)

    # ── via pd.read_sql ──
    sql_join = """
        SELECT c.category_name,
               b.title,
               b.rating,
               b.price_gbp,
               b.price_inr
        FROM   books b
        JOIN   categories c ON b.category_id = c.category_id
        ORDER  BY c.category_name, b.rating DESC
        LIMIT  20;
    """
    df_sql = pd.read_sql(sql_join, conn)
    print("\n  [pd.read_sql result — first 10 rows]")
    print(df_sql.head(10).to_string(index=False))

    # ── via pd.merge (in-memory) ──
    # Load categories table into a DataFrame
    df_cats = pd.read_sql("SELECT category_id, category_name FROM categories", conn)
    df_books_db = pd.read_sql("SELECT * FROM books", conn)

    df_merged = (
        df_books_db
        .merge(df_cats, on="category_id")
        [["category_name", "title", "rating", "price_gbp", "price_inr"]]
        .sort_values(["category_name", "rating"], ascending=[True, False])
        .head(20)
        .reset_index(drop=True)
    )
    print("\n  [pd.merge result — first 10 rows]")
    print(df_merged.head(10).to_string(index=False))

    # Verify equivalence
    # Reset index and align columns for comparison
    df_sql_cmp = df_sql.reset_index(drop=True)
    df_merge_cmp = df_merged.reset_index(drop=True)
    match = df_sql_cmp.equals(df_merge_cmp)
    print(f"\n  pd.read_sql == pd.merge: {match}")
    return df_sql, df_merged


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ZEPTO DATA PIPELINE — Module 1")
    print("=" * 60)

    # 1. Scrape
    print("\n[1] Scraping books.toscrape.com …")
    raw_rows = scrape_all(min_books=60, min_categories=3)

    # 2. Clean
    print("\n[2] Cleaning data …")
    df = clean(raw_rows)
    print(df.dtypes)
    print(df.head(3))

    # 3. Load into SQLite
    print(f"\n[3] Loading into SQLite at {DB_PATH} …")
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)
    insert_data(conn, df)

    # 4. SQL queries
    print("\n[4] Running SQL queries …")
    query_results = run_queries(conn)

    # 5. pandas operations
    print("\n[5] pandas pd.read_sql and pd.merge …")
    pandas_operations(conn, df)

    conn.close()
    print("\n[✓] Pipeline complete. Database saved to:", DB_PATH)


if __name__ == "__main__":
    main()
