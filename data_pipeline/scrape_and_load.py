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