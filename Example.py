#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mini demo: ChatGPT brand mention correlation
- Loads a small brand list
- Generates prompts
- (Stub) fills a few sample responses
- Labels Mentioned (1/0)
- Summarizes by brand/category
- Checks whether brands have a Wikipedia page
- Runs chi-square test and phi effect size on Mentioned ~ HasWiki
"""

import time
import warnings
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
CSV_PATH = "/Users/erdinc/PycharmProjects/pythonProject3/New Project/brands.csv"  # update if needed
CSV_SEP  = ";"  # your file uses ';' as delimiter

# Throttle between Wikipedia requests (be nice)
WIKI_SLEEP_SECONDS = 0.3

# Silence noisy BeautifulSoup warnings inside wikipedia package
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

# Force English Wikipedia
wikipedia.set_lang("en")

# Common alias corrections to resolve disambiguation
WIKI_ALIASES = {
    "HP": ["HP Inc.", "Hewlett-Packard"],
    "Apple": ["Apple Inc."],
    "Dell": ["Dell (company)", "Dell Inc."],
    "Lenovo": ["Lenovo"],
    "Asus": ["Asus"],
    "Jabra": ["Jabra", "Jabra (company)"],
    "Samsung": ["Samsung Electronics", "Samsung"]
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def has_wikipedia_page(brand: str):
    """
    Return (flag, chosen_title):
        flag = 1 if a resolvable Wikipedia page is found for the brand; else 0
        chosen_title = the resolved page title (or None)
    """
    # 1) Try direct titles (aliases first)
    titles_to_try = list(dict.fromkeys(WIKI_ALIASES.get(brand, [brand]) + [brand]))
    for title in titles_to_try:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            return 1, page.title
        except DisambiguationError as e:
            # Try a few likely candidates from the disambiguation list
            # Prioritize candidates that include the brand name
            candidates = sorted(
                e.options,
                key=lambda x: (brand.lower() not in x.lower(), len(x))
            )
            for cand in candidates[:3]:
                try:
                    page = wikipedia.page(cand, auto_suggest=False)
                    return 1, page.title
                except Exception:
                    continue
        except PageError:
            continue
        except Exception:
            continue

    # 2) Fallback to search
    try:
        hits = wikipedia.search(brand)
        for cand in hits[:5]:
            try:
                page = wikipedia.page(cand, auto_suggest=False)
                return 1, page.title
            except Exception:
                continue
    except Exception:
        pass

    return 0, None


def build_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """Create a simple yes/no prompt: 'Is {Brand} a good {Category} brand?'"""
    df = df.copy()
    df["Prompt"] = "Is " + df["Brand"] + " a good " + df["Category"] + " brand?"
    return df


def seed_sample_responses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stub a few sample responses just to demonstrate the pipeline.
    Everything else left blank intentionally.
    """
    df = df.copy()
    df["Response"] = ""
    # Seed the first few rows with toy responses (same as your original)
    seed = [
        "Yes, Apple is one of the most popular laptop brands.",
        "Yes, Dell laptops are known for reliability.",
        "HP is a solid laptop brand with many models.",
    ]
    for i, text in enumerate(seed):
        if i < len(df):
            df.at[i, "Response"] = text
    return df


def label_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mentioned = 1 if brand name appears in Response (case-insensitive),
    else 0. Empty response -> force 0.
    """
    df = df.copy()
    df["Response"] = df["Response"].fillna("").astype(str)
    df["ResponseEmpty"] = df["Response"].str.strip().eq("")
    df["Mentioned"] = df.apply(
        lambda row: 1 if (row["Brand"].lower() in row["Response"].lower()) else 0,
        axis=1
    )
    df["Mentioned"] = np.where(df["ResponseEmpty"], 0, df["Mentioned"])
    return df


def summarize(df: pd.DataFrame):
    """Print brand- and category-level summaries."""
    by_brand = (
        df.groupby(["Category", "Brand"], as_index=False)
          .agg(
              prompts=("Prompt", "count"),
              responses_nonempty=("ResponseEmpty", lambda s: (~s).sum()),
              mentions=("Mentioned", "sum")
          )
    )
    by_brand["mention_rate"] = np.where(
        by_brand["responses_nonempty"] > 0,
        by_brand["mentions"] / by_brand["responses_nonempty"],
        0.0
    )
    print(by_brand.sort_values(["Category", "mention_rate"], ascending=[True, False]).head(20))

    by_cat = (
        df.groupby("Category", as_index=False)
          .agg(
              prompts=("Prompt", "count"),
              responses_nonempty=("ResponseEmpty", lambda s: (~s).sum()),
              mentions=("Mentioned", "sum")
          )
    )
    by_cat["mention_rate"] = np.where(
        by_cat["responses_nonempty"] > 0,
        by_cat["mentions"] / by_cat["responses_nonempty"],
        0.0
    )
    print("\nCategory summary:")
    print(by_cat.sort_values("mention_rate", ascending=False))


def compute_haswiki_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Query Wikipedia once per unique brand; add HasWiki and WikiTitle to df."""
    df = df.copy()
    brands = sorted(df["Brand"].dropna().unique())
    wiki_flag, wiki_title = {}, {}

    for b in brands:
        has, title = has_wikipedia_page(b)
        wiki_flag[b] = has
        wiki_title[b] = title
        print(f"{b:12} -> HasWiki={has}  ({title})")
        time.sleep(WIKI_SLEEP_SECONDS)

    df["HasWiki"] = df["Brand"].map(wiki_flag).fillna(0).astype(int)
    df["WikiTitle"] = df["Brand"].map(wiki_title)
    return df


def chi_square_on_haswiki(df: pd.DataFrame):
    """
    Run chi-square of independence on Mentioned (0/1) ~ HasWiki (0/1).
    Also compute phi effect size.
    """
    tab = pd.crosstab(df["HasWiki"], df["Mentioned"])
    # Guard against degenerate tables
    if tab.size == 0 or tab.values.sum() == 0:
        chi2, p, dof = 0.0, 1.0, 0
        phi = 0.0
    else:
        chi2, p, dof, _ = chi2_contingency(tab)
        n = len(df)
        phi = sqrt(chi2 / n) if n > 0 else float("nan")

    print("\n--- Contingency Table (HasWiki x Mentioned) ---")
    print(tab)
    print(f"\nChi-square: {chi2:.4f} | p-value: {p:.4g} | dof: {dof}")
    print(f"Phi coefficient (effect size): {phi:.4f}")

    rates = df.groupby("HasWiki")["Mentioned"].mean()
    print("\nMention rate by HasWiki:")
    print(rates)

    print("\nInterpretation:")
    if p < 0.05:
        print("- There is a statistically significant association between having a Wikipedia page and being mentioned.")
    else:
        print("- No statistically significant association detected at the 0.05 level.")
    print("- As a rule of thumb for phi: ~0.1 small, ~0.3 medium, ~0.5 large.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # 1) Load brands
    df = pd.read_csv(CSV_PATH, sep=CSV_SEP)
    print(df.head())
    print("\nShape:", df.shape)

    # 2) Build prompts
    df = build_prompts(df)
    print(df.head())

    # 3) Seed a few sample responses (demo only)
    df = seed_sample_responses(df)
    print(df.head(6))

    # 4) Label mentions
    df = label_mentions(df)
    print(df[["Brand", "Response", "Mentioned"]].head(6))

    # 5) Summaries
    summarize(df)

    # 6) Wikipedia flags
    df = compute_haswiki_flags(df)

    # 7) Chi-square + phi
    chi_square_on_haswiki(df)


if __name__ == "__main__":
    main()