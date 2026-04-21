"""
text_data_demo.py
-----------------
Part 2: Sample financial news dataset creation and exploratory data analysis.

This module demonstrates how to import, structure, and analyse financial
news/text data in Python using a small, manually created representative sample.
The same workflow applies to any CSV-exported news dataset — simply replace
create_sample_news_df() with pd.read_csv('your_news_file.csv').
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# ── Sample data ───────────────────────────────────────────────────────────────

_SAMPLE = {
    "date": [
        "2023-01-10", "2023-01-11", "2023-01-12", "2023-01-13", "2023-01-16",
        "2023-01-17", "2023-01-18", "2023-01-19", "2023-01-20", "2023-01-23",
        "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27", "2023-01-30",
    ],
    "headline": [
        "Central bank holds rates steady amid inflation concerns",
        "Mining sector reports strong quarterly earnings growth",
        "Political uncertainty weighs on Chilean equity markets",
        "Copper prices surge on strong China demand outlook",
        "Peso weakens against dollar as investors seek safety",
        "New infrastructure spending plan announced by government",
        "Unemployment rises slightly in latest monthly report",
        "Major bank posts record profits for the fiscal year",
        "Trade deficit widens in December data release",
        "Equity market reaches three-month high on optimism",
        "Pension reform debate continues in national congress",
        "Retail sales growth beats analyst expectations",
        "Mining union threatens strike action over wage dispute",
        "GDP growth revised upward by international analysts",
        "Foreign investment inflows increase significantly in Q4",
    ],
    "source": [
        "Reuters", "Bloomberg", "Reuters", "Reuters", "Bloomberg",
        "FT", "Reuters", "Bloomberg", "Reuters", "Bloomberg",
        "Reuters", "Bloomberg", "Reuters", "FT", "Reuters",
    ],
    "sentiment_label": [
        "neutral",  "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "neutral",  "positive", "negative", "positive", "positive",
    ],
}

# Stop-words to exclude from word-frequency analysis
_STOPWORDS = {
    "in", "on", "of", "the", "a", "for", "by", "as", "to", "at",
    "is", "and", "with", "its", "amid", "over", "new", "an", "by",
}


# ── Data creation ─────────────────────────────────────────────────────────────

def create_sample_news_df() -> pd.DataFrame:
    """
    Build and return the sample financial news DataFrame.

    The dataset contains 15 headlines covering Chilean market events
    in January 2023, with manual sentiment labels (positive / negative / neutral).
    A word_count column is added automatically.

    Returns
    -------
    pd.DataFrame
        Columns: date, headline, source, sentiment_label, word_count.
    """
    df = pd.DataFrame(_SAMPLE)
    df["date"]       = pd.to_datetime(df["date"])
    df["word_count"] = df["headline"].apply(lambda x: len(x.split()))
    return df


def save_sample_news_csv(df: pd.DataFrame,
                          path: str = "data/sample_news_data.csv") -> None:
    """
    Save the sample news DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
    path : str
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# ── EDA ───────────────────────────────────────────────────────────────────────

def run_news_eda(df: pd.DataFrame) -> dict:
    """
    Compute basic exploratory statistics for the news DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with keys:
        sentiment_counts, source_counts, top_words,
        total_records, avg_word_count.
    """
    sent_counts = df["sentiment_label"].value_counts()
    src_counts  = df["source"].value_counts()

    all_words = " ".join(df["headline"].str.lower()).split()
    filtered  = [w.strip(".,") for w in all_words
                 if w.strip(".,") not in _STOPWORDS and len(w) > 2]
    top_words = Counter(filtered).most_common(10)

    return {
        "sentiment_counts": sent_counts,
        "source_counts":    src_counts,
        "top_words":        top_words,
        "total_records":    len(df),
        "avg_word_count":   round(df["word_count"].mean(), 1),
    }


def plot_news_eda(df: pd.DataFrame,
                  save_path: str = "outputs/figure3_part2_eda.png") -> None:
    """
    Generate a two-panel EDA chart:
        Left  — Sentiment label distribution (bar chart)
        Right — Top-10 word frequency in headlines (horizontal bar chart)

    Figure is saved to save_path as a PNG.

    Parameters
    ----------
    df : pd.DataFrame
    save_path : str
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    eda          = run_news_eda(df)
    sent_counts  = eda["sentiment_counts"]
    top_words    = eda["top_words"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 1 — Sentiment distribution
    colour_map  = {"positive": "steelblue", "negative": "tomato", "neutral": "silver"}
    bar_colours = [colour_map.get(s, "gray") for s in sent_counts.index]
    axes[0].bar(sent_counts.index, sent_counts.values,
                color=bar_colours, edgecolor="black", linewidth=0.6)
    axes[0].set_title("Sentiment Label Distribution", fontsize=11)
    axes[0].set_xlabel("Sentiment Label", fontsize=9)
    axes[0].set_ylabel("Count", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2 — Top-10 words
    words, cnts = zip(*top_words)
    axes[1].barh(list(words), list(cnts),
                 color="steelblue", edgecolor="black", linewidth=0.5)
    axes[1].set_title("Top 10 Words in Headlines", fontsize=11)
    axes[1].set_xlabel("Frequency", fontsize=9)
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.3)

    plt.suptitle(
        "Part 2 — Exploratory Analysis: Sample Financial News Data (January 2023)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
