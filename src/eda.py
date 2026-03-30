"""
Exploratory Data Analysis for the MIND-small dataset.

Generates visualizations saved to the outputs/ directory:
  - Click label distribution
  - Category distribution
  - Headline word-length histogram
  - Click rate by category
  - Click rate by hour of day
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import OUTPUT_DIR, logger
from src.data_loader import load_and_build_datasets

# ─── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGSIZE = (10, 6)


def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved plot: {path}")


# ─── Plot functions ───────────────────────────────────────────────────────────

def plot_label_distribution(df: pd.DataFrame):
    """Bar chart of click vs. non-click counts."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["label"].value_counts().sort_index()
    bars = ax.bar(["Not Clicked (0)", "Clicked (1)"], counts.values,
                  color=["#4A90D9", "#E74C3C"], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                f"{val:,}", ha="center", fontweight="bold", fontsize=12)
    ax.set_title("Click Label Distribution", fontsize=15, fontweight="bold")
    ax.set_ylabel("Count")
    _save(fig, "eda_label_distribution.png")


def plot_category_distribution(df: pd.DataFrame):
    """Horizontal bar chart of article categories."""
    fig, ax = plt.subplots(figsize=(10, 8))
    cat_counts = df["category"].value_counts().head(18)
    cat_counts.sort_values().plot.barh(ax=ax, color="#4A90D9", edgecolor="white")
    ax.set_title("Top News Categories", fontsize=15, fontweight="bold")
    ax.set_xlabel("Count")
    _save(fig, "eda_category_distribution.png")


def plot_headline_length(df: pd.DataFrame):
    """Histogram of headline word counts."""
    df = df.copy()
    df["word_count"] = df["title"].fillna("").str.split().str.len()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(df["word_count"], bins=30, color="#2ECC71", edgecolor="white", alpha=0.85)
    ax.axvline(df["word_count"].median(), color="#E74C3C", linestyle="--",
               label=f"Median = {df['word_count'].median():.0f}")
    ax.set_title("Headline Word Count Distribution", fontsize=15, fontweight="bold")
    ax.set_xlabel("Number of words")
    ax.set_ylabel("Frequency")
    ax.legend()
    _save(fig, "eda_headline_length.png")


def plot_ctr_by_category(df: pd.DataFrame):
    """Click rate per category."""
    ctr = df.groupby("category")["label"].mean().sort_values(ascending=False).head(18)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ctr.plot.bar(ax=ax, color="#9B59B6", edgecolor="white")
    ax.set_title("Click-Through Rate by Category", fontsize=15, fontweight="bold")
    ax.set_ylabel("CTR (fraction clicked)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.axhline(df["label"].mean(), color="#E74C3C", linestyle="--",
               label=f"Overall CTR = {df['label'].mean():.2%}")
    ax.legend()
    _save(fig, "eda_ctr_by_category.png")


def plot_ctr_by_hour(df: pd.DataFrame):
    """Click rate by hour of day."""
    df = df.copy()
    df["hour"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p",
                                errors="coerce").dt.hour
    hourly = df.dropna(subset=["hour"]).groupby("hour")["label"].mean()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(hourly.index, hourly.values, marker="o", color="#E67E22", linewidth=2)
    ax.fill_between(hourly.index, hourly.values, alpha=0.15, color="#E67E22")
    ax.set_title("Click-Through Rate by Hour of Day", fontsize=15, fontweight="bold")
    ax.set_xlabel("Hour (0–23)")
    ax.set_ylabel("CTR")
    ax.set_xticks(range(0, 24))
    _save(fig, "eda_ctr_by_hour.png")


# ─── Public API ───────────────────────────────────────────────────────────────

def run_eda(train_df: pd.DataFrame = None):
    """Generate all EDA plots. If no dataframe given, loads dataset first."""
    if train_df is None:
        train_df, _ = load_and_build_datasets()

    logger.info("\n" + "=" * 60)
    logger.info("  EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    # Summary stats
    total = len(train_df)
    clicks = train_df["label"].sum()
    logger.info(f"  Total impression-article pairs : {total:,}")
    logger.info(f"  Clicks (label=1)               : {clicks:,}  ({clicks/total:.2%})")
    logger.info(f"  Non-clicks (label=0)            : {total - clicks:,}")
    logger.info(f"  Unique news articles            : {train_df['news_id'].nunique():,}")
    logger.info(f"  Unique users                    : {train_df['user_id'].nunique():,}")
    logger.info(f"  Categories                      : {train_df['category'].nunique()}")

    # Generate plots
    plot_label_distribution(train_df)
    plot_category_distribution(train_df)
    plot_headline_length(train_df)
    plot_ctr_by_category(train_df)
    plot_ctr_by_hour(train_df)

    logger.info("  EDA complete — all plots saved to outputs/\n")
