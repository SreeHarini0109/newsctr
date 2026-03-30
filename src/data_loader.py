"""
Data loading and preprocessing for the MIND-small dataset.

Parses news.tsv and behaviors.tsv, expands impressions into individual rows,
and joins them into a flat dataframe ready for feature engineering.
"""

import os
import pandas as pd
from src.utils import TRAIN_DIR, DEV_DIR, logger


# ─── Column names (MIND TSV files have no header) ────────────────────────────

NEWS_COLS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIOR_COLS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_news(data_dir: str) -> pd.DataFrame:
    """Load and parse news.tsv from a MIND split directory."""
    path = os.path.join(data_dir, "news.tsv")
    logger.info(f"Loading news from {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=NEWS_COLS,
        usecols=range(len(NEWS_COLS)),
        quoting=3,  # QUOTE_NONE — avoids issues with quotes inside text
    )
    # Fill missing abstracts / entities
    df["abstract"] = df["abstract"].fillna("")
    df["title_entities"] = df["title_entities"].fillna("[]")
    df["abstract_entities"] = df["abstract_entities"].fillna("[]")

    logger.info(f"  Loaded {len(df):,} news articles  |  categories: {df['category'].nunique()}")
    return df


def load_behaviors(data_dir: str) -> pd.DataFrame:
    """
    Load behaviors.tsv and expand the Impressions column into individual rows.

    Each row in the output represents one (impression, news_id, label) triple.
    """
    path = os.path.join(data_dir, "behaviors.tsv")
    logger.info(f"Loading behaviors from {path}")

    raw = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=BEHAVIOR_COLS,
        quoting=3,
    )
    raw["history"] = raw["history"].fillna("")

    logger.info(f"  Raw behavior rows: {len(raw):,}")

    # Expand impressions — vectorized approach (much faster than iterrows)
    # Each entry is like "N12345-1" or "N12345-0"
    raw = raw.dropna(subset=["impressions"])
    raw = raw.assign(impressions=raw["impressions"].str.split())
    expanded = raw[["impression_id", "user_id", "time", "impressions"]].explode("impressions")
    expanded = expanded.rename(columns={"impressions": "token"})

    # Split "N12345-1" into news_id and label
    split = expanded["token"].str.rsplit("-", n=1, expand=True)
    expanded["news_id"] = split[0]
    expanded["label"] = split[1].astype(int)
    expanded = expanded.drop(columns=["token"]).reset_index(drop=True)
    logger.info(
        f"  Expanded to {len(expanded):,} impression-article pairs  |  "
        f"click rate: {expanded['label'].mean():.2%}"
    )
    return expanded


def build_dataset(news_df: pd.DataFrame, behaviors_df: pd.DataFrame) -> pd.DataFrame:
    """Join news metadata onto expanded behavior rows."""
    df = behaviors_df.merge(news_df, on="news_id", how="inner")
    logger.info(f"  Merged dataset: {len(df):,} rows, {df.columns.tolist()}")
    return df


# ─── Public API ───────────────────────────────────────────────────────────────

def load_and_build_datasets():
    """
    Full pipeline: load both splits, expand, join, return (train_df, val_df).
    """
    logger.info("=" * 60)
    logger.info("  LOADING MIND-small DATASET")
    logger.info("=" * 60)

    # --- Train ---
    train_news = load_news(TRAIN_DIR)
    train_behaviors = load_behaviors(TRAIN_DIR)
    train_df = build_dataset(train_news, train_behaviors)

    # --- Validation ---
    val_news = load_news(DEV_DIR)
    val_behaviors = load_behaviors(DEV_DIR)
    val_df = build_dataset(val_news, val_behaviors)

    logger.info(f"\n  FINAL  Train: {len(train_df):,} rows  |  Val: {len(val_df):,} rows")
    return train_df, val_df
