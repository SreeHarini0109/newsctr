"""
Feature engineering for the News CTR Prediction project.

Extracts hand-crafted text features, encodes categorical columns,
and builds the final supplementary feature matrix.
"""

import re
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import logger


# ─── Text cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags, remove special chars from headline."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML
    text = re.sub(r"[^a-zA-Z0-9\s\?\!\.,']", " ", text)  # keep basic punct
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


# ─── Hand-crafted features ───────────────────────────────────────────────────

def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute supplementary features from headline text and metadata.

    Returns a DataFrame with the same index as input, containing:
      word_count, char_count, has_number, has_question_mark,
      has_exclamation_mark, named_entity_count, impression_hour,
      impression_dow
    """
    feats = pd.DataFrame(index=df.index)
    title = df["title"].fillna("")

    feats["word_count"] = title.str.split().str.len().fillna(0).astype(int)
    feats["char_count"] = title.str.len().fillna(0).astype(int)
    feats["has_number"] = title.str.contains(r"\d", regex=True).astype(int)
    feats["has_question_mark"] = title.str.strip().str.endswith("?").astype(int)
    feats["has_exclamation_mark"] = title.str.strip().str.endswith("!").astype(int)

    # Named entity count from title_entities JSON
    def _entity_count(s):
        try:
            return len(json.loads(s))
        except (json.JSONDecodeError, TypeError):
            return 0

    feats["named_entity_count"] = df["title_entities"].apply(_entity_count)

    # Temporal features from impression time
    ts = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    feats["impression_hour"] = ts.dt.hour.fillna(12).astype(int)
    feats["impression_dow"] = ts.dt.dayofweek.fillna(0).astype(int)

    return feats


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode category and subcategory columns."""
    cat_dummies = pd.get_dummies(df["category"], prefix="cat")
    subcat_dummies = pd.get_dummies(df["subcategory"], prefix="subcat")
    return pd.concat([cat_dummies, subcat_dummies], axis=1).astype(int)


# ─── Public API ───────────────────────────────────────────────────────────────

def build_feature_matrix(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame = None,
) -> tuple:
    """
    Build the full supplementary feature matrix (no BERT embeddings).

    Returns:
        If val_df is None:  (X_train, y_train, scaler, feature_names)
        If val_df given:    (X_train, y_train, X_val, y_val, scaler, feature_names)
    """
    logger.info("Building supplementary feature matrix …")

    # --- Text features ---
    train_text = extract_text_features(train_df)

    # --- Category encoding ---
    # Fit on training categories, then align validation to same columns
    train_cat = encode_categories(train_df)

    # Combine
    X_train_raw = pd.concat([train_text, train_cat], axis=1).fillna(0)
    y_train = train_df["label"].values

    feature_names = list(X_train_raw.columns)

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw.values)

    logger.info(f"  Train features shape: {X_train.shape}")

    if val_df is not None:
        val_text = extract_text_features(val_df)
        val_cat = encode_categories(val_df)
        X_val_raw = pd.concat([val_text, val_cat], axis=1).fillna(0)

        # Align columns to training set
        for col in feature_names:
            if col not in X_val_raw.columns:
                X_val_raw[col] = 0
        X_val_raw = X_val_raw[feature_names]

        y_val = val_df["label"].values
        X_val = scaler.transform(X_val_raw.values)

        logger.info(f"  Val features shape:   {X_val.shape}")
        return X_train, y_train, X_val, y_val, scaler, feature_names

    return X_train, y_train, scaler, feature_names
