"""
Baseline model: TF-IDF + Logistic Regression for CTR prediction.

Provides a comparison point for the BERT-based models.
"""

import os
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.utils import OUTPUT_DIR, logger, evaluate_model, print_metrics
from src.feature_engineering import clean_text
from src.data_loader import load_and_build_datasets


# ─── Configuration ────────────────────────────────────────────────────────────

TFIDF_MAX_FEATURES = 10000
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "baseline_model.joblib")


# ─── Public API ───────────────────────────────────────────────────────────────

def run_baseline(train_df=None, val_df=None) -> dict:
    """
    Train TF-IDF + Logistic Regression baseline and evaluate.

    Returns:
        metrics dict with AUC-ROC, Log Loss, F1, Precision, Recall
    """
    if train_df is None or val_df is None:
        train_df, val_df = load_and_build_datasets()

    logger.info("\n" + "=" * 60)
    logger.info("  BASELINE MODEL: TF-IDF + Logistic Regression")
    logger.info("=" * 60)

    # Clean titles
    train_titles = train_df["title"].fillna("").apply(clean_text).tolist()
    val_titles = val_df["title"].fillna("").apply(clean_text).tolist()

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    # TF-IDF
    logger.info(f"  Fitting TF-IDF vectorizer (max_features={TFIDF_MAX_FEATURES}) …")
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train_titles)
    X_val = tfidf.transform(val_titles)
    logger.info(f"  TF-IDF shape: train={X_train.shape}, val={X_val.shape}")

    # Logistic Regression
    logger.info("  Training Logistic Regression (class_weight='balanced') …")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    lr.fit(X_train, y_train)

    # Predict
    y_prob = lr.predict_proba(X_val)[:, 1]
    metrics = evaluate_model(y_val, y_prob)
    print_metrics(metrics, "TF-IDF + Logistic Regression (Baseline)")

    # Save model
    joblib.dump({"tfidf": tfidf, "model": lr}, MODEL_SAVE_PATH)
    logger.info(f"  Saved baseline model to {MODEL_SAVE_PATH}")

    return metrics, y_prob
