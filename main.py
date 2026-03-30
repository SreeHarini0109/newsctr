#!/usr/bin/env python3
"""
News Headline Click-Through Rate Prediction
============================================
Full pipeline: data loading → EDA → feature engineering →
BERT embeddings → baseline model → BERT classifiers → analysis.

Usage:
    python main.py              # run full pipeline
    python main.py --skip-bert  # skip BERT embedding (baseline only)
    python main.py --sample 500000  # use subset for faster training
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import logger, OUTPUT_DIR
from src.data_loader import load_and_build_datasets
from src.eda import run_eda
from src.feature_engineering import build_feature_matrix
from src.bert_embeddings import get_unique_title_embeddings
from src.baseline_model import run_baseline
from src.bert_classifier import train_bert_lr, train_bert_mlp
from src.analysis import run_full_analysis


DEFAULT_SAMPLE_SIZE = 500_000  # Stratified sample for manageable training


def stratified_sample(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    """Stratified sample keeping the natural class distribution."""
    if len(df) <= n:
        return df
    # Keep all positive samples, downsample negatives
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    target_pos = min(len(pos), int(n * len(pos) / len(df)))
    target_neg = n - target_pos
    sampled_pos = pos.sample(n=min(target_pos, len(pos)), random_state=random_state)
    sampled_neg = neg.sample(n=min(target_neg, len(neg)), random_state=random_state)
    result = pd.concat([sampled_pos, sampled_neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    logger.info(f"  Sampled {len(result):,} rows (pos: {len(sampled_pos):,}, neg: {len(sampled_neg):,})")
    return result


def main(skip_bert: bool = False, sample_size: int = DEFAULT_SAMPLE_SIZE):
    """Run the full CTR prediction pipeline."""

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  NEWS HEADLINE CTR PREDICTION PIPELINE                  ║")
    logger.info("║  MIND-small Dataset  •  BERT Embeddings                 ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # ── Step 1: Load data ─────────────────────────────────────────────────
    logger.info("\n▶ STEP 1/7 — Loading MIND-small dataset")
    train_df_full, val_df_full = load_and_build_datasets()

    # ── Step 2: EDA (on full data) ────────────────────────────────────────
    logger.info("\n▶ STEP 2/7 — Exploratory Data Analysis")
    run_eda(train_df_full)

    # ── Sampling for model training ───────────────────────────────────────
    logger.info(f"\n▶ Stratified sampling for model training (target: {sample_size:,})")
    train_df = stratified_sample(train_df_full, sample_size)
    val_df = stratified_sample(val_df_full, sample_size // 2)

    # Free memory from full datasets
    del train_df_full, val_df_full

    # ── Step 3: Feature engineering ───────────────────────────────────────
    logger.info("\n▶ STEP 3/7 — Feature Engineering")
    X_train_feat, y_train, X_val_feat, y_val, scaler, feature_names = build_feature_matrix(
        train_df, val_df
    )

    # ── Step 4: Baseline model ────────────────────────────────────────────
    logger.info("\n▶ STEP 4/7 — Baseline Model (TF-IDF + Logistic Regression)")
    baseline_metrics, baseline_probs = run_baseline(train_df, val_df)

    # Collect results
    all_metrics = {"Baseline (TF-IDF + LR)": baseline_metrics}
    all_predictions = {"Baseline (TF-IDF + LR)": baseline_probs}
    lr_model_for_importance = None
    combined_feature_names = feature_names

    if not skip_bert:
        # ── Step 5: BERT embeddings ───────────────────────────────────────
        logger.info("\n▶ STEP 5/7 — BERT Embedding Extraction")
        # Smart strategy: extract embeddings for UNIQUE titles only, then map to rows
        train_emb = get_unique_title_embeddings(train_df)
        val_emb = get_unique_title_embeddings(val_df, split="val")

        # Concatenate BERT embeddings with supplementary features
        X_train_combined = np.hstack([train_emb, X_train_feat])
        X_val_combined = np.hstack([val_emb, X_val_feat])
        logger.info(f"  Combined feature vector: {X_train_combined.shape[1]} dims "
                     f"(BERT: {train_emb.shape[1]} + Supplementary: {X_train_feat.shape[1]})")

        # Combined feature names for importance analysis
        bert_feature_names = [f"bert_{i}" for i in range(train_emb.shape[1])]
        combined_feature_names = bert_feature_names + feature_names

        # ── Step 6: BERT classifiers ──────────────────────────────────────
        logger.info("\n▶ STEP 6/7 — BERT-based Classifiers")

        # 6a. BERT + Logistic Regression
        bert_lr_metrics, bert_lr_probs, lr_model = train_bert_lr(
            X_train_combined, y_train, X_val_combined, y_val
        )
        all_metrics["BERT + LR"] = bert_lr_metrics
        all_predictions["BERT + LR"] = bert_lr_probs
        lr_model_for_importance = lr_model

        # 6b. BERT + MLP
        bert_mlp_metrics, bert_mlp_probs = train_bert_mlp(
            X_train_combined, y_train, X_val_combined, y_val
        )
        all_metrics["BERT + MLP"] = bert_mlp_metrics
        all_predictions["BERT + MLP"] = bert_mlp_probs
    else:
        logger.info("\n⏭  Skipping BERT steps (--skip-bert flag set)")

    # ── Step 7: Analysis & Visualization ──────────────────────────────────
    logger.info("\n▶ STEP 7/7 — Analysis & Visualization")
    run_full_analysis(
        y_val=y_val,
        predictions=all_predictions,
        all_metrics=all_metrics,
        val_df=val_df,
        lr_model=lr_model_for_importance,
        feature_names=combined_feature_names if lr_model_for_importance else None,
    )

    # ── Done ──────────────────────────────────────────────────────────────
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  PIPELINE COMPLETE                                      ║")
    logger.info(f"║  Outputs saved to: {OUTPUT_DIR:<37s}  ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News CTR Prediction Pipeline")
    parser.add_argument("--skip-bert", action="store_true",
                        help="Skip BERT embedding extraction (baseline only)")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help=f"Training sample size (default: {DEFAULT_SAMPLE_SIZE:,})")
    args = parser.parse_args()
    main(skip_bert=args.skip_bert, sample_size=args.sample)
