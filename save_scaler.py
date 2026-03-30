import argparse
import sys
import os
import joblib
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import logger, OUTPUT_DIR
from src.data_loader import load_and_build_datasets
from main import stratified_sample, DEFAULT_SAMPLE_SIZE
from src.feature_engineering import build_feature_matrix

def save_scaler():
    """Reconstruct datasets with the same seed to fit and save the exact scaler."""
    logger.info("Reconstructing datasets to save the StandardScaler...")
    train_df_full, val_df_full = load_and_build_datasets()
    
    train_df = stratified_sample(train_df_full, DEFAULT_SAMPLE_SIZE)
    val_df = stratified_sample(val_df_full, DEFAULT_SAMPLE_SIZE // 2)
    
    del train_df_full, val_df_full
    
    logger.info("Fitting scaler...")
    X_train_feat, y_train, X_val_feat, y_val, scaler, feature_names = build_feature_matrix(
        train_df, val_df
    )
    
    scaler_path = os.path.join(OUTPUT_DIR, "feature_scaler.joblib")
    joblib.dump({"scaler": scaler, "feature_names": feature_names}, scaler_path)
    logger.info(f"Successfully saved scaler to {scaler_path}")

if __name__ == "__main__":
    save_scaler()
