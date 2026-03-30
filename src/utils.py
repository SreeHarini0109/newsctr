"""
Utility functions and constants for the News CTR Prediction project.
"""

import os
import logging
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# ─── Project Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train", "MINDsmall_train")
DEV_DIR = os.path.join(DATA_DIR, "dev", "MINDsmall_dev")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logger(name: str = "newsctr", level=logging.INFO) -> logging.Logger:
    """Create a formatted console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

logger = setup_logger()

# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(y_true, y_prob, threshold: float = 0.5) -> dict:
    """
    Compute standard binary-classification metrics.

    Args:
        y_true:  ground-truth labels (0/1)
        y_prob:  predicted probabilities  P(click)
        threshold: decision boundary for hard predictions

    Returns:
        dict with AUC-ROC, Log Loss, F1, Precision, Recall
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    return metrics


def print_metrics(metrics: dict, model_name: str = "Model"):
    """Pretty-print a metrics dictionary."""
    logger.info(f"\n{'='*50}")
    logger.info(f"  {model_name} — Evaluation Results")
    logger.info(f"{'='*50}")
    for key, val in metrics.items():
        logger.info(f"  {key:>12s}: {val:.4f}")
    logger.info(f"{'='*50}\n")


def print_comparison_table(results: dict):
    """
    Print a side-by-side comparison of multiple models.

    Args:
        results: dict of {model_name: metrics_dict}
    """
    metric_names = ["auc_roc", "log_loss", "f1", "precision", "recall"]
    header = f"{'Model':<35s}" + "".join(f"{m:>12s}" for m in metric_names)
    logger.info(f"\n{'='*len(header)}")
    logger.info("  MODEL COMPARISON TABLE")
    logger.info(f"{'='*len(header)}")
    logger.info(header)
    logger.info("-" * len(header))
    for model_name, metrics in results.items():
        row = f"{model_name:<35s}"
        for m in metric_names:
            row += f"{metrics.get(m, 0):.4f}".rjust(12)
        logger.info(row)
    logger.info(f"{'='*len(header)}\n")
