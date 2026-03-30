"""
Analysis and visualization module for model comparison.

Generates:
  - Overlaid ROC curves for all models
  - Confusion matrices
  - Model comparison table
  - Feature importance (from Logistic Regression)
  - Error analysis (sampled false positives / negatives)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

from src.utils import OUTPUT_DIR, logger, print_comparison_table


# ─── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {"Baseline (TF-IDF + LR)": "#3498DB",
          "BERT + LR": "#2ECC71",
          "BERT + MLP": "#E74C3C"}


def _save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path}")


# ─── ROC Curves ───────────────────────────────────────────────────────────────

def plot_roc_curves(y_true: np.ndarray, predictions: dict):
    """
    Plot overlaid ROC curves.

    Args:
        y_true: ground-truth labels
        predictions: dict of {model_name: y_prob}
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, y_prob in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        color = COLORS.get(name, "#95A5A6")
        ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves — Model Comparison", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    _save(fig, "roc_curves.png")


# ─── Confusion Matrices ──────────────────────────────────────────────────────

def plot_confusion_matrices(y_true: np.ndarray, predictions: dict, threshold: float = 0.5):
    """Plot side-by-side confusion matrices for all models."""
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, y_prob) in zip(axes, predictions.items()):
        y_pred = (np.asarray(y_prob) >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                    xticklabels=["Not Clicked", "Clicked"],
                    yticklabels=["Not Clicked", "Clicked"])
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices — Model Comparison", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "confusion_matrices.png")


# ─── Feature Importance ──────────────────────────────────────────────────────

def plot_feature_importance(lr_model, feature_names: list, top_k: int = 20):
    """Plot top-K feature importances from LR coefficients."""
    coefs = lr_model.coef_[0]

    # Get top positive and negative features
    indices = np.argsort(np.abs(coefs))[::-1][:top_k]
    top_names = [feature_names[i] if i < len(feature_names) else f"bert_{i}" for i in indices]
    top_coefs = coefs[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2ECC71" if c > 0 else "#E74C3C" for c in top_coefs]
    ax.barh(range(len(top_names)), top_coefs, color=colors, edgecolor="white")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_k} Feature Importances (LR Coefficients)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Coefficient Value")
    _save(fig, "feature_importance.png")


# ─── Error Analysis ──────────────────────────────────────────────────────────

def run_error_analysis(
    val_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Best Model",
    n_samples: int = 10,
    threshold: float = 0.5,
):
    """Print sampled false positives and false negatives for manual inspection."""
    y_pred = (y_prob >= threshold).astype(int)
    val = val_df.copy().reset_index(drop=True)
    val["y_true"] = y_true
    val["y_prob"] = y_prob
    val["y_pred"] = y_pred

    fp = val[(val["y_pred"] == 1) & (val["y_true"] == 0)]
    fn = val[(val["y_pred"] == 0) & (val["y_true"] == 1)]

    logger.info(f"\n{'='*60}")
    logger.info(f"  ERROR ANALYSIS — {model_name}")
    logger.info(f"{'='*60}")

    logger.info(f"\n  FALSE POSITIVES (predicted click, actual no-click): {len(fp):,}")
    for _, row in fp.sample(min(n_samples, len(fp)), random_state=42).iterrows():
        logger.info(f"  P={row['y_prob']:.3f}  |  [{row['category']}]  {row['title']}")

    logger.info(f"\n  FALSE NEGATIVES (predicted no-click, actual click): {len(fn):,}")
    for _, row in fn.sample(min(n_samples, len(fn)), random_state=42).iterrows():
        logger.info(f"  P={row['y_prob']:.3f}  |  [{row['category']}]  {row['title']}")


# ─── Prediction Distribution ─────────────────────────────────────────────────

def plot_prediction_distribution(y_true: np.ndarray, y_prob: np.ndarray, model_name: str):
    """Histogram of predicted probabilities split by true label."""
    fig, ax = plt.subplots(figsize=(10, 6))
    mask_pos = y_true == 1
    ax.hist(y_prob[~mask_pos], bins=50, alpha=0.6, color="#3498DB", label="Not Clicked (0)", density=True)
    ax.hist(y_prob[mask_pos], bins=50, alpha=0.6, color="#E74C3C", label="Clicked (1)", density=True)
    ax.set_title(f"Prediction Distribution — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted P(click)")
    ax.set_ylabel("Density")
    ax.legend()
    _save(fig, f"pred_distribution_{model_name.lower().replace(' ', '_')}.png")


# ─── Public API ───────────────────────────────────────────────────────────────

def run_full_analysis(
    y_val: np.ndarray,
    predictions: dict,
    all_metrics: dict,
    val_df: pd.DataFrame = None,
    lr_model=None,
    feature_names: list = None,
):
    """
    Run the complete analysis suite.

    Args:
        y_val: ground-truth validation labels
        predictions: dict of {model_name: y_prob_array}
        all_metrics: dict of {model_name: metrics_dict}
        val_df: validation dataframe (for error analysis)
        lr_model: fitted LR model (for feature importance)
        feature_names: list of feature names matching LR input
    """
    logger.info("\n" + "=" * 60)
    logger.info("  ANALYSIS & VISUALIZATION")
    logger.info("=" * 60)

    # Comparison table
    print_comparison_table(all_metrics)

    # ROC curves
    plot_roc_curves(y_val, predictions)

    # Confusion matrices
    plot_confusion_matrices(y_val, predictions)

    # Prediction distributions for best model
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["auc_roc"])
    plot_prediction_distribution(y_val, predictions[best_name], best_name)

    # Feature importance
    if lr_model is not None and feature_names is not None:
        plot_feature_importance(lr_model, feature_names)

    # Error analysis for best model
    if val_df is not None:
        run_error_analysis(val_df, y_val, predictions[best_name], best_name)

    logger.info("\n  Analysis complete — all outputs saved to outputs/\n")
