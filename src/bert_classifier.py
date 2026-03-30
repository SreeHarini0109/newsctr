"""
BERT-based classifiers for CTR prediction.

Two classifiers that consume concatenated BERT [CLS] embeddings
and supplementary hand-crafted features:
  1. Logistic Regression (scikit-learn)
  2. Multi-Layer Perceptron (PyTorch)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from src.utils import OUTPUT_DIR, logger, evaluate_model, print_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BERT + Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════════

def train_bert_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """
    Train Logistic Regression on BERT+supplementary features.

    Returns (metrics_dict, y_prob_val)
    """
    logger.info("\n" + "=" * 60)
    logger.info("  BERT [CLS] + Logistic Regression")
    logger.info("=" * 60)
    logger.info(f"  Feature dimension: {X_train.shape[1]}")

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    logger.info("  Training …")
    lr.fit(X_train, y_train)

    y_prob = lr.predict_proba(X_val)[:, 1]
    metrics = evaluate_model(y_val, y_prob)
    print_metrics(metrics, "BERT [CLS] + Logistic Regression")

    return metrics, y_prob, lr


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BERT + MLP Classifier (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

class CTR_MLP(nn.Module):
    """Two-hidden-layer MLP for binary CTR classification."""

    def __init__(self, input_dim: int, hidden1: int = 256, hidden2: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_bert_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 5,
) -> tuple[dict, np.ndarray]:
    """
    Train PyTorch MLP on BERT+supplementary features with early stopping.

    Returns (metrics_dict, y_prob_val)
    """
    logger.info("\n" + "=" * 60)
    logger.info("  BERT [CLS] + MLP Classifier")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}  |  Input dim: {X_train.shape[1]}")

    # --- Class weighting ---
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(device)
    logger.info(f"  Class balance — pos: {pos_count:,}  neg: {neg_count:,}  pos_weight: {pos_weight.item():.2f}")

    # --- DataLoaders ---
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    # --- Model ---
    model = CTR_MLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_auc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        avg_loss = total_loss / len(train_dataset)

        # ---- Validate ----
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(y_batch.numpy())

        y_prob_val = np.concatenate(all_probs)
        y_true_val = np.concatenate(all_labels)
        val_metrics = evaluate_model(y_true_val, y_prob_val)
        val_auc = val_metrics["auc_roc"]
        scheduler.step(val_metrics["log_loss"])

        logger.info(
            f"  Epoch {epoch:2d}/{epochs}  —  "
            f"train_loss: {avg_loss:.4f}  |  val_AUC: {val_auc:.4f}  |  val_F1: {val_metrics['f1']:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # --- Restore best model and final evaluation ---
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    y_prob = np.concatenate(all_probs)
    metrics = evaluate_model(y_val, y_prob)
    print_metrics(metrics, "BERT [CLS] + MLP Classifier")

    # Save model checkpoint
    ckpt_path = os.path.join(OUTPUT_DIR, "mlp_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"  Saved MLP checkpoint to {ckpt_path}")

    return metrics, y_prob
