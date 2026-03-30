"""
BERT [CLS] embedding extraction for news headlines.

Uses frozen bert-base-uncased to produce 768-dim sentence embeddings.
Smart strategy: extracts embeddings for UNIQUE titles only, then maps
back to all dataset rows. Caches results to disk.
"""

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from src.utils import OUTPUT_DIR, logger
from src.feature_engineering import clean_text


# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 64
DEFAULT_BATCH_SIZE = 32


# ─── Core extraction ─────────────────────────────────────────────────────────

def _load_model():
    """Load tokenizer and model once."""
    logger.info(f"Loading BERT model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()  # freeze
    for param in model.parameters():
        param.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"  BERT loaded on device: {device}")
    return tokenizer, model, device


def extract_embeddings(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache_path: str | None = None,
) -> np.ndarray:
    """
    Extract [CLS] embeddings for a list of texts.

    Args:
        texts: list of headline strings
        batch_size: number of texts per forward pass
        cache_path: if given, save/load .npy from this path

    Returns:
        numpy array of shape (N, 768)
    """
    # Check cache
    if cache_path and os.path.exists(cache_path):
        logger.info(f"  Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    tokenizer, model, device = _load_model()

    # Clean texts
    texts = [clean_text(t) for t in texts]

    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size),
                  total=n_batches, desc="BERT embedding"):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] token is at position 0
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    logger.info(f"  Extracted embeddings shape: {embeddings.shape}")

    # Cache to disk
    if cache_path:
        np.save(cache_path, embeddings)
        logger.info(f"  Cached embeddings to {cache_path}")

    return embeddings


# ─── Public API ───────────────────────────────────────────────────────────────

def get_unique_title_embeddings(
    df: pd.DataFrame,
    split: str = "train",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Extract BERT embeddings efficiently by processing only UNIQUE titles
    and then mapping back to all rows in the dataframe.

    This is dramatically faster than processing every row:
    e.g., 20k unique titles vs 5.8M rows.

    Args:
        df: dataframe containing 'title' and 'news_id' columns
        split: 'train' or 'val' (used for cache filename)
        batch_size: BERT batch size

    Returns:
        numpy array of shape (len(df), 768) — one embedding per row
    """
    cache_path = os.path.join(OUTPUT_DIR, f"embeddings_unique_{split}.npy")
    mapping_path = os.path.join(OUTPUT_DIR, f"embeddings_newsids_{split}.npy")
    row_emb_path = os.path.join(OUTPUT_DIR, f"embeddings_rows_{split}.npy")

    # Check if we already have per-row embeddings cached
    if os.path.exists(row_emb_path):
        emb = np.load(row_emb_path)
        if len(emb) == len(df):
            logger.info(f"  Loading cached per-row embeddings from {row_emb_path}")
            return emb
        else:
            logger.info(f"  Cached row embeddings have wrong size ({len(emb)} != {len(df)}), recomputing")

    logger.info(f"\n  BERT embedding strategy: unique titles → map to rows ({split})")

    # Get unique titles
    unique_news = df[["news_id", "title"]].drop_duplicates(subset="news_id")
    unique_titles = unique_news["title"].fillna("").tolist()
    unique_news_ids = unique_news["news_id"].tolist()
    logger.info(f"  Unique titles to embed: {len(unique_titles):,} (vs {len(df):,} total rows)")

    # Extract embeddings for unique titles
    unique_embeddings = extract_embeddings(unique_titles, batch_size, cache_path=cache_path)

    # Build news_id → embedding lookup
    emb_dict = {nid: unique_embeddings[i] for i, nid in enumerate(unique_news_ids)}

    # Map embeddings to all rows
    logger.info("  Mapping embeddings to all rows …")
    row_embeddings = np.zeros((len(df), unique_embeddings.shape[1]), dtype=np.float32)
    news_ids = df["news_id"].values
    for i, nid in enumerate(news_ids):
        if nid in emb_dict:
            row_embeddings[i] = emb_dict[nid]

    # Cache per-row embeddings
    np.save(row_emb_path, row_embeddings)
    logger.info(f"  Saved per-row embeddings: {row_embeddings.shape} → {row_emb_path}")

    return row_embeddings
