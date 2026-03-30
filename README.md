# News Headline Click-Through Rate Prediction

Predict the probability that a user will click on a news headline using **BERT embeddings** and the **MIND-small dataset** from Microsoft News.

## Project Overview

News platforms show dozens of headlines but users only click a handful. This project trains a model to predict click-through probability by understanding headline semantics — not just keywords. It uses frozen BERT (`bert-base-uncased`) for [CLS] embeddings combined with hand-crafted features, and benchmarks against a TF-IDF baseline.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py

# 3. OR run baseline only (skip BERT — much faster)
python main.py --skip-bert
```

## Dataset

**MIND-small** — a 50,000-user subset of Microsoft's MIND news recommendation corpus.

| File | Description |
|------|-------------|
| `behaviors.tsv` | User impression logs with click labels |
| `news.tsv` | Article metadata (title, category, entities) |

## Pipeline Steps

| Step | Description | Output |
|------|-------------|--------|
| 1 | Load & parse MIND-small TSV files | Flat dataframe |
| 2 | Exploratory Data Analysis | Plots in `outputs/` |
| 3 | Feature engineering | Supplementary feature matrix |
| 4 | TF-IDF + Logistic Regression baseline | Baseline AUC-ROC |
| 5 | BERT [CLS] embedding extraction | Cached `.npy` files |
| 6 | BERT + LR / MLP classifiers | Model comparison |
| 7 | Analysis & visualization | ROC curves, confusion matrices |

## Project Structure

```
newsctr/
├── data/               # MIND-small train/dev splits
├── outputs/            # Generated plots, cached embeddings, models
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── bert_embeddings.py
│   ├── baseline_model.py
│   ├── bert_classifier.py
│   ├── analysis.py
│   └── utils.py
├── main.py             # Full pipeline
├── requirements.txt
└── README.md
```

## Models

| Model | Strategy |
|-------|----------|
| **Baseline** | TF-IDF (10k features) + Logistic Regression |
| **BERT + LR** | Frozen BERT [CLS] (768-dim) + supplementary features → Logistic Regression |
| **BERT + MLP** | Frozen BERT [CLS] + features → 2-layer MLP (256→64→1) with dropout & batch norm |

## Key Metrics

- **AUC-ROC** (primary) — ranking quality, threshold-independent
- **Log Loss** — probability calibration
- **F1, Precision, Recall** — at threshold 0.5

## Technology Stack

Python 3.9+ • PyTorch • Hugging Face Transformers • scikit-learn • pandas • matplotlib • seaborn
