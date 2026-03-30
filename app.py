import os
import joblib
import pandas as pd
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template

from src.utils import OUTPUT_DIR, logger
from src.feature_engineering import clean_text
from src.bert_classifier import CTR_MLP

# We also need the extract_embeddings function from our BERT module
from src.bert_embeddings import extract_embeddings

app = Flask(__name__)

# --- MODEL LOADING LOGIC ---
logger.info("Initializing UI Backend with BERT + MLP Model (Best Overall)...")

# 1. Load the Scaler for the 221 extracted features
try:
    scaler_data = joblib.load(os.path.join(OUTPUT_DIR, "feature_scaler.joblib"))
    scaler = scaler_data["scaler"]
    feature_names = scaler_data["feature_names"]
    logger.info(f"Loaded feature scaler for {len(feature_names)} features.")
except Exception as e:
    logger.error(f"Failed to load scaler: {e}")
    scaler = None

# 2. Load the PyTorch MLP Model
device = torch.device("cpu")
try:
    # 768 (BERT dims) + len(feature_names) (221 supplementary features) = 989
    input_dim = 768 + (len(feature_names) if scaler else 221)
    
    mlp_model = CTR_MLP(input_dim=input_dim).to(device)
    mlp_model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "mlp_model.pt"), map_location=device))
    mlp_model.eval()
    logger.info("Loaded PyTorch MLP classifier successfully.")
except Exception as e:
    logger.error(f"Failed to load MLP model: {e}")
    mlp_model = None

# We use the BERT extractor directly from bert_embeddings.py which loads frozen BERT on first call.

@app.route("/")
def index():
    """Render the main UI page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to predict CTR probability using BERT + MLP."""
    if scaler is None or mlp_model is None:
        return jsonify({"error": "BERT+MLP Model not fully initialized."}), 500

    data = request.json
    headline = data.get("headline", "").strip()

    if not headline:
        return jsonify({"error": "Headline is required."}), 400

    try:
        cleaned_title = clean_text(headline)
        
        # 1. Extract BERT [CLS] embedding (Shape: 1, 768)
        logger.info("Extracting BERT embeddings...")
        bert_emb = extract_embeddings([cleaned_title], batch_size=1)
        
        # 2. Re-create the 221 supplementary features (mocking empty category for UI)
        # Note: We create a dummy dataframe with the exact columns the feature extractor expects internally
        
        # Basic parsing identical to pipeline
        word_count = len(cleaned_title.split())
        char_count = len(cleaned_title)
        has_number = int(any(char.isdigit() for char in cleaned_title))
        has_question = int("?" in headline)
        has_exclamation = int("!" in headline)
        
        # We need a 221-length vector matching `feature_names`. 
        # The easiest way is to build a zero-vector and populate known indices.
        raw_feat_vector = np.zeros((1, len(feature_names)))
        
        # Manually assign known stats based on our feature_names order
        # Our first 5 features are always the text stats
        raw_feat_vector[0][0] = word_count
        raw_feat_vector[0][1] = char_count
        raw_feat_vector[0][2] = has_number
        raw_feat_vector[0][3] = has_question
        raw_feat_vector[0][4] = has_exclamation
        # (The rest are entity counts and one-hot categories, left as 0 for generic input)

        # Scale features
        scaled_feat = scaler.transform(raw_feat_vector)
        
        # 3. Concatenate (BERT + Features) -> (1, 989)
        combined_feat = np.hstack([bert_emb, scaled_feat])
        
        # 4. Predict via PyTorch MLP
        tensor_feat = torch.FloatTensor(combined_feat).to(device)
        with torch.no_grad():
            logits = mlp_model(tensor_feat) # Shape: [1, 1]
            prob = torch.sigmoid(logits).item()

        return jsonify({
            "headline": headline,
            "probability": prob,
            "features": {
                "word_count": word_count,
                "char_count": char_count,
                "model_used": "BERT + PyTorch MLP"
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 5555))
    app.run(host="0.0.0.0", port=port)
