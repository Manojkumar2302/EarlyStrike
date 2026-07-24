"""
predict.py  –  Use trained models to classify a new PE file
================================================================
Usage:
    python predict.py --csv new_samples.csv

The CSV must have the same feature columns as the training dataset.
Outputs a prediction for each row: is_malware, confidence, category.
"""

import argparse, joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from train import (STATIC_COLS, BEHAVIORAL_COLS, HEX_COLS,
                   SEQ_FEAT, hex_to_int)
from autoencoder_model import AutoencoderAnomalyDetector
from cnn_bilstm_model  import CNNBiLSTM


def load_models():
    scaler       = joblib.load('outputs/scaler.pkl')
    le_cat       = joblib.load('outputs/label_encoder_category.pkl')
    ae_model     = tf.keras.models.load_model('outputs/autoencoder_malware.keras')
    cnn_bin_k    = tf.keras.models.load_model('outputs/cnn_bilstm_binary.keras')
    cnn_cat_k    = tf.keras.models.load_model('outputs/cnn_bilstm_category.keras')
    return scaler, le_cat, ae_model, cnn_bin_k, cnn_cat_k


def preprocess(df, scaler):
    for col in HEX_COLS:
        if col in df.columns:
            df[col] = df[col].apply(hex_to_int)
    all_cols = STATIC_COLS + BEHAVIORAL_COLS
    X_flat = df[all_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    X_scaled = scaler.transform(X_flat)

    n_s = len(STATIC_COLS)
    X_s = np.hstack([X_scaled[:, :n_s],
                     np.zeros((len(X_scaled), SEQ_FEAT - n_s))])
    X_b = np.hstack([X_scaled[:, n_s:],
                     np.zeros((len(X_scaled), SEQ_FEAT - len(BEHAVIORAL_COLS)))])
    X_seq = np.stack([X_s, X_b], axis=1)
    return X_scaled, X_seq


def predict(csv_path):
    df = pd.read_csv(csv_path)
    scaler, le_cat, ae_model, cnn_bin_k, cnn_cat_k = load_models()

    X_scaled, X_seq = preprocess(df, scaler)

    # Reconstruction error
    X_hat  = ae_model.predict(X_scaled, verbose=0)
    ae_err = np.mean(np.square(X_scaled - X_hat), axis=1)
    # Normalise: score 0-1 (capped at 5× mean error)
    ae_score = np.clip(ae_err / (ae_err.mean() * 5 + 1e-9), 0, 1)

    # CNN-BiLSTM binary
    cnn_bin_proba = cnn_bin_k.predict(X_seq, verbose=0).flatten()
    # CNN-BiLSTM category
    cnn_cat_proba = cnn_cat_k.predict(X_seq, verbose=0)
    cat_pred_idx  = np.argmax(cnn_cat_proba, axis=1)
    cat_pred_name = le_cat.inverse_transform(cat_pred_idx)
    cat_confidence = cnn_cat_proba.max(axis=1)

    # Ensemble decision
    is_malware = (ae_score > 0.5) | (cnn_bin_proba >= 0.6)

    results = pd.DataFrame({
        'is_malware'       : is_malware,
        'ae_anomaly_score' : ae_score.round(4),
        'cnn_malware_prob' : cnn_bin_proba.round(4),
        'predicted_category': cat_pred_name,
        'category_confidence': cat_confidence.round(4),
    })
    if 'md5' in df.columns:
        results.insert(0, 'md5', df['md5'])

    print(results.to_string())
    results.to_csv('outputs/predictions.csv', index=False)
    print("\nSaved → outputs/predictions.csv")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True,
                        help='Path to CSV with PE features')
    args = parser.parse_args()
    predict(args.csv)
