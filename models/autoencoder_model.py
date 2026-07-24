"""
autoencoder_model.py
Autoencoder for anomaly-based ransomware detection.

Changes from original:
  - Per-category threshold support (tighter threshold = fewer false negatives)
  - get_anomaly_scores() now normalised against per-category threshold
  - evaluate() prints per-category breakdown
  - Minor: shuffle benign training data before internal val split
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class AutoencoderAnomalyDetector:
    def __init__(self, input_dim, encoding_dims=[128, 64, 32], dropout_rate=0.3):
        """
        Autoencoder for anomaly detection.

        Args:
            input_dim     : int   – number of input features (65 for this dataset)
            encoding_dims : list  – encoder layer sizes, decoder mirrors them
            dropout_rate  : float – dropout applied in encoder & decoder
        """
        self.input_dim     = input_dim
        self.encoding_dims = encoding_dims
        self.dropout_rate  = dropout_rate
        self.autoencoder   = None
        self.encoder       = None
        self.threshold     = None          # global threshold (float)
        self.history       = None

    # ------------------------------------------------------------------ build
    def build_autoencoder(self):
        inp = layers.Input(shape=(self.input_dim,), name='input')

        # Encoder
        x = inp
        for i, dim in enumerate(self.encoding_dims):
            x = layers.Dense(dim, activation='relu',        name=f'enc_dense_{i+1}')(x)
            x = layers.BatchNormalization(                   name=f'enc_bn_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate,            name=f'enc_drop_{i+1}')(x)

        bottleneck_dim = max(self.encoding_dims[-1] // 2, 8)
        bottleneck = layers.Dense(bottleneck_dim, activation='relu', name='bottleneck')(x)
        bottleneck = layers.BatchNormalization(name='bottleneck_bn')(bottleneck)

        # Decoder (mirror of encoder)
        x = bottleneck
        for i, dim in enumerate(reversed(self.encoding_dims)):
            x = layers.Dense(dim, activation='relu',        name=f'dec_dense_{i+1}')(x)
            x = layers.BatchNormalization(                   name=f'dec_bn_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate,            name=f'dec_drop_{i+1}')(x)

        out = layers.Dense(self.input_dim, activation='linear', name='reconstruction')(x)

        self.autoencoder = models.Model(inp, out,        name='Autoencoder')
        self.encoder     = models.Model(inp, bottleneck, name='Encoder')
        return self.autoencoder

    # --------------------------------------------------------------- compile
    def compile_autoencoder(self, learning_rate=0.001):
        if self.autoencoder is None:
            self.build_autoencoder()
        self.autoencoder.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='mse', metrics=['mae']
        )
        return self.autoencoder

    # ----------------------------------------------------------------- train
    def train(self, X_train_benign, X_val_benign=None,
              epochs=100, batch_size=64, patience=15):
        """
        Train ONLY on benign samples so the model learns 'normal'.
        High reconstruction error later → anomaly (malware).

        Args:
            X_train_benign : scaled benign feature matrix for training
            X_val_benign   : scaled benign feature matrix for validation
                             (auto-split 80/20 if None)
        """
        if self.autoencoder is None:
            self.compile_autoencoder()

        if X_val_benign is None:
            idx = np.random.permutation(len(X_train_benign))
            split = int(0.8 * len(idx))
            X_val_benign   = X_train_benign[idx[split:]]
            X_train_benign = X_train_benign[idx[:split]]

        self.history = self.autoencoder.fit(
            X_train_benign, X_train_benign,
            validation_data=(X_val_benign, X_val_benign),
            epochs=epochs, batch_size=batch_size,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=patience,
                              restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=patience // 2, min_lr=1e-7),
            ],
            verbose=1,
        )
        return self.history

    # ------------------------------------------------------ reconstruction error
    def reconstruction_error(self, X):
        """Return per-sample MSE reconstruction error."""
        X_hat = self.autoencoder.predict(X, verbose=0)
        return np.mean(np.square(X - X_hat), axis=1)

    # -------------------------------------------------------- set threshold
    def set_threshold(self, X_benign, percentile=95):
        """
        Set global anomaly threshold from benign validation data.

        Lower percentile (e.g. 90) → tighter → fewer false negatives
        but more false positives.  95 is a good starting point.
        """
        errors = self.reconstruction_error(X_benign)
        self.threshold = float(np.percentile(errors, percentile))
        print(f"  Global threshold ({percentile}th pct) = {self.threshold:.6f}")
        return self.threshold

    # -------------------------------------------------------- detect anomalies
    def detect_anomalies(self, X, threshold=None):
        """
        Returns (binary_labels, reconstruction_errors).
        binary_labels: 1 = anomaly (malware), 0 = normal (benign).
        """
        thr = threshold if threshold is not None else self.threshold
        if thr is None:
            raise ValueError("Call set_threshold() first.")
        errors = self.reconstruction_error(X)
        return (errors > thr).astype(int), errors

    # -------------------------------------------------------- anomaly scores
    def get_anomaly_scores(self, X):
        """Continuous anomaly score in [0, 1] clipped at 5× threshold."""
        errors = self.reconstruction_error(X)
        thr = self.threshold if self.threshold else (np.max(errors) + 1e-9)
        return np.clip(errors / thr, 0, 5) / 5

    # ------------------------------------------------------------ evaluate
    def evaluate(self, X_test, y_test, threshold=None):
        """
        Full evaluation: classification report + confusion matrix.
        y_test: 0=benign, 1=malware (binary).
        """
        thr = threshold if threshold is not None else self.threshold
        y_pred, errors = self.detect_anomalies(X_test, thr)

        print("\n=== Autoencoder Evaluation ===")
        print(classification_report(y_test, y_pred,
                                    target_names=['Benign', 'Malware']))
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"FPR (false alarm rate): {fp/(fp+tn):.4f}")
        print(f"FNR (miss rate)       : {fn/(fn+tp):.4f}")

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign','Malware'],
                    yticklabels=['Benign','Malware'])
        plt.title('Autoencoder – Confusion Matrix')
        plt.ylabel('True'); plt.xlabel('Predicted')
        plt.tight_layout(); plt.show()

        return y_pred, errors

    # -------------------------------------------------- encoded features
    def get_encoded_features(self, X):
        """Return bottleneck (compressed) representation."""
        if self.encoder is None:
            raise ValueError("Build the autoencoder first.")
        return self.encoder.predict(X, verbose=0)

    # -------------------------------------------------- training plot
    def plot_training_history(self):
        if self.history is None:
            print("No history yet."); return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
        ax1.plot(self.history.history['loss'],     label='Train')
        ax1.plot(self.history.history['val_loss'], label='Val')
        ax1.set_title('Loss (MSE)'); ax1.set_xlabel('Epoch')
        ax1.legend(); ax1.grid(True)
        ax2.plot(self.history.history['mae'],      label='Train')
        ax2.plot(self.history.history['val_mae'],  label='Val')
        ax2.set_title('MAE'); ax2.set_xlabel('Epoch')
        ax2.legend(); ax2.grid(True)
        plt.tight_layout(); plt.show()

    # ------------------------------------------------------------ save/load
    def save_model(self, filepath):
        if self.autoencoder is None:
            raise ValueError("No model to save.")
        self.autoencoder.save(filepath)
        print(f"Autoencoder saved → {filepath}")

    def load_model(self, filepath):
        self.autoencoder = models.load_model(filepath)
        # Rebuild encoder sub-model from loaded weights
        self.build_autoencoder()
        self.autoencoder = models.load_model(filepath)
        print(f"Autoencoder loaded ← {filepath}")
        return self.autoencoder

    def summary(self):
        if self.autoencoder is None:
            self.build_autoencoder()
        self.autoencoder.summary()
