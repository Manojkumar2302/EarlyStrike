"""
train.py  –  Full training pipeline for ransomware detection
================================================================
Dataset : Final_Dataset_without_duplicate.csv
Models  : (1) XGBoost baseline
          (2) Autoencoder  (anomaly detection – trained on benign only)
          (3) CNN-BiLSTM   (binary: Benign vs Malware)
          (4) CNN-BiLSTM   (multi-class: Benign/Ransomware/RAT/Stealer/Trojan)
          (5) Ensemble     (autoencoder score + CNN-BiLSTM probabilities)
          (6) SHAP         (feature importance explainability)

Run:
    pip install tensorflow scikit-learn xgboost shap matplotlib seaborn joblib
    python train.py

Outputs (saved in ./outputs/):
    scaler.pkl, label_encoder_category.pkl
    autoencoder_malware.keras
    cnn_bilstm_binary.keras
    cnn_bilstm_category.keras
    xgb_baseline.json
    shap_summary.png
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.metrics          import classification_report, confusion_matrix

# ── local model files (must be in the same folder as this script) ──────────
from autoencoder_model import AutoencoderAnomalyDetector
from cnn_bilstm_model  import CNNBiLSTM

os.makedirs('outputs', exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – LOAD & PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════

STATIC_COLS = [
    'EntryPoint', 'bytes_on_last_page', 'pages_in_file', 'relocations',
    'size_of_header', 'min_extra_paragraphs', 'max_extra_paragraphs',
    'init_ss_value', 'init_sp_value', 'init_ip_value', 'init_cs_value',
    'over_lay_number', 'oem_identifier', 'address_of_ne_header', 'Magic',
    'SizeOfCode', 'SizeOfInitializedData', 'SizeOfUninitializedData',
    'AddressOfEntryPoint', 'BaseOfCode', 'BaseOfData', 'ImageBase',
    'SectionAlignment', 'FileAlignment', 'OperatingSystemVersion',
    'ImageVersion', 'SizeOfImage', 'SizeOfHeaders', 'Checksum', 'Subsystem',
    'SizeofStackReserve', 'SizeofStackCommit', 'SizeofHeapCommit',
    'SizeofHeapReserve', 'LoaderFlags',
    'text_VirtualSize', 'text_VirtualAddress', 'text_SizeOfRawData',
    'text_PointerToRawData', 'text_PointerToRelocations',
    'text_PointerToLineNumbers',
    'rdata_VirtualSize', 'rdata_VirtualAddress', 'rdata_SizeOfRawData',
    'rdata_PointerToRawData', 'rdata_PointerToRelocations',
    'rdata_PointerToLineNumbers',
]   # 47 columns

BEHAVIORAL_COLS = [
    'registry_read', 'registry_write', 'registry_delete', 'registry_total',
    'network_threats', 'network_dns', 'network_http', 'network_connections',
    'processes_malicious', 'processes_suspicious', 'processes_monitored',
    'total_procsses', 'files_malicious', 'files_suspicious', 'files_text',
    'files_unknown', 'dlls_calls', 'apis',
]   # 18 columns

# Hex-string columns that need int conversion
HEX_COLS = [
    'EntryPoint', 'bytes_on_last_page', 'pages_in_file', 'relocations',
    'size_of_header', 'min_extra_paragraphs', 'max_extra_paragraphs',
    'init_ss_value', 'init_sp_value', 'init_ip_value', 'init_cs_value',
    'over_lay_number', 'oem_identifier', 'address_of_ne_header',
    'SizeOfCode', 'SizeOfInitializedData', 'SizeOfUninitializedData',
    'AddressOfEntryPoint', 'BaseOfCode', 'BaseOfData', 'ImageBase',
    'SectionAlignment', 'FileAlignment', 'SizeOfImage', 'SizeOfHeaders',
    'Checksum', 'SizeofStackReserve', 'SizeofStackCommit',
    'SizeofHeapCommit', 'SizeofHeapReserve', 'LoaderFlags',
    'text_VirtualSize', 'text_VirtualAddress', 'text_SizeOfRawData',
    'text_PointerToRawData', 'text_PointerToRelocations',
    'text_PointerToLineNumbers',
    'rdata_VirtualSize', 'rdata_VirtualAddress', 'rdata_SizeOfRawData',
    'rdata_PointerToRawData', 'rdata_PointerToRelocations',
    'rdata_PointerToLineNumbers',
]


def hex_to_int(val):
    try:
        s = str(val).strip().split(' ')[0]   # drop "(Section: .text)" suffix
        return int(s, 16) if s.startswith('0x') else float(s)
    except Exception:
        return 0.0


def load_dataset(csv_path: str):
    print("=" * 65)
    print("STEP 1 – Loading dataset")
    print("=" * 65)
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")

    # Convert hex strings
    for col in HEX_COLS:
        df[col] = df[col].apply(hex_to_int)

    # Build flat feature matrix (65 features total)
    all_feature_cols = STATIC_COLS + BEHAVIORAL_COLS
    X_flat = df[all_feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values

    # Labels
    y_binary   = (df['Class'] == 'Malware').astype(int).values   # 0/1
    le_cat     = LabelEncoder()
    y_category = le_cat.fit_transform(df['Category'])             # 0-4
    le_fam     = LabelEncoder()
    y_family   = le_fam.fit_transform(df['Family'])               # 0-26

    cat_names = list(le_cat.classes_)
    fam_names = list(le_fam.classes_)

    print(f"  Binary    : {dict(zip(['Benign','Malware'], np.bincount(y_binary)))}")
    print(f"  Categories: {dict(zip(cat_names, np.bincount(y_category)))}")
    print(f"  Families  : {len(fam_names)} unique ({fam_names[:5]} ...)")

    # Save encoders
    joblib.dump(le_cat, 'outputs/label_encoder_category.pkl')
    joblib.dump(le_fam, 'outputs/label_encoder_family.pkl')

    return X_flat, y_binary, y_category, y_family, cat_names, fam_names, df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – FEATURE ENGINEERING
#   - Scale flat features for the Autoencoder
#   - Build 2-step sequence for CNN-BiLSTM:
#       step 0 = scaled static features   (padded to SEQ_FEAT dims)
#       step 1 = scaled behavioral features (padded to SEQ_FEAT dims)
# ══════════════════════════════════════════════════════════════════════════════

SEQ_FEAT = 48   # max(47, 18) rounded up to even number → each step has 48 dims


def build_features(X_flat):
    """
    Returns:
        X_flat_scaled  : (n, 65)  – for Autoencoder + XGBoost
        X_seq          : (n, 2, 48) – for CNN-BiLSTM
        scaler         : fitted StandardScaler
    """
    print("\n" + "=" * 65)
    print("STEP 2 – Feature engineering")
    print("=" * 65)

    n_static = len(STATIC_COLS)      # 47
    n_behav  = len(BEHAVIORAL_COLS)  # 18

    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    joblib.dump(scaler, 'outputs/scaler.pkl')
    print(f"  Flat scaled shape : {X_flat_scaled.shape}")

    # Pad static (47→48) and behavioral (18→48) separately
    X_static = X_flat_scaled[:, :n_static]                         # (n, 47)
    X_behav  = X_flat_scaled[:, n_static:]                         # (n, 18)

    pad_s = SEQ_FEAT - n_static   # 1
    pad_b = SEQ_FEAT - n_behav    # 30
    X_static_pad = np.hstack([X_static, np.zeros((len(X_static), pad_s))])
    X_behav_pad  = np.hstack([X_behav,  np.zeros((len(X_behav),  pad_b))])

    # Stack into (n, 2, SEQ_FEAT) sequence
    X_seq = np.stack([X_static_pad, X_behav_pad], axis=1)
    print(f"  Sequence shape    : {X_seq.shape}  "
          f"(step-0=static, step-1=behavioral)")

    return X_flat_scaled, X_seq, scaler


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – XGBOOST BASELINE
# ══════════════════════════════════════════════════════════════════════════════

def train_xgboost_baseline(X_train, y_train, X_test, y_test,
                           label_names, task='binary'):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  xgboost not installed – skipping baseline. "
              "  pip install xgboost"); return None

    print("\n" + "=" * 65)
    print(f"STEP 3 – XGBoost baseline  [{task}]")
    print("=" * 65)

    params = dict(
        n_estimators=400, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, n_jobs=-1,
    )
    if task != 'binary':
        params['objective'] = 'multi:softprob'
        params['num_class']  = len(label_names)

    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train,
            eval_set=[(X_test, y_test)], verbose=False)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_names))
    clf.save_model('outputs/xgb_baseline.json')
    print("  XGBoost saved → outputs/xgb_baseline.json")
    return clf


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – AUTOENCODER
# ══════════════════════════════════════════════════════════════════════════════

def train_autoencoder(X_flat_scaled, y_binary):
    print("\n" + "=" * 65)
    print("STEP 4 – Autoencoder  (anomaly detection)")
    print("=" * 65)

    # Train / test split – stratified
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_flat_scaled, y_binary,
        test_size=0.2, random_state=42, stratify=y_binary
    )

    # Autoencoder sees ONLY benign samples during training
    X_tr_benign = X_tr[y_tr == 0]
    print(f"  Training on {len(X_tr_benign):,} benign samples")
    print(f"  Testing  on {len(X_te):,} samples (benign + malware)")

    ae = AutoencoderAnomalyDetector(
        input_dim=X_flat_scaled.shape[1],
        encoding_dims=[128, 64, 32],
        dropout_rate=0.3,
    )
    ae.build_autoencoder()
    ae.compile_autoencoder(learning_rate=0.001)
    ae.summary()

    ae.train(X_tr_benign, epochs=100, batch_size=64, patience=15)
    ae.plot_training_history()

    # Set threshold on benign validation data (held out automatically)
    # Use 95th percentile – tighten to 90 if you want fewer missed malware
    ae.set_threshold(X_tr_benign, percentile=95)

    ae.evaluate(X_te, y_te)
    ae.save_model('outputs/autoencoder_malware.keras')

    return ae, X_tr, X_te, y_tr, y_te


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – CNN-BiLSTM  (binary + category)
# ══════════════════════════════════════════════════════════════════════════════

def train_cnn_bilstm(X_seq, y, num_classes, label_names,
                     save_path, task_name, epochs=50, batch_size=64):
    print("\n" + "=" * 65)
    print(f"STEP 5 – CNN-BiLSTM  [{task_name}]")
    print("=" * 65)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_seq, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")

    clf = CNNBiLSTM(
        input_shape=(2, SEQ_FEAT),
        num_classes=num_classes,
        cnn_filters=[64, 128],
        lstm_units=[128, 64],
        dropout_rate=0.4,
    )
    clf.build_model()
    clf.compile_model(learning_rate=0.001)
    clf.summary()

    clf.train(X_tr, y_tr, epochs=epochs, batch_size=batch_size, patience=10)
    clf.plot_training_history()
    clf.evaluate(X_te, y_te, label_names=label_names)
    clf.save_model(save_path)

    return clf, X_te, y_te


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – ENSEMBLE
#   Rule: flag as MALWARE if EITHER condition is true:
#     (a) autoencoder anomaly score > 0.5  (high reconstruction error)
#     (b) CNN-BiLSTM predicts malware with confidence >= threshold
#   Final category label comes from the multi-class CNN-BiLSTM.
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_predict(ae, cnn_binary, X_flat_scaled, X_seq,
                     ae_score_threshold=0.5,
                     cnn_confidence_threshold=0.6):
    """
    Returns:
        is_malware      : np.array bool
        category_labels : np.array int (from multi-class CNN – pass separately)
        ae_scores       : np.array float [0-1]
        cnn_proba       : np.array float [0-1]
    """
    ae_scores = ae.get_anomaly_scores(X_flat_scaled)
    cnn_proba = cnn_binary.predict_proba(X_seq).flatten()

    ae_flag  = ae_scores  > ae_score_threshold
    cnn_flag = cnn_proba  >= cnn_confidence_threshold

    is_malware = ae_flag | cnn_flag
    return is_malware, ae_scores, cnn_proba


def evaluate_ensemble(ae, cnn_binary, cnn_category,
                      X_flat_scaled, X_seq_test, y_binary_test,
                      y_category_test, cat_names):
    print("\n" + "=" * 65)
    print("STEP 6 – Ensemble evaluation")
    print("=" * 65)

    is_malware, ae_scores, cnn_proba = ensemble_predict(
        ae, cnn_binary, X_flat_scaled, X_seq_test
    )
    y_pred = is_malware.astype(int)

    print("=== Ensemble (Autoencoder OR CNN-BiLSTM) ===")
    print(classification_report(y_binary_test, y_pred,
                                target_names=['Benign','Malware']))

    # Category predictions (only meaningful for predicted malware)
    cat_pred = cnn_category.predict(X_seq_test)
    print("=== Category classification (all samples) ===")
    print(classification_report(y_category_test, cat_pred,
                                target_names=cat_names))

    # Visualise score distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    for label, mask, color in [
        ('Benign',  y_binary_test==0, 'steelblue'),
        ('Malware', y_binary_test==1, 'tomato'),
    ]:
        ax1.hist(ae_scores[mask], bins=40, alpha=0.6, label=label, color=color)
        ax2.hist(cnn_proba[mask], bins=40, alpha=0.6, label=label, color=color)
    ax1.axvline(0.5, color='k', ls='--', lw=1, label='threshold=0.5')
    ax2.axvline(0.6, color='k', ls='--', lw=1, label='threshold=0.6')
    ax1.set_title('Autoencoder anomaly score'); ax1.set_xlabel('Score')
    ax2.set_title('CNN-BiLSTM malware probability'); ax2.set_xlabel('P(malware)')
    for ax in (ax1, ax2): ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/ensemble_score_distribution.png', dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – SHAP EXPLAINABILITY  (on XGBoost for speed)
# ══════════════════════════════════════════════════════════════════════════════

def run_shap(xgb_model, X_test, feature_names):
    try:
        import shap
    except ImportError:
        print("  shap not installed – skipping. pip install shap")
        return

    print("\n" + "=" * 65)
    print("STEP 7 – SHAP feature importance")
    print("=" * 65)

    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test[:500])   # subsample for speed

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test[:500],
                      feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('outputs/shap_summary.png', dpi=150)
    plt.show()
    print("  SHAP plot saved → outputs/shap_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── 1. Load ───────────────────────────────────────────────────────────────
    CSV_PATH = 'Final_Dataset_without_duplicate.csv'
    (X_flat, y_binary, y_category, y_family,
     cat_names, fam_names, df) = load_dataset(CSV_PATH)

    # ── 2. Feature engineering ────────────────────────────────────────────────
    X_flat_scaled, X_seq, scaler = build_features(X_flat)
    feature_names = STATIC_COLS + BEHAVIORAL_COLS

    # ── 3. Train/test split for XGBoost baseline ──────────────────────────────
    X_tr_f, X_te_f, y_tr_b, y_te_b = train_test_split(
        X_flat_scaled, y_binary,
        test_size=0.2, random_state=42, stratify=y_binary
    )

    # ── 3. XGBoost baseline (binary) ──────────────────────────────────────────
    xgb = train_xgboost_baseline(
        X_tr_f, y_tr_b, X_te_f, y_te_b,
        label_names=['Benign','Malware'], task='binary'
    )

    # ── 4. Autoencoder ────────────────────────────────────────────────────────
    ae, X_tr_ae, X_te_ae, y_tr_ae, y_te_ae = train_autoencoder(
        X_flat_scaled, y_binary
    )

    # ── 5a. CNN-BiLSTM  binary ────────────────────────────────────────────────
    cnn_bin, X_te_seq_bin, y_te_bin = train_cnn_bilstm(
        X_seq, y_binary,
        num_classes=2,
        label_names=['Benign','Malware'],
        save_path='outputs/cnn_bilstm_binary.keras',
        task_name='binary',
    )

    # ── 5b. CNN-BiLSTM  category (5-class) ───────────────────────────────────
    cnn_cat, X_te_seq_cat, y_te_cat = train_cnn_bilstm(
        X_seq, y_category,
        num_classes=len(cat_names),
        label_names=cat_names,
        save_path='outputs/cnn_bilstm_category.keras',
        task_name='category (5-class)',
    )

    # ── 6. Ensemble ───────────────────────────────────────────────────────────
    # Use same test split for fair comparison
    _, X_seq_test, _, y_bin_test = train_test_split(
        X_seq, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    _, X_flat_test, _, y_cat_test = train_test_split(
        X_flat_scaled, y_category, test_size=0.2,
        random_state=42, stratify=y_category
    )

    evaluate_ensemble(
        ae, cnn_bin, cnn_cat,
        X_flat_test, X_seq_test,
        y_bin_test, y_cat_test,
        cat_names,
    )

    # ── 7. SHAP ───────────────────────────────────────────────────────────────
    if xgb is not None:
        run_shap(xgb, X_te_f, feature_names)

    print("\n" + "=" * 65)
    print("ALL DONE – outputs saved in ./outputs/")
    print("=" * 65)
