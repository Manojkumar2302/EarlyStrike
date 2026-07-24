import pandas as pd
import numpy as np
from cnn_bilstm_model import CNNBiLSTM
from data_preprocessing import SecurityDataPreprocessor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
import re
import warnings
warnings.filterwarnings('ignore')

class RansomwareDataPreprocessor(SecurityDataPreprocessor):
    """Enhanced preprocessor specifically for ransomware detection"""
    
    def __init__(self):
        super().__init__()
        self.ransomware_indicators = {
            'encryption_extensions': ['.encrypted', '.locked', '.crypted', '.crypto', '.rsa'],
            'ransomware_processes': ['crypt', 'encrypt', 'lock', 'decrypt', 'wannacry', 'notpetya'],
            'suspicious_paths': ['desktop', 'documents', 'pictures', 'music', 'videos'],
            'crypto_file_types': ['.doc', '.docx', '.pdf', '.jpg', '.png', '.mp3', '.mp4', '.zip', '.rar']
        }
    
    def extract_ransomware_features(self, df):
        """Extract ransomware-specific features"""
        
        # Ransomware process indicators
        if 'process_n_pid' in df.columns:
            df['is_ransomware_process'] = df['process_n_pid'].astype(str).apply(
                lambda x: any(indicator in x.lower() for indicator in self.ransomware_indicators['ransomware_processes'])
            ).astype(int)
        
        # File encryption indicators
        if 'path' in df.columns:
            df['has_encryption_extension'] = df['path'].astype(str).apply(
                lambda x: any(x.lower().endswith(ext) for ext in self.ransomware_indicators['encryption_extensions'])
            ).astype(int)
            
            df['targets_user_files'] = df['path'].astype(str).apply(
                lambda x: any(path in x.lower() for path in self.ransomware_indicators['suspicious_paths'])
            ).astype(int)
            
            df['is_crypto_file_type'] = df['path'].astype(str).apply(
                lambda x: any(x.lower().endswith(ext) for ext in self.ransomware_indicators['crypto_file_types'])
            ).astype(int)
        
        # High entropy indicator (potential encryption)
        if 'entropy' in df.columns:
            df['high_entropy'] = (df['entropy'] > 7.0).astype(int)
            df['very_high_entropy'] = (df['entropy'] > 7.5).astype(int)
        
        # CPU usage patterns (ransomware often uses high CPU)
        if 'cpu_usage' in df.columns:
            df['high_cpu_usage'] = (df['cpu_usage'] > 80).astype(int)
            df['extreme_cpu_usage'] = (df['cpu_usage'] > 95).astype(int)
        
        # File permission changes (ransomware modifies permissions)
        if 'permission' in df.columns:
            df['permission_modified'] = df['permission'].astype(str).apply(
                lambda x: x != '644' and x != '755'  # Unusual permissions
            ).astype(int)
        
        # Multiple file operations in short time
        df['file_operation_burst'] = 0
        if 'syscall' in df.columns:
            write_operations = ['write', 'create', 'modify', 'encrypt']
            df['is_write_operation'] = df['syscall'].astype(str).apply(
                lambda x: any(op in x.lower() for op in write_operations)
            ).astype(int)
        
        return df
    
    def create_ransomware_sequences(self, df, sequence_length=None):
        """Create sequences optimized for ransomware detection patterns"""
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        # Separate features and target
        feature_cols = [col for col in df.columns if col != self.target_column]
        X = df[feature_cols].values
        y = df[self.target_column].values
        
        sequences = []
        targets = []
        
        # Create overlapping sequences to capture rapid ransomware behavior
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i + sequence_length])
            targets.append(y[i + sequence_length - 1])
        
        return np.array(sequences), np.array(targets)
    
    def preprocess_ransomware_data(self, df, sequence_length=None):
        """Complete preprocessing pipeline for ransomware detection"""
        print("Starting ransomware-specific data preprocessing...")
        
        df_processed = df.copy()
        
        # Apply standard preprocessing
        df_processed = self.preprocess_timestamp(df_processed)
        df_processed = self.preprocess_path(df_processed)
        df_processed = self.preprocess_permission(df_processed)
        df_processed = self.preprocess_syscall(df_processed)
        
        # Extract ransomware-specific features
        print("Extracting ransomware indicators...")
        df_processed = self.extract_ransomware_features(df_processed)
        
        # Handle missing values and scaling
        df_processed = self.handle_missing_values(df_processed)
        df_processed = self.scale_features(df_processed)
        
        print(f"Final ransomware dataset shape: {df_processed.shape}")
        print(f"Ransomware features: {[col for col in df_processed.columns if 'ransomware' in col or 'crypto' in col or 'encrypt' in col]}")
        
        return df_processed

class RansomwareDetector:
    """Specialized CNN-BiLSTM model for ransomware detection"""
    
    def __init__(self, sequence_length=30, cnn_filters=[128, 256], lstm_units=[256, 128], dropout_rate=0.4):
        """
        Initialize Ransomware Detector
        
        Args:
            sequence_length: int, shorter sequences for faster ransomware detection
            cnn_filters: list, more filters for complex pattern detection
            lstm_units: list, larger units for sequence learning
            dropout_rate: float, higher dropout for better generalization
        """
        self.sequence_length = sequence_length
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.preprocessor = RansomwareDataPreprocessor()
        self.model = None
        self.training_history = None
        
        # Ransomware detection thresholds
        self.detection_threshold = 0.5
        self.high_risk_threshold = 0.8
        
    def prepare_ransomware_data(self, data_path):
        """Prepare data specifically for ransomware detection"""
        print("Loading and preprocessing ransomware data...")
        
        df = self.preprocessor.load_data(data_path)
        if df is None:
            raise ValueError("Failed to load ransomware data")
        
        # Preprocess with ransomware-specific features
        df_processed = self.preprocessor.preprocess_ransomware_data(df, self.sequence_length)
        
        # Create sequences
        X, y = self.preprocessor.create_ransomware_sequences(df_processed, self.sequence_length)
        
        # Split data with stratification to maintain ransomware ratio
        from sklearn.model_selection import train_test_split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        print(f"Ransomware data prepared:")
        print(f"  Training: {X_train.shape}, Ransomware rate: {np.mean(y_train):.3f}")
        print(f"  Validation: {X_val.shape}, Ransomware rate: {np.mean(y_val):.3f}")
        print(f"  Test: {X_test.shape}, Ransomware rate: {np.mean(y_test):.3f}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_ransomware_model(self, input_shape):
        """Build CNN-BiLSTM optimized for ransomware detection"""
        print("Building ransomware detection model...")
        
        self.model = CNNBiLSTM(
            input_shape=input_shape,
            num_classes=2,
            cnn_filters=self.cnn_filters,
            lstm_units=self.lstm_units,
            dropout_rate=self.dropout_rate
        )
        
        # Compile with appropriate metrics for ransomware detection
        self.model.build_model()
        self.model.compile_model(learning_rate=0.0005)  # Lower learning rate for stability
        
        print("Ransomware detection model architecture:")
        self.model.get_model_summary()
        
        return self.model
    
    def train_ransomware_detector(self, epochs=150, batch_size=64, patience=20):
        """Train model with ransomware-specific considerations"""
        if self.model is None:
            raise ValueError("Model not built. Call build_ransomware_model() first.")
        
        print("Training ransomware detection model...")
        
        # Use class weights if imbalanced
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weight_dict = dict(enumerate(class_weights))
        
        # Train with class weights
        self.training_history = self.model.train(
            self.X_train, self.y_train,
            X_val=self.X_val, y_val=self.y_val,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        
        return self.training_history
    
    def detect_ransomware(self, data_path, threshold=None):
        """
        Detect ransomware in new data with detailed analysis
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        if threshold is None:
            threshold = self.detection_threshold
        
        print(f"Detecting ransomware with threshold: {threshold}")
        
        # Load and preprocess data
        df = self.preprocessor.load_data(data_path)
        df_processed = self.preprocessor.preprocess_ransomware_data(df)
        
        # Create sequences
        X_new, y_new = self.preprocessor.create_ransomware_sequences(df_processed, self.sequence_length)
        
        # Make predictions
        y_pred_proba = self.model.predict_proba(X_new)
        y_pred = (y_pred_proba > threshold).astype(int)
        
        # Detailed analysis
        self._analyze_ransomware_predictions(y_pred, y_pred_proba, df_processed)
        
        return y_pred, y_pred_proba
    
    def _analyze_ransomware_predictions(self, y_pred, y_pred_proba, df_processed):
        """Analyze ransomware detection results"""
        
        # Detection statistics
        total_sequences = len(y_pred)
        ransomware_detected = np.sum(y_pred)
        high_risk_sequences = np.sum(y_pred_proba > self.high_risk_threshold)
        
        print(f"\n=== RANSOMWARE DETECTION RESULTS ===")
        print(f"Total sequences analyzed: {total_sequences}")
        print(f"Ransomware detected: {ransomware_detected} ({ransomware_detected/total_sequences:.1%})")
        print(f"High-risk sequences: {high_risk_sequences} ({high_risk_sequences/total_sequences:.1%})")
        
        # Risk distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(y_pred_proba, bins=50, alpha=0.7, color='red', edgecolor='black')
        plt.axvline(x=self.detection_threshold, color='blue', linestyle='--', label=f'Detection Threshold: {self.detection_threshold}')
        plt.axvline(x=self.high_risk_threshold, color='darkred', linestyle='--', label=f'High Risk: {self.high_risk_threshold}')
        plt.xlabel('Ransomware Probability')
        plt.ylabel('Frequency')
        plt.title('Ransomware Risk Distribution')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        risk_levels = ['Normal', 'Suspicious', 'High Risk']
        risk_counts = [
            np.sum(y_pred_proba <= self.detection_threshold),
            np.sum((y_pred_proba > self.detection_threshold) & (y_pred_proba <= self.high_risk_threshold)),
            np.sum(y_pred_proba > self.high_risk_threshold)
        ]
        colors = ['green', 'orange', 'red']
        plt.pie(risk_counts, labels=risk_levels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Risk Level Distribution')
        
        plt.subplot(2, 2, 3)
        # Time series of predictions
        plt.plot(y_pred_proba, 'r-', alpha=0.7, label='Ransomware Probability')
        plt.axhline(y=self.detection_threshold, color='blue', linestyle='--', alpha=0.5)
        plt.axhline(y=self.high_risk_threshold, color='darkred', linestyle='--', alpha=0.5)
        plt.xlabel('Sequence Index')
        plt.ylabel('Ransomware Probability')
        plt.title('Ransomware Detection Timeline')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        # Feature importance for detected ransomware
        if hasattr(df_processed, 'columns'):
            ransomware_features = [col for col in df_processed.columns if any(keyword in col.lower() for keyword in ['ransomware', 'crypto', 'encrypt', 'high'])]
            if ransomware_features:
                feature_means = df_processed[ransomware_features].mean()
                feature_means.sort_values(ascending=False).plot(kind='bar', color='darkred')
                plt.title('Ransomware Indicator Activation')
                plt.ylabel('Average Value')
                plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_ransomware_detection(self):
        """Comprehensive evaluation for ransomware detection"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        print("Evaluating ransomware detection performance...")
        
        # Standard evaluation
        y_pred, y_pred_proba = self.model.evaluate(self.X_test, self.y_test)
        
        # Ransomware-specific metrics
        self._calculate_ransomware_metrics(self.y_test, y_pred, y_pred_proba)
        
        return y_pred, y_pred_proba
    
    def _calculate_ransomware_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate ransomware-specific metrics"""
        
        # Confusion matrix analysis
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Ransomware detection metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"\n=== RANSOMWARE DETECTION METRICS ===")
        print(f"True Positives: {tp} (Correctly detected ransomware)")
        print(f"False Positives: {fp} (Normal flagged as ransomware)")
        print(f"True Negatives: {tn} (Correctly identified normal)")
        print(f"False Negatives: {fn} (Missed ransomware)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        print(f"False Positive Rate: {false_positive_rate:.4f}")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkred', lw=2, label=f'Ransomware ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Ransomware Detection ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
        
        return precision, recall, f1_score, roc_auc

# Complete ransomware detection pipeline
def train_ransomware_detector(data_path, sequence_length=30):
    """
    Complete pipeline for training ransomware detection system
    """
    print("=== INITIATING RANSOMWARE DETECTION SYSTEM ===")
    
    # Initialize detector
    detector = RansomwareDetector(sequence_length=sequence_length)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = detector.prepare_ransomware_data(data_path)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    detector.build_ransomware_model(input_shape)
    
    # Train model
    detector.train_ransomware_detector(epochs=150, batch_size=64, patience=20)
    
    # Plot training history
    detector.model.plot_training_history()
    
    # Evaluate performance
    detector.evaluate_ransomware_detection()
    
    # Save model
    detector.model.save_model("ransomware_detection_model.h5")
    
    print("\n=== RANSOMWARE DETECTION SYSTEM READY ===")
    return detector

# Example usage
if __name__ == "__main__":
    # Create sample ransomware dataset
    print("Creating sample ransomware dataset...")
    
    # Simulate ransomware behavior patterns
    n_samples = 3000
    sample_data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='10s'),
        'process_n_pid': np.random.choice(['cryptolocker', 'encrypt_process', 'normal_process', 'system'], n_samples),
        'ppid': np.random.randint(1, 100, n_samples),
        'syscall': np.random.choice(['encrypt', 'write', 'read', 'delete', 'modify', 'create'], n_samples),
        'path': np.random.choice([
            '/desktop/documents/file.docx.encrypted',
            '/pictures/photo.jpg.locked',
            '/tmp/temp_file',
            '/usr/bin/normal_file',
            '/documents/important.pdf.crypted'
        ], n_samples),
        'inode': np.random.randint(100000, 999999, n_samples),
        'permission': np.random.choice(['644', '755', '000', '444', '777'], n_samples),
        'uid': np.random.randint(0, 1000, n_samples),
        'gid': np.random.randint(0, 1000, n_samples),
        'exit': np.random.randint(0, 2, n_samples),
        'entropy': np.concatenate([
            np.random.uniform(7.5, 8.0, n_samples//3),  # High entropy (ransomware)
            np.random.uniform(0, 6, 2*n_samples//3)    # Normal entropy
        ]),
        'cpu_usage': np.concatenate([
            np.random.uniform(80, 100, n_samples//3),  # High CPU (ransomware)
            np.random.uniform(0, 50, 2*n_samples//3)    # Normal CPU
        ]),
        'label': np.concatenate([
            np.ones(n_samples//3),    # Ransomware
            np.zeros(2*n_samples//3)   # Normal
        ])
    }
    
    df = pd.DataFrame(sample_data)
    data_path = "ransomware_sample_data.csv"
    df.to_csv(data_path, index=False)
    
    # Train ransomware detection system
    detector = train_ransomware_detector(data_path, sequence_length=30)
    
    print("Ransomware Detection System Training Completed!")
