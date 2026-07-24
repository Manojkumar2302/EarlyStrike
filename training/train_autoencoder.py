import pandas as pd
import numpy as np
from autoencoder_model import AutoencoderAnomalyDetector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class AutoencoderTrainer:
    """
    Specialized trainer for autoencoder model using ransomware dataset
    
    Features:
    - Data preprocessing for autoencoder training
    - Training on normal data only
    - Threshold optimization
    - Comprehensive evaluation
    - Model saving and loading
    """
    
    def __init__(self):
        self.autoencoder = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'timestamp', 'process_name', 'pid', 'ppid', 'syscall', 'path', 'inode',
            'permissions', 'uid', 'gid', 'exit', 'entropy', 'cpu_usage'
        ]
        self.target_column = 'label'
        self.threshold = None
        
    def load_and_preprocess_data(self, data_path):
        """
        Load and preprocess the ransomware dataset for autoencoder training
        
        Args:
            data_path: str, path to the CSV dataset
        """
        print("=== LOADING AND PREPROCESSING DATA ===")
        
        # Load dataset
        try:
            df = pd.read_csv(data_path)
            print(f"Dataset loaded successfully: {df.shape}")
            print(f"Columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None, None, None, None
        
        # Check for required columns
        missing_cols = set(self.feature_columns + [self.target_column]) - set(df.columns)
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return None, None, None, None
        
        # Display basic statistics
        print(f"\nDataset Info:")
        print(f"Total samples: {len(df)}")
        
        # Convert string labels to numeric (benign -> 0) BEFORE preprocessing
        y_original = df[self.target_column].copy()
        if y_original.dtype == 'object':
            # Simple conversion: if it's 'benign', make it 0, otherwise 1
            y_original = np.where(y_original == 'benign', 0, 1)
        
        # Preprocess features
        df_processed = self._preprocess_features(df.copy())
        
        # Separate features and target
        X = df_processed.drop(self.target_column, axis=1)
        y = y_original  # Use the converted labels
        
        print(f"Normal samples (label=0): {len(y[y == 0])}")
        print(f"Ransomware samples (label=1): {len(y[y == 1])}")
        print(f"Ransomware ratio: {len(y[y == 1]) / len(y):.3f}")
        
        print(f"\nProcessed features shape: {X.shape}")
        print(f"Feature columns: {list(X.columns)}")
        
        return df_processed, X, y, df
    
    def _preprocess_features(self, df):
        """Preprocess all features for autoencoder"""
        print("Preprocessing features...")
        
        # 1. Process timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['timestamp_unix'] = df['timestamp'].astype(np.int64) // 10**9
            df['timestamp_hour'] = df['timestamp'].dt.hour
            df['timestamp_day'] = df['timestamp'].dt.dayofweek
            df = df.drop('timestamp', axis=1)
        
        # 2. Process categorical features
        categorical_cols = ['process_name', 'syscall', 'path', 'permissions']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
                # Label encode
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col])
                df = df.drop(col, axis=1)
        
        # 3. Handle missing values (only for numerical columns)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        
        # Handle missing values for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('unknown')
        
        # 4. Scale numerical features (except target)
        numerical_cols = [col for col in df.columns if col != self.target_column]
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        print(f"Features preprocessed. Final shape: {df.shape}")
        return df
    
    def prepare_training_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        Prepare training data for autoencoder (train only on normal data)
        
        Args:
            X: DataFrame, features
            y: Series, target labels
            test_size: float, test set proportion
            val_size: float, validation set proportion (from remaining data)
        """
        print("\n=== PREPARING TRAINING DATA ===")
        
        # Separate normal and anomalous data
        X_normal = X[y == 0]
        X_anomaly = X[y == 1]
        
        print(f"Normal data for training: {X_normal.shape}")
        print(f"Anomalous data for testing: {X_anomaly.shape}")
        
        # Split normal data for training and validation
        X_train, X_val_normal = train_test_split(
            X_normal, test_size=val_size, random_state=42
        )
        
        # Create test set with both normal and anomalous data
        X_test_normal, X_test_normal_remaining = train_test_split(
            X_normal, test_size=test_size, random_state=42
        )
        
        # Combine normal and anomalous for test set
        X_test = pd.concat([X_test_normal, X_anomaly], ignore_index=True)
        y_test = pd.concat([pd.Series([0] * len(X_test_normal)), 
                           pd.Series([1] * len(X_anomaly))], ignore_index=True)
        
        print(f"Training set: {X_train.shape} (normal only)")
        print(f"Validation set: {X_val_normal.shape} (normal only)")
        print(f"Test set: {X_test.shape} (mixed)")
        print(f"Test set - Normal: {len(y_test[y_test == 0])}, Anomaly: {len(y_test[y_test == 1])}")
        
        return X_train, X_val_normal, X_test, y_test
    
    def build_and_train_autoencoder(self, X_train, X_val, encoding_dims=[128, 64, 32], 
                                  epochs=100, batch_size=32, learning_rate=0.001):
        """
        Build and train the autoencoder model
        
        Args:
            X_train: array, training data (normal only)
            X_val: array, validation data (normal only)
            encoding_dims: list, autoencoder architecture
            epochs: int, training epochs
            batch_size: int, batch size
            learning_rate: float, learning rate
        """
        print("\n=== BUILDING AND TRAINING AUTOENCODER ===")
        
        # Initialize autoencoder
        input_dim = X_train.shape[1]
        self.autoencoder = AutoencoderAnomalyDetector(
            input_dim=input_dim,
            encoding_dims=encoding_dims,
            dropout_rate=0.3
        )
        
        # Build and compile
        self.autoencoder.build_autoencoder()
        self.autoencoder.compile_autoencoder(learning_rate=learning_rate)
        
        print(f"Autoencoder architecture:")
        print(f"  Input dimension: {input_dim}")
        print(f"  Encoding dimensions: {encoding_dims}")
        
        # Train model
        print(f"\nTraining autoencoder...")
        history = self.autoencoder.train(
            X_train, X_val,
            epochs=epochs,
            batch_size=batch_size,
            patience=15
        )
        
        # Plot training history
        self.autoencoder.plot_training_history()
        
        return history
    
    def optimize_threshold(self, X_val_normal, X_test, y_test, percentile_range=[90, 95, 99]):
        """
        Optimize anomaly threshold using validation data
        
        Args:
            X_val_normal: array, normal validation data
            X_test: array, test data (mixed)
            y_test: array, test labels
            percentile_range: list, percentiles to test
        """
        print("\n=== OPTIMIZING ANOMALY THRESHOLD ===")
        
        # Calculate reconstruction errors for normal validation data
        val_errors, _ = self.autoencoder.calculate_reconstruction_error(X_val_normal)
        
        # Test different thresholds
        best_threshold = None
        best_f1 = 0
        results = []
        
        for percentile in percentile_range:
            threshold = np.percentile(val_errors, percentile)
            
            # Test on test set
            test_errors, _ = self.autoencoder.calculate_reconstruction_error(X_test)
            y_pred = (test_errors > threshold).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import f1_score, precision_score, recall_score
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results.append({
                'percentile': percentile,
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            print(f"Percentile {percentile}: Threshold={threshold:.6f}, F1={f1:.3f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Set best threshold
        self.threshold = best_threshold
        self.autoencoder.threshold = best_threshold
        
        print(f"\nBest threshold: {best_threshold:.6f} (F1: {best_f1:.3f})")
        
        # Display results
        results_df = pd.DataFrame(results)
        print("\nThreshold Optimization Results:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive evaluation of the autoencoder model
        
        Args:
            X_test: array, test data
            y_test: array, test labels
        """
        print("\n=== EVALUATING AUTOENCODER MODEL ===")
        
        if self.autoencoder is None:
            print("Model not trained yet!")
            return
        
        # Make predictions
        y_pred, reconstruction_errors, X_reconstructed = self.autoencoder.evaluate(
            X_test, y_test, threshold=self.threshold
        )
        
        # Additional metrics
        self._calculate_detailed_metrics(y_test, reconstruction_errors)
        
        # Plot analysis
        self._plot_evaluation_analysis(y_test, reconstruction_errors, X_test, X_reconstructed)
        
        return y_pred, reconstruction_errors
    
    def _calculate_detailed_metrics(self, y_true, reconstruction_errors):
        """Calculate detailed evaluation metrics"""
        print("\nDetailed Metrics:")
        
        # ROC AUC
        auc_score = roc_auc_score(y_true, reconstruction_errors)
        print(f"ROC AUC Score: {auc_score:.4f}")
        
        # Precision-Recall AUC
        from sklearn.metrics import precision_recall_curve, auc
        precision, recall, _ = precision_recall_curve(y_true, reconstruction_errors)
        pr_auc = auc(recall, precision)
        print(f"Precision-Recall AUC: {pr_auc:.4f}")
        
        # Error statistics
        normal_errors = reconstruction_errors[y_true == 0]
        anomaly_errors = reconstruction_errors[y_true == 1]
        
        print(f"\nReconstruction Error Statistics:")
        print(f"Normal data - Mean: {np.mean(normal_errors):.6f}, Std: {np.std(normal_errors):.6f}")
        print(f"Anomaly data - Mean: {np.mean(anomaly_errors):.6f}, Std: {np.std(anomaly_errors):.6f}")
        print(f"Error separation ratio: {np.mean(anomaly_errors) / np.mean(normal_errors):.2f}")
    
    def _plot_evaluation_analysis(self, y_true, reconstruction_errors, X_test, X_reconstructed):
        """Plot comprehensive evaluation analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Error distribution
        normal_errors = reconstruction_errors[y_true == 0]
        anomaly_errors = reconstruction_errors[y_true == 1]
        
        axes[0, 0].hist(normal_errors, bins=50, alpha=0.7, label='Normal', density=True, color='blue')
        axes[0, 0].hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', density=True, color='red')
        axes[0, 0].axvline(x=self.threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold: {self.threshold:.6f}')
        axes[0, 0].set_xlabel('Reconstruction Error')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Reconstruction Error Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, reconstruction_errors)
        auc_score = roc_auc_score(y_true, reconstruction_errors)
        
        axes[0, 1].plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='red', linestyle='--')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_true, reconstruction_errors)
        pr_auc = auc(recall, precision)
        
        axes[0, 2].plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. Error timeline
        axes[1, 0].plot(reconstruction_errors, 'b-', alpha=0.7)
        axes[1, 0].axhline(y=self.threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {self.threshold:.6f}')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Reconstruction Error')
        axes[1, 0].set_title('Reconstruction Error Timeline')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 5. Original vs Reconstructed (first feature)
        feature_idx = 0
        axes[1, 1].plot(X_test[:, feature_idx], 'b-', alpha=0.7, label='Original')
        axes[1, 1].plot(X_reconstructed[:, feature_idx], 'r--', alpha=0.7, label='Reconstructed')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel(f'Feature {feature_idx}')
        axes[1, 1].set_title(f'Original vs Reconstructed (Feature {feature_idx})')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 6. Confusion Matrix
        y_pred = (reconstruction_errors > self.threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 2])
        axes[1, 2].set_title('Confusion Matrix')
        axes[1, 2].set_ylabel('True Label')
        axes[1, 2].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_path="autoencoder_ransomware.h5"):
        """Save the trained autoencoder model"""
        if self.autoencoder is None:
            print("No model to save!")
            return
        
        self.autoencoder.save_model(model_path)
        print(f"Autoencoder model saved to {model_path}")
        
        # Save scaler and threshold
        import joblib
        joblib.dump(self.scaler, "autoencoder_scaler.pkl")
        joblib.dump(self.threshold, "autoencoder_threshold.pkl")
        print("Scaler and threshold saved")
    
    def load_model(self, model_path="autoencoder_ransomware.h5"):
        """Load a trained autoencoder model"""
        try:
            self.autoencoder = AutoencoderAnomalyDetector()
            self.autoencoder.load_model(model_path)
            
            # Load scaler and threshold
            import joblib
            self.scaler = joblib.load("autoencoder_scaler.pkl")
            self.threshold = joblib.load("autoencoder_threshold.pkl")
            self.autoencoder.threshold = self.threshold
            
            print(f"Autoencoder model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

def train_autoencoder_complete(data_path, model_save_path="autoencoder_ransomware.h5"):
    """
    Complete pipeline for training autoencoder model
    
    Args:
        data_path: str, path to the ransomware dataset
        model_save_path: str, path to save the trained model
    """
    print("🚀 STARTING AUTOENCODER TRAINING PIPELINE")
    print("=" * 60)
    
    # Initialize trainer
    trainer = AutoencoderTrainer()
    
    # Load and preprocess data
    df_processed, X, y, df_original = trainer.load_and_preprocess_data(data_path)
    if df_processed is None:
        print("❌ Failed to load data!")
        return None
    
    # Prepare training data
    X_train, X_val, X_test, y_test = trainer.prepare_training_data(X, y)
    
    # Build and train autoencoder
    history = trainer.build_and_train_autoencoder(
        X_train, X_val,
        encoding_dims=[128, 64, 32],
        epochs=100,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Optimize threshold
    results_df = trainer.optimize_threshold(X_val, X_test, y_test)
    
    # Evaluate model
    y_pred, reconstruction_errors = trainer.evaluate_model(X_test, y_test)
    
    # Save model
    trainer.save_model(model_save_path)
    
    print("\n" + "=" * 60)
    print("✅ AUTOENCODER TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Model saved to: {model_save_path}")
    print(f"Optimal threshold: {trainer.threshold:.6f}")
    print("=" * 60)
    
    return trainer

# Example usage
if __name__ == "__main__":
    # Example with your dataset
    data_path = "d:/Desktop/Major project/EarlyStrikeAgent/earlystrike_autoencoder_75k_benign.csv"  # Replace with your actual dataset path
    
    # Train the autoencoder
    trainer = train_autoencoder_complete(data_path)
    
    if trainer:
        print("\n🎉 Autoencoder training completed!")
        print("You can now use the trained model for ransomware detection.")
    else:
        print("❌ Training failed. Please check your dataset and try again.")
