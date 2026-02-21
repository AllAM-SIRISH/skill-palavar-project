import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class IsolationForestDetector:
    def __init__(self, contamination=0.03, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.threshold = None
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """Fit the Isolation Forest model"""
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False
        )
        self.model.fit(X)
        
        # Set threshold based on training data
        scores = self.model.decision_function(X)
        self.threshold = np.percentile(scores, self.contamination * 100)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Predict anomalies (-1 for anomaly, 1 for normal)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        # Convert to 0/1 (1 for fraud, 0 for normal)
        return (predictions == -1).astype(int)
    
    def predict_proba(self, X):
        """Get anomaly scores (lower = more anomalous)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        scores = self.model.decision_function(X)
        # Convert to probability-like scores (0-1, higher = more likely fraud)
        min_score, max_score = scores.min(), scores.max()
        if max_score == min_score:
            return np.zeros(len(scores))
        
        # Invert and normalize to get fraud probability
        probas = 1 - (scores - min_score) / (max_score - min_score)
        return probas
    
    def get_anomaly_scores(self, X):
        """Get raw anomaly scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.decision_function(X)

class AutoencoderDetector:
    def __init__(self, encoding_dim=16, input_dim=None, epochs=100, batch_size=32):
        self.encoding_dim = encoding_dim
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.threshold = None
        self.history = None
        self.is_fitted = False
        
    def build_model(self):
        """Build the autoencoder architecture"""
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(self.input_dim, activation='linear')(decoded)
        
        # Autoencoder model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return autoencoder
    
    def fit(self, X, y=None, validation_split=0.2):
        """Fit the autoencoder model"""
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        self.model = self.build_model()
        
        # Train only on normal transactions (if labels available)
        if y is not None:
            normal_data = X[y == 0]
            if len(normal_data) > 0:
                X_train = normal_data
            else:
                X_train = X
        else:
            X_train = X
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        
        # Train the model
        self.history = self.model.fit(
            X_train, X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Calculate reconstruction error threshold
        reconstructions = self.model.predict(X_train)
        reconstruction_errors = np.mean(np.square(X_train - reconstructions), axis=1)
        self.threshold = np.percentile(reconstruction_errors, 95)  # Top 5% as anomalies
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Predict anomalies (1 for fraud, 0 for normal)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        reconstructions = self.model.predict(X)
        reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
        return (reconstruction_errors > self.threshold).astype(int)
    
    def predict_proba(self, X):
        """Get fraud probabilities based on reconstruction error"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        reconstructions = self.model.predict(X)
        reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
        
        # Normalize reconstruction errors to probabilities
        max_error = reconstruction_errors.max()
        if max_error == 0:
            return np.zeros(len(reconstruction_errors))
        
        probas = reconstruction_errors / max_error
        return np.clip(probas, 0, 1)
    
    def get_reconstruction_errors(self, X):
        """Get raw reconstruction errors"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        reconstructions = self.model.predict(X)
        return np.mean(np.square(X - reconstructions), axis=1)

class EnsembleFraudDetector:
    def __init__(self, models=None, weights=None):
        self.models = models if models else []
        self.weights = weights if weights else [1.0] * len(models)
        self.is_fitted = False
        
    def add_model(self, model, weight=1.0):
        """Add a model to the ensemble"""
        self.models.append(model)
        self.weights.append(weight)
    
    def fit(self, X, y=None):
        """Fit all models in the ensemble"""
        for model in self.models:
            if hasattr(model, 'fit'):
                model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Ensemble prediction"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                predictions.append(pred)
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Weighted voting
        weighted_predictions = np.average(predictions, axis=0, weights=self.weights)
        return (weighted_predictions > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Ensemble probability prediction"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        probas = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X)
                probas.append(prob)
        
        if not probas:
            raise ValueError("No models available for probability prediction")
        
        # Weighted average of probabilities
        weighted_probas = np.average(probas, axis=0, weights=self.weights)
        return weighted_probas

class ModelEvaluator:
    @staticmethod
    def evaluate_model(model, X_test, y_test, model_name="Model"):
        """Comprehensive model evaluation"""
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
        
        # Basic metrics
        print(f"\n=== {model_name} Evaluation ===")
        print(f"Test set size: {len(y_test)}")
        print(f"Fraud cases in test: {y_test.sum()} ({y_test.mean():.4f})")
        print(f"Predicted fraud cases: {y_pred.sum()} ({y_pred.mean():.4f})")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # ROC AUC if probabilities available
        if y_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_proba)
                print(f"\nROC AUC Score: {auc:.4f}")
            except:
                print("\nCould not calculate ROC AUC score")
        
        # Calculate business metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nBusiness Metrics:")
        print(f"True Positives (Caught Fraud): {tp}")
        print(f"False Positives (Blocked Legitimate): {fp}")
        print(f"False Negatives (Missed Fraud): {fn}")
        print(f"True Negatives (Allowed Legitimate): {tn}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return {
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    @staticmethod
    def plot_feature_importance(model, feature_names, top_n=20):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(10, 8))
            plt.title('Top Feature Importances')
            plt.bar(range(top_n), importances[indices])
            plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        else:
            print("Model does not support feature importance")

def train_multiple_models(X_train, X_test, y_train, y_test, feature_names):
    """Train multiple fraud detection models"""
    models = {}
    results = {}
    
    print("Training Isolation Forest...")
    iso_forest = IsolationForestDetector(contamination=0.03)
    iso_forest.fit(X_train, y_train)
    models['isolation_forest'] = iso_forest
    results['isolation_forest'] = ModelEvaluator.evaluate_model(
        iso_forest, X_test, y_test, "Isolation Forest"
    )
    
    print("\nTraining Autoencoder...")
    autoencoder = AutoencoderDetector(encoding_dim=16, epochs=50, batch_size=64)
    autoencoder.fit(X_train, y_train)
    models['autoencoder'] = autoencoder
    results['autoencoder'] = ModelEvaluator.evaluate_model(
        autoencoder, X_test, y_test, "Autoencoder"
    )
    
    print("\nTraining Ensemble...")
    ensemble = EnsembleFraudDetector([iso_forest, autoencoder], weights=[0.6, 0.4])
    ensemble.fit(X_train, y_train)
    models['ensemble'] = ensemble
    results['ensemble'] = ModelEvaluator.evaluate_model(
        ensemble, X_test, y_test, "Ensemble"
    )
    
    return models, results

if __name__ == "__main__":
    # Example usage
    print("Testing fraud detection models...")
    
    # Load preprocessed data
    try:
        preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
        df = pd.read_csv('financial_data_transactions.csv')
        X, y = preprocessor.transform(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Train models
        models, results = train_multiple_models(
            X_train, X_test, y_train, y_test, preprocessor.feature_columns
        )
        
        # Save models
        with open('fraud_models.pkl', 'wb') as f:
            pickle.dump(models, f)
        
        print("\nModels saved to fraud_models.pkl")
        
    except FileNotFoundError as e:
        print(f"Missing files: {e}")
        print("Please run data_generator.py and data_preprocessor.py first")
