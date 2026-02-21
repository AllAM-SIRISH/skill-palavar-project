import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score
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

def train_simple_models(X_train, X_test, y_train, y_test, feature_names):
    """Train simplified fraud detection models"""
    models = {}
    results = {}
    
    print("Training Isolation Forest...")
    iso_forest = IsolationForestDetector(contamination=0.03)
    iso_forest.fit(X_train, y_train)
    models['isolation_forest'] = iso_forest
    results['isolation_forest'] = ModelEvaluator.evaluate_model(
        iso_forest, X_test, y_test, "Isolation Forest"
    )
    
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    results['random_forest'] = ModelEvaluator.evaluate_model(
        rf, X_test, y_test, "Random Forest"
    )
    
    return models, results

if __name__ == "__main__":
    # Example usage
    print("Testing simplified fraud detection models...")
    
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
        models, results = train_simple_models(
            X_train, X_test, y_train, y_test, preprocessor.feature_columns
        )
        
        # Save models
        with open('fraud_models_simple.pkl', 'wb') as f:
            pickle.dump(models, f)
        
        print("\nModels saved to fraud_models_simple.pkl")
        
    except FileNotFoundError as e:
        print(f"Missing files: {e}")
        print("Please run data_generator.py and data_preprocessor.py first")
