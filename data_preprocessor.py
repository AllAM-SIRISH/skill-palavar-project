import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.column_transformer = None
        self.feature_columns = []
        self.numeric_features = []
        self.categorical_features = []
        self.is_fitted = False
        
    def identify_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify numeric and categorical features"""
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target and ID columns
        exclude_cols = ['is_fraud', 'transaction_id', 'customer_id', 'timestamp', 'fraud_type']
        numeric_features = [col for col in numeric_features if col not in exclude_cols]
        categorical_features = [col for col in categorical_features if col not in exclude_cols]
        
        return numeric_features, categorical_features
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced derived features for fraud detection"""
        df = df.copy()
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Amount-based features
        if 'amount' in df.columns:
            # Log transformation for amount
            df['log_amount'] = np.log1p(df['amount'])
            
            # Amount bins
            df['amount_category'] = pd.cut(df['amount'], 
                                         bins=[0, 50, 200, 500, 2000, float('inf')],
                                         labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Customer behavior features
        if 'amount' in df.columns and 'income' in df.columns:
            df['amount_to_income_ratio'] = df['amount'] / (df['income'] / 12)
            
        if 'amount' in df.columns and 'account_balance' in df.columns:
            df['amount_to_balance_ratio'] = df['amount'] / df['account_balance']
            df['balance_impact'] = df['amount'] / (df['account_balance'] + df['amount'])
        
        # Risk scoring features
        if 'location' in df.columns:
            high_risk_locations = ['international']
            df['is_high_risk_location'] = df['location'].isin(high_risk_locations).astype(int)
        
        if 'merchant_category' in df.columns:
            high_risk_merchants = ['electronics', 'luxury', 'gambling', 'cryptocurrency', 'unknown']
            df['is_high_risk_merchant'] = df['merchant_category'].isin(high_risk_merchants).astype(int)
        
        if 'device_type' in df.columns:
            df['is_mobile_device'] = (df['device_type'] == 'mobile').astype(int)
        
        # Transaction frequency features (would need historical data in real scenario)
        if 'customer_id' in df.columns:
            # For synthetic data, we'll simulate this
            customer_stats = df.groupby('customer_id')['amount'].agg(['count', 'mean', 'std']).reset_index()
            customer_stats.columns = ['customer_id', 'customer_transaction_count', 'customer_avg_amount', 'customer_amount_std']
            df = df.merge(customer_stats, on='customer_id', how='left')
            
            # Deviation from customer's normal behavior
            df['amount_deviation'] = (df['amount'] - df['customer_avg_amount']) / (df['customer_amount_std'] + 1)
            df['is_amount_above_avg'] = (df['amount'] > df['customer_avg_amount']).astype(int)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()
        
        categorical_cols = ['merchant_category', 'location', 'transaction_type', 'device_type']
        if 'amount_category' in df.columns:
            categorical_cols.append('amount_category')
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_values = set(self.label_encoders[col].classes_)
                        df[col + '_encoded'] = df[col].astype(str).apply(
                            lambda x: x if x in unique_values else 'unknown'
                        )
                        # Add 'unknown' to encoder classes if not present
                        if 'unknown' not in unique_values:
                            self.label_encoders[col].classes_ = np.append(
                                self.label_encoders[col].classes_, 'unknown'
                            )
                        df[col + '_encoded'] = self.label_encoders[col].transform(df[col + '_encoded'])
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for machine learning"""
        # Create derived features
        df = self.create_derived_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit)
        
        # Identify feature columns
        exclude_cols = ['is_fraud', 'transaction_id', 'customer_id', 'timestamp', 'fraud_type',
                       'merchant_category', 'location', 'transaction_type', 'device_type', 'amount_category']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove any remaining non-numeric columns
        numeric_features = []
        for col in feature_cols:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_features.append(col)
        
        # Handle missing values
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
        
        # Scale features
        if fit:
            self.feature_columns = numeric_features
            # Store median values for later use
            self.feature_medians = df[numeric_features].median()
            X_scaled = self.scaler.fit_transform(df[numeric_features])
            self.is_fitted = True
        else:
            # Ensure columns are in the same order as during fit
            df_ordered = df[self.feature_columns]
            # Fill missing values with 0 (safe default for financial features)
            df_filled = df_ordered.fillna(0)
            X_scaled = self.scaler.transform(df_filled)
        
        return X_scaled, numeric_features
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Fit preprocessor and transform data"""
        X, feature_names = self.prepare_features(df, fit=True)
        y = df['is_fraud'].values if 'is_fraud' in df.columns else None
        return X, y, feature_names
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform new data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        X, _ = self.prepare_features(df, fit=False)
        y = df['is_fraud'].values if 'is_fraud' in df.columns else None
        return X, y
    
    def save_preprocessor(self, filepath: str):
        """Save the preprocessor object"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath: str):
        """Load a saved preprocessor"""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get processed data for feature importance analysis"""
        X, feature_names = self.prepare_features(df, fit=False)
        return pd.DataFrame(X, columns=feature_names)

if __name__ == "__main__":
    # Example usage
    print("Testing data preprocessor...")
    
    # Load or generate sample data
    try:
        df = pd.read_csv('financial_data_transactions.csv')
        print(f"Loaded data with shape: {df.shape}")
    except FileNotFoundError:
        print("No data file found. Please run data_generator.py first")
        exit()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Fit and transform data
    X, y, feature_names = preprocessor.fit_transform(df)
    
    print(f"Processed data shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Save preprocessor
    preprocessor.save_preprocessor('preprocessor.pkl')
