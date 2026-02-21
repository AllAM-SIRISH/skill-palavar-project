import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from sklearn.preprocessing import StandardScaler
import pickle

class FinancialDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_customer_profiles(self, n_customers=1000):
        """Generate realistic customer profiles"""
        customers = []
        
        for i in range(n_customers):
            customer = {
                'customer_id': f'CUST_{i:06d}',
                'age': np.random.normal(35, 12),
                'income': np.random.lognormal(10.5, 0.5),
                'account_balance': np.random.lognormal(9, 1.5),
                'account_age_days': np.random.randint(30, 3650),
                'credit_score': np.random.normal(650, 100),
                'transaction_frequency': np.random.poisson(15),  # transactions per week
                'avg_transaction_amount': np.random.lognormal(6, 1)
            }
            # Ensure realistic bounds
            customer['age'] = max(18, min(80, customer['age']))
            customer['income'] = max(20000, min(500000, customer['income']))
            customer['account_balance'] = max(100, customer['account_balance'])
            customer['credit_score'] = max(300, min(850, customer['credit_score']))
            customers.append(customer)
            
        return pd.DataFrame(customers)
    
    def generate_normal_transactions(self, customers, n_transactions=100000):
        """Generate normal legitimate transactions"""
        transactions = []
        
        for i in range(n_transactions):
            customer = customers.iloc[np.random.randint(0, len(customers))]
            
            # Time-based patterns
            base_time = datetime.now() - timedelta(days=np.random.randint(0, 365))
            hour = np.random.choice(
                [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                p=[0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.12, 0.1, 0.05, 0.03, 0.01, 0.01]
            )
            
            transaction = {
                'transaction_id': f'TXN_{i:08d}',
                'customer_id': customer['customer_id'],
                'timestamp': base_time.replace(hour=hour, minute=np.random.randint(0, 60)),
                'amount': np.random.lognormal(
                    np.log(customer['avg_transaction_amount']), 
                    0.8
                ),
                'merchant_category': np.random.choice([
                    'grocery', 'restaurant', 'gas', 'retail', 'online', 
                    'entertainment', 'healthcare', 'education', 'utilities'
                ], p=[0.2, 0.15, 0.1, 0.15, 0.2, 0.08, 0.05, 0.04, 0.03]),
                'location': np.random.choice([
                    'local', 'regional', 'national', 'international'
                ], p=[0.6, 0.25, 0.12, 0.03]),
                'transaction_type': np.random.choice([
                    'purchase', 'withdrawal', 'transfer', 'payment'
                ], p=[0.7, 0.1, 0.15, 0.05]),
                'device_type': np.random.choice([
                    'mobile', 'web', 'pos', 'atm'
                ], p=[0.4, 0.3, 0.2, 0.1]),
                'is_fraud': 0
            }
            
            # Ensure amount is reasonable
            transaction['amount'] = max(1, min(10000, transaction['amount']))
            
            transactions.append(transaction)
            
        return pd.DataFrame(transactions)
    
    def generate_fraudulent_transactions(self, customers, n_fraud=2000):
        """Generate various types of fraudulent transactions"""
        fraud_transactions = []
        
        fraud_types = [
            'high_amount', 'rapid_sequence', 'unusual_location', 
            'unusual_time', 'new_merchant', 'account_takeover'
        ]
        
        for i in range(n_fraud):
            customer = customers.iloc[np.random.randint(0, len(customers))]
            fraud_type = np.random.choice(fraud_types)
            
            base_time = datetime.now() - timedelta(days=np.random.randint(0, 30))
            
            if fraud_type == 'high_amount':
                # Unusually high transaction amount
                amount = customer['avg_transaction_amount'] * np.random.uniform(5, 20)
                
            elif fraud_type == 'rapid_sequence':
                # Multiple transactions in short time
                amount = customer['avg_transaction_amount'] * np.random.uniform(0.5, 2)
                
            elif fraud_type == 'unusual_location':
                # Transaction from unusual location
                amount = customer['avg_transaction_amount'] * np.random.uniform(1, 3)
                
            elif fraud_type == 'unusual_time':
                # Transaction at unusual time (2-5 AM)
                hour = np.random.randint(2, 6)
                base_time = base_time.replace(hour=hour)
                amount = customer['avg_transaction_amount'] * np.random.uniform(1, 2)
                
            elif fraud_type == 'new_merchant':
                # Transaction from new merchant category
                amount = customer['avg_transaction_amount'] * np.random.uniform(1, 5)
                
            else:  # account_takeover
                # Multiple suspicious patterns
                amount = customer['avg_transaction_amount'] * np.random.uniform(2, 10)
            
            transaction = {
                'transaction_id': f'FRD_{i:08d}',
                'customer_id': customer['customer_id'],
                'timestamp': base_time,
                'amount': amount,
                'merchant_category': np.random.choice([
                    'electronics', 'luxury', 'gambling', 'cryptocurrency', 'unknown'
                ]) if fraud_type in ['account_takeover', 'new_merchant'] else np.random.choice([
                    'grocery', 'restaurant', 'gas', 'retail', 'online'
                ]),
                'location': 'international' if fraud_type == 'unusual_location' else np.random.choice([
                    'local', 'regional', 'national', 'international'
                ], p=[0.3, 0.2, 0.2, 0.3]),
                'transaction_type': 'purchase' if np.random.random() > 0.3 else 'transfer',
                'device_type': np.random.choice(['mobile', 'web', 'pos'], p=[0.5, 0.4, 0.1]),
                'is_fraud': 1,
                'fraud_type': fraud_type
            }
            
            fraud_transactions.append(transaction)
            
        return pd.DataFrame(fraud_transactions)
    
    def generate_dataset(self, n_customers=1000, n_transactions=100000, fraud_ratio=0.02):
        """Generate complete dataset"""
        print("Generating customer profiles...")
        customers = self.generate_customer_profiles(n_customers)
        
        print("Generating normal transactions...")
        n_normal = int(n_transactions * (1 - fraud_ratio))
        normal_transactions = self.generate_normal_transactions(customers, n_normal)
        
        print("Generating fraudulent transactions...")
        n_fraud = n_transactions - n_normal
        fraud_transactions = self.generate_fraudulent_transactions(customers, n_fraud)
        
        print("Combining datasets...")
        all_transactions = pd.concat([normal_transactions, fraud_transactions], ignore_index=True)
        
        # Shuffle the dataset
        all_transactions = all_transactions.sample(frac=1).reset_index(drop=True)
        
        # Add customer features to transactions
        transactions_with_features = all_transactions.merge(
            customers[['customer_id', 'age', 'income', 'account_balance', 
                      'account_age_days', 'credit_score', 'transaction_frequency']],
            on='customer_id',
            how='left'
        )
        
        # Add derived features
        transactions_with_features['amount_to_income_ratio'] = (
            transactions_with_features['amount'] / 
            transactions_with_features['income'] * 12
        )
        transactions_with_features['amount_to_balance_ratio'] = (
            transactions_with_features['amount'] / 
            transactions_with_features['account_balance']
        )
        
        # Time-based features
        transactions_with_features['hour'] = pd.to_datetime(
            transactions_with_features['timestamp']
        ).dt.hour
        transactions_with_features['day_of_week'] = pd.to_datetime(
            transactions_with_features['timestamp']
        ).dt.dayofweek
        transactions_with_features['is_weekend'] = (
            transactions_with_features['day_of_week'].isin([5, 6]).astype(int)
        )
        
        print(f"Dataset generated: {len(transactions_with_features)} transactions")
        print(f"Fraud ratio: {transactions_with_features['is_fraud'].mean():.4f}")
        
        return transactions_with_features, customers
    
    def save_dataset(self, transactions, customers, filename_prefix='financial_data'):
        """Save datasets to files"""
        transactions.to_csv(f'{filename_prefix}_transactions.csv', index=False)
        customers.to_csv(f'{filename_prefix}_customers.csv', index=False)
        print(f"Datasets saved as {filename_prefix}_transactions.csv and {filename_prefix}_customers.csv")

if __name__ == "__main__":
    generator = FinancialDataGenerator(seed=42)
    transactions, customers = generator.generate_dataset(
        n_customers=1000, 
        n_transactions=50000, 
        fraud_ratio=0.03
    )
    generator.save_dataset(transactions, customers)
