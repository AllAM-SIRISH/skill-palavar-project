import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import time
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_generator import FinancialDataGenerator
from data_preprocessor import DataPreprocessor
from fraud_models import train_multiple_models, ModelEvaluator
from real_time_detector import RealTimeFraudDetector

# Page configuration
st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

def load_models_and_data():
    """Load trained models and data"""
    try:
        # Try to load existing models
        with open('fraud_models.pkl', 'rb') as f:
            st.session_state.models = pickle.load(f)
        
        with open('preprocessor.pkl', 'rb') as f:
            st.session_state.preprocessor = pickle.load(f)
        
        # Initialize detector
        st.session_state.detector = RealTimeFraudDetector()
        
        return True
    except FileNotFoundError:
        return False

def generate_sample_data():
    """Generate sample data if not exists"""
    try:
        df = pd.read_csv('financial_data_transactions.csv')
        return df
    except FileNotFoundError:
        st.info("Generating sample data...")
        generator = FinancialDataGenerator(seed=42)
        transactions, customers = generator.generate_dataset(
            n_customers=500, 
            n_transactions=10000, 
            fraud_ratio=0.03
        )
        generator.save_dataset(transactions, customers)
        return transactions

def train_models():
    """Train fraud detection models"""
    st.info("Training fraud detection models...")
    
    # Generate or load data
    df = generate_sample_data()
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X, y, feature_names = preprocessor.fit_transform(df)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    models, results = train_multiple_models(X_train, X_test, y_train, y_test, feature_names)
    
    # Save models
    with open('fraud_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    preprocessor.save_preprocessor('preprocessor.pkl')
    
    st.session_state.models = models
    st.session_state.preprocessor = preprocessor
    st.session_state.detector = RealTimeFraudDetector()
    
    st.success("Models trained and saved successfully!")
    return models, results

def create_transaction_input():
    """Create transaction input form"""
    st.subheader("📝 Submit New Transaction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=100000.0, value=100.0, step=0.01)
        merchant_category = st.selectbox("Merchant Category", 
                                       ['grocery', 'restaurant', 'gas', 'retail', 'online', 
                                        'entertainment', 'healthcare', 'education', 'utilities',
                                        'electronics', 'luxury', 'gambling', 'cryptocurrency', 'unknown'])
        location = st.selectbox("Location", ['local', 'regional', 'national', 'international'])
        transaction_type = st.selectbox("Transaction Type", ['purchase', 'withdrawal', 'transfer', 'payment'])
    
    with col2:
        device_type = st.selectbox("Device Type", ['mobile', 'web', 'pos', 'atm'])
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
        customer_income = st.number_input("Annual Income ($)", min_value=20000, max_value=1000000, value=75000)
        account_balance = st.number_input("Account Balance ($)", min_value=100, max_value=1000000, value=10000)
    
    # Generate transaction ID
    transaction_id = f"TXN_{int(time.time())}"
    
    transaction_data = {
        'transaction_id': transaction_id,
        'customer_id': f'CUST_{np.random.randint(0, 1000):06d}',
        'amount': amount,
        'merchant_category': merchant_category,
        'location': location,
        'transaction_type': transaction_type,
        'device_type': device_type,
        'age': customer_age,
        'income': customer_income,
        'account_balance': account_balance,
        'account_age_days': np.random.randint(30, 3650),
        'credit_score': np.random.randint(300, 850),
        'transaction_frequency': np.random.randint(1, 50),
        'avg_transaction_amount': amount * np.random.uniform(0.8, 1.2),
        'timestamp': datetime.now().isoformat()
    }
    
    return transaction_data

def process_transaction(transaction_data):
    """Process a transaction and return results"""
    if st.session_state.detector is None:
        return None
    
    result = st.session_state.detector.simulate_real_time_processing(transaction_data)
    
    # Add to history
    st.session_state.transaction_history.append({
        'transaction_id': result.transaction_id,
        'amount': transaction_data['amount'],
        'merchant_category': transaction_data['merchant_category'],
        'location': transaction_data['location'],
        'is_fraud': result.is_fraud,
        'fraud_probability': result.fraud_probability,
        'risk_level': result.risk_level,
        'confidence_score': result.confidence_score,
        'timestamp': result.timestamp,
        'processing_time': result.processing_time
    })
    
    # Keep only last 100 transactions in history
    if len(st.session_state.transaction_history) > 100:
        st.session_state.transaction_history = st.session_state.transaction_history[-100:]
    
    # Add to alerts if high risk
    if result.risk_level in ['high', 'critical']:
        st.session_state.alerts.append({
            'alert_id': f"ALERT_{int(time.time())}",
            'transaction_id': result.transaction_id,
            'risk_level': result.risk_level,
            'fraud_probability': result.fraud_probability,
            'timestamp': result.timestamp,
            'amount': transaction_data['amount'],
            'merchant_category': transaction_data['merchant_category']
        })
        
        # Keep only last 50 alerts
        if len(st.session_state.alerts) > 50:
            st.session_state.alerts = st.session_state.alerts[-50:]
    
    return result

def display_metrics():
    """Display system metrics"""
    st.subheader("📊 System Metrics")
    
    if not st.session_state.transaction_history:
        st.info("No transactions processed yet")
        return
    
    history_df = pd.DataFrame(st.session_state.transaction_history)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_transactions = len(history_df)
        st.metric("Total Transactions", total_transactions)
    
    with col2:
        fraud_count = history_df['is_fraud'].sum()
        fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
        st.metric("Fraud Detected", f"{fraud_count} ({fraud_rate:.1f}%)")
    
    with col3:
        high_risk_count = (history_df['risk_level'].isin(['high', 'critical'])).sum()
        st.metric("High Risk Alerts", high_risk_count)
    
    with col4:
        avg_processing_time = history_df['processing_time'].mean()
        st.metric("Avg Processing Time", f"{avg_processing_time:.4f}s")
    
    # Risk level distribution
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = history_df['risk_level'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                     title="Risk Level Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Fraud probability over time
        history_df['datetime'] = pd.to_datetime(history_df['timestamp'])
        fig = px.line(history_df, x='datetime', y='fraud_probability', 
                     title="Fraud Probability Over Time",
                     color='is_fraud')
        st.plotly_chart(fig, use_container_width=True)

def display_transaction_history():
    """Display transaction history"""
    st.subheader("📜 Recent Transactions")
    
    if not st.session_state.transaction_history:
        st.info("No transactions processed yet")
        return
    
    history_df = pd.DataFrame(st.session_state.transaction_history)
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Add color coding for risk levels
    def highlight_risk(row):
        if row['risk_level'] == 'critical':
            return ['background-color: #ffebee'] * len(row)
        elif row['risk_level'] == 'high':
            return ['background-color: #fff3e0'] * len(row)
        elif row['risk_level'] == 'medium':
            return ['background-color: #f3e5f5'] * len(row)
        else:
            return ['background-color: #e8f5e8'] * len(row)
    
    styled_df = history_df.style.apply(highlight_risk, axis=1)
    st.dataframe(styled_df, use_container_width=True)

def display_alerts():
    """Display fraud alerts"""
    st.subheader("🚨 Fraud Alerts")
    
    if not st.session_state.alerts:
        st.info("No fraud alerts")
        return
    
    for alert in reversed(st.session_state.alerts[-10:]):  # Show last 10 alerts
        if alert['risk_level'] == 'critical':
            st.markdown(f"""
            <div class="fraud-alert">
                <strong>🚨 CRITICAL ALERT</strong><br>
                Transaction: {alert['transaction_id']}<br>
                Amount: ${alert['amount']:.2f}<br>
                Risk Level: {alert['risk_level']}<br>
                Fraud Probability: {alert['fraud_probability']:.3f}<br>
                Time: {alert['timestamp']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="fraud-alert">
                <strong>⚠️ HIGH RISK ALERT</strong><br>
                Transaction: {alert['transaction_id']}<br>
                Amount: ${alert['amount']:.2f}<br>
                Risk Level: {alert['risk_level']}<br>
                Fraud Probability: {alert['fraud_probability']:.3f}<br>
                Time: {alert['timestamp']}
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">🔍 Financial Fraud Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("System Control")
    
    # Initialize models
    if not load_models_and_data():
        if st.sidebar.button("🚀 Initialize System", type="primary"):
            with st.spinner("Training models... This may take a few minutes."):
                train_models()
                st.rerun()
        
        st.sidebar.info("Please initialize the system to start fraud detection.")
        return
    
    st.sidebar.success("✅ System Ready")
    
    # System status
    st.sidebar.subheader("System Status")
    st.sidebar.write(f"Models Loaded: {len(st.session_state.models)}")
    st.sidebar.write(f"Transactions Processed: {len(st.session_state.transaction_history)}")
    st.sidebar.write(f"Active Alerts: {len(st.session_state.alerts)}")
    
    if st.sidebar.button("🔄 Reset System"):
        st.session_state.transaction_history = []
        st.session_state.alerts = []
        st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Real-time Detection", "📊 Analytics", "📜 History", "🚨 Alerts"])
    
    with tab1:
        # Real-time detection
        transaction_data = create_transaction_input()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🔍 Process Transaction", type="primary", use_container_width=True):
                with st.spinner("Processing transaction..."):
                    result = process_transaction(transaction_data)
                    
                    if result:
                        if result.is_fraud:
                            st.markdown(f"""
                            <div class="fraud-alert">
                                <strong>⚠️ FRAUD DETECTED!</strong><br>
                                Transaction ID: {result.transaction_id}<br>
                                Risk Level: {result.risk_level.upper()}<br>
                                Fraud Probability: {result.fraud_probability:.3f}<br>
                                Confidence: {result.confidence_score:.3f}<br>
                                Processing Time: {result.processing_time:.4f}s
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="success-message">
                                <strong>✅ Transaction Approved</strong><br>
                                Transaction ID: {result.transaction_id}<br>
                                Risk Level: {result.risk_level.upper()}<br>
                                Fraud Probability: {result.fraud_probability:.3f}<br>
                                Confidence: {result.confidence_score:.3f}<br>
                                Processing Time: {result.processing_time:.4f}s
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show model predictions
                        st.subheader("Model Predictions")
                        model_df = pd.DataFrame(list(result.model_predictions.items()), 
                                               columns=['Model', 'Fraud Probability'])
                        st.dataframe(model_df, use_container_width=True)
        
        with col2:
            # Quick stats
            if st.session_state.transaction_history:
                history_df = pd.DataFrame(st.session_state.transaction_history)
                recent_fraud = history_df.tail(10)['is_fraud'].sum()
                st.metric("Last 10 - Fraud", recent_fraud)
    
    with tab2:
        # Analytics
        display_metrics()
    
    with tab3:
        # Transaction history
        display_transaction_history()
    
    with tab4:
        # Alerts
        display_alerts()

if __name__ == "__main__":
    main()
