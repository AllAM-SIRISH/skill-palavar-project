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

# Import our custom modules (without TensorFlow)
from data_generator import FinancialDataGenerator
from data_preprocessor import DataPreprocessor
from simple_fraud_models import IsolationForestDetector, ModelEvaluator, train_simple_models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
    /* Main Header */
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Main App Background */
    .stApp {
        background-color: #ffffff !important;
    }
    
    .main .block-container {
        background-color: #ffffff !important;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Force all containers to be white */
    .stApp > div {
        background-color: #ffffff !important;
    }
    
    .stApp > div > div {
        background-color: #ffffff !important;
    }
    
    .stApp > div > div > div {
        background-color: #ffffff !important;
    }
    
    /* Main content area */
    .main {
        background-color: #ffffff !important;
    }
    
    .main > div {
        background-color: #ffffff !important;
    }
    
    .main > div > div {
        background-color: #ffffff !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #3498db;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Alert Messages */
    .fraud-alert {
        background: linear-gradient(135deg, #ff4757 0%, #c44569 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(255, 71, 87, 0.4);
        border-left: 6px solid #ee5a24;
        font-weight: 600;
        font-size: 1.1rem;
        position: relative;
        overflow: hidden;
    }
    
    .fraud-alert::before {
        content: "⚠️";
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 2rem;
        opacity: 0.8;
    }
    
    .success-message {
        background: linear-gradient(135deg, #00d2d3 0%, #01a3a4 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 210, 211, 0.4);
        border-left: 6px solid #00b894;
        font-weight: 600;
        font-size: 1.1rem;
        position: relative;
        overflow: hidden;
    }
    
    .success-message::before {
        content: "✅";
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 2rem;
        opacity: 0.8;
    }
    
    .normal-alert {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(116, 185, 255, 0.4);
        border-left: 6px solid #0984e3;
        font-weight: 600;
        font-size: 1.1rem;
        position: relative;
        overflow: hidden;
    }
    
    .normal-alert::before {
        content: "ℹ️";
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 2rem;
        opacity: 0.8;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(52, 152, 219, 0.4);
    }
    
    /* Input Elements - Enhanced Nice Colors */
    .stSelectbox > div > div,
    .stNumberInput > div > div,
    .stTextInput > div > div {
        background: transparent !important;
        border: 2px solid #3498db;
        border-radius: 0.75rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stSelectbox > div > div::before,
    .stNumberInput > div > div::before,
    .stTextInput > div > div::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(52, 152, 219, 0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .stSelectbox > div > div:hover::before,
    .stNumberInput > div > div:hover::before,
    .stTextInput > div > div:hover::before {
        left: 100%;
    }
    
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div:hover,
    .stTextInput > div > div:hover {
        border-color: #2980b9;
        background: rgba(52, 152, 219, 0.1) !important;
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.3);
        transform: translateY(-2px);
    }
    
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div:focus-within,
    .stTextInput > div > div:focus-within {
        border-color: #1e88e5;
        background: rgba(30, 136, 229, 0.15) !important;
        box-shadow: 0 0 0 4px rgba(30, 136, 229, 0.2);
        transform: translateY(-2px);
    }
    
    /* Input Labels */
    .stSelectbox > label,
    .stNumberInput > label,
    .stTextInput > label {
        font-weight: 600;
        color: #000000;
        margin-bottom: 0.5rem;
    }
    
    /* Data Tables */
    .stDataFrame {
        border-radius: 0.75rem;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        background-color: #ffffff;
    }
    
    .stDataFrame table {
        color: #000000 !important;
    }
    
    .stDataFrame th {
        color: #000000 !important;
        font-weight: 600;
    }
    
    .stDataFrame td {
        color: #000000 !important;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"] * {
        color: #000000 !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #ffffff !important;
        border-right: 1px solid #dee2e6 !important;
    }
    
    .css-1d391kg * {
        color: #000000 !important;
    }
    
    .css-1d391kg .stMarkdown {
        color: #000000 !important;
    }
    
    .css-1d391kg .stSubheader {
        color: #000000 !important;
    }
    
    .css-1d391kg .stSelectbox > label {
        color: #000000 !important;
    }
    
    .css-1d391kg .stNumberInput > label {
        color: #000000 !important;
    }
    
    .css-1d391kg .stTextInput > label {
        color: #000000 !important;
    }
    
    .css-1d391kg .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
    }
    
    .css-1d391kg .stButton > button:hover {
        background: linear-gradient(135deg, #2980b9 0%, #3498db 100%) !important;
        color: #ffffff !important;
    }
    
    .css-1d391kg .stInfo {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
        color: #000000 !important;
        border-left: 4px solid #2196f3 !important;
    }
    
    .css-1d391kg .stSuccess {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%) !important;
        color: #000000 !important;
        border-left: 4px solid #4caf50 !important;
    }
    
    /* Ensure sidebar content is white */
    .css-1d391kg .element-container {
        background-color: #ffffff !important;
    }
    
    .css-1d391kg .streamlit-container {
        background-color: #ffffff !important;
    }
    
    /* Hamburger menu button */
    .css-17eq0hr {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem !important;
        margin: 0.5rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .css-17eq0hr:hover {
        background-color: #f8f9fa !important;
        border-color: #3498db !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Menu icon */
    .css-17eq0hr svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* Alternative hamburger menu styles */
    button[title="View sidebar"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #dee2e6 !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem !important;
        margin: 0.5rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    button[title="View sidebar"]:hover {
        background-color: #f8f9fa !important;
        border-color: #3498db !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    button[title="View sidebar"] svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ffffff !important;
        border-radius: 0.75rem;
        padding: 0.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff !important;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid #e9ecef;
        color: #000000 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white !important;
        border-color: #3498db;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background-color: #f8f9fa !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        color: #000000 !important;
    }
    
    /* Form Sections */
    .stForm {
        background-color: #ffffff !important;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid #e9ecef;
    }
    
    /* Columns */
    .stColumns > div {
        padding: 1rem;
    }
    
    /* Number Input Styling */
    .stNumberInput > div > div > input {
        font-weight: 500;
        color: #2c3e50 !important;
        background: transparent !important;
        border: none !important;
        outline: none !important;
    }
    
    .stNumberInput > div > div > input:focus {
        color: #1e88e5 !important;
        background: transparent !important;
        border: none !important;
        outline: none !important;
    }
    
    .stNumberInput > div > div > input::placeholder {
        color: #95a5a6 !important;
    }
    
    .stNumberInput > div > div {
        background: transparent !important;
    }
    
    .stNumberInput > div > div > div {
        color: #2c3e50 !important;
        background: transparent !important;
    }
    
    /* Number input arrows/buttons */
    .stNumberInput > div > div > button {
        background: transparent !important;
        color: #3498db !important;
        border: none !important;
    }
    
    .stNumberInput > div > div > button:hover {
        background: rgba(52, 152, 219, 0.1) !important;
        color: #2980b9 !important;
    }
    
    /* Selectbox Dropdown */
    .stSelectbox > div > div > div {
        font-weight: 500;
        color: #000000 !important;
    }
    
    /* Subheaders */
    .stSubheader {
        font-weight: 600;
        color: #000000 !important;
        margin-bottom: 1rem;
    }
    
    /* Main Content Text */
    .stMarkdown {
        color: #000000 !important;
    }
    
    .stMarkdown p {
        color: #000000 !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #000000 !important;
    }
    
    /* Info Messages */
    .stInfo {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        border-radius: 0.5rem;
        color: #000000 !important;
    }
    
    /* Success Messages */
    .stSuccess {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-left: 4px solid #4caf50;
        border-radius: 0.5rem;
        color: #000000 !important;
    }
    
    /* Error Messages */
    .stError {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 4px solid #f44336;
        border-radius: 0.5rem;
        color: #000000 !important;
    }
    
    /* All text elements */
    * {
        color: #000000 !important;
    }
    
    /* Exceptions (keep original colors) */
    .fraud-alert,
    .success-message,
    .normal-alert,
    .stButton > button,
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

def load_models_and_data():
    """Load trained models and data"""
    try:
        # Try to load existing models
        with open('fraud_models_simple.pkl', 'rb') as f:
            st.session_state.models = pickle.load(f)
        
        with open('preprocessor.pkl', 'rb') as f:
            st.session_state.preprocessor = pickle.load(f)
        
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

def train_models_app():
    """Train simplified fraud detection models"""
    st.info("Training fraud detection models...")
    
    # Generate or load data
    df = generate_sample_data()
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X, y, feature_names = preprocessor.fit_transform(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models using the imported function
    from simple_fraud_models import train_simple_models as train_models_function
    models, results = train_models_function(X_train, X_test, y_train, y_test, feature_names)
    
    # Save models
    with open('fraud_models_simple.pkl', 'wb') as f:
        pickle.dump(models, f)
    preprocessor.save_preprocessor('preprocessor.pkl')
    
    st.session_state.models = models
    st.session_state.preprocessor = preprocessor
    
    st.success("Models trained and saved successfully!")
    return models

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
        'timestamp': datetime.now().isoformat()
    }
    
    return transaction_data

def process_transaction(transaction_data):
    """Process a transaction and return results"""
    if not st.session_state.models or not st.session_state.preprocessor:
        return None
    
    # Create a DataFrame with the same structure as training data
    df = pd.DataFrame([transaction_data])
    
    # Add missing features that the preprocessor expects
    # These will be created by the preprocessor's create_derived_features method
    try:
        X, _ = st.session_state.preprocessor.transform(df)
    except Exception as e:
        st.error(f"Error processing transaction: {e}")
        return None
    
    # Get predictions from all models
    model_predictions = {}
    fraud_probabilities = []
    
    for model_name, model in st.session_state.models.items():
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X)[0]
            # Handle array probability (for binary classification)
            if isinstance(prob, np.ndarray):
                prob = float(prob[1]) if len(prob) > 1 else float(prob)
            model_predictions[model_name] = prob
            fraud_probabilities.append(prob)
        elif hasattr(model, 'predict'):
            pred = model.predict(X)[0]
            model_predictions[model_name] = float(pred)
            fraud_probabilities.append(float(pred))
    
    # Calculate ensemble probability
    if fraud_probabilities:
        ensemble_probability = np.mean(fraud_probabilities)
    else:
        ensemble_probability = 0.0
    
    # Calculate confidence score
    confidence_score = 1.0 / (1.0 + np.var(fraud_probabilities)) if len(fraud_probabilities) > 1 else 0.5
    
    # Determine risk level
    if ensemble_probability >= 0.8:
        risk_level = 'critical'
    elif ensemble_probability >= 0.6:
        risk_level = 'high'
    elif ensemble_probability >= 0.3:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    # Make final decision
    is_fraud = ensemble_probability >= 0.3
    
    # Create result
    result = {
        'transaction_id': transaction_data['transaction_id'],
        'is_fraud': is_fraud,
        'fraud_probability': ensemble_probability,
        'confidence_score': confidence_score,
        'risk_level': risk_level,
        'model_predictions': model_predictions,
        'amount': transaction_data['amount'],
        'merchant_category': transaction_data['merchant_category'],
        'location': transaction_data['location'],
        'timestamp': datetime.now()
    }
    
    # Add to history
    st.session_state.transaction_history.append(result)
    
    # Keep only last 100 transactions
    if len(st.session_state.transaction_history) > 100:
        st.session_state.transaction_history = st.session_state.transaction_history[-100:]
    
    # Add to alerts if high risk
    if result['risk_level'] in ['high', 'critical']:
        st.session_state.alerts.append(result)
        
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
        avg_fraud_prob = history_df['fraud_probability'].mean()
        st.metric("Avg Fraud Probability", f"{avg_fraud_prob:.3f}")
    
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
    
    # Select columns to display
    display_cols = ['transaction_id', 'amount', 'merchant_category', 'location', 
                   'is_fraud', 'fraud_probability', 'risk_level', 'confidence_score', 'timestamp']
    
    st.dataframe(history_df[display_cols], use_container_width=True)

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
                train_models_app()
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
        
        if st.button("🔍 Process Transaction", type="primary", use_container_width=True):
            with st.spinner("Processing transaction..."):
                result = process_transaction(transaction_data)
                
                if result:
                    if result['is_fraud']:
                        st.markdown(f"""
                        <div class="fraud-alert">
                            <strong>⚠️ FRAUD DETECTED!</strong><br>
                            Transaction ID: {result['transaction_id']}<br>
                            Risk Level: {result['risk_level'].upper()}<br>
                            Fraud Probability: {result['fraud_probability']:.3f}<br>
                            Confidence: {result['confidence_score']:.3f}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-message">
                            <strong>✅ Transaction Approved</strong><br>
                            Transaction ID: {result['transaction_id']}<br>
                            Risk Level: {result['risk_level'].upper()}<br>
                            Fraud Probability: {result['fraud_probability']:.3f}<br>
                            Confidence: {result['confidence_score']:.3f}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show model predictions
                    st.subheader("Model Predictions")
                    model_df = pd.DataFrame(list(result['model_predictions'].items()), 
                                           columns=['Model', 'Fraud Probability'])
                    st.dataframe(model_df, use_container_width=True)
    
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
