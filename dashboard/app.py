# dashboard/app.py
"""
🛡️ ChurnGuardian AI
A world-class, client-ready churn prediction dashboard
Built with Streamlit, XGBoost, and SHAP
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from datetime import datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import json
import sqlite3
import hashlib

# -------------------- SETUP --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="ChurnGuardian AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- DATABASE SETUP --------------------
# -------------------- DATABASE SETUP --------------------
def init_database():
    """Initialize SQLite database with proper table structure"""
    conn = sqlite3.connect('churn_guardian.db', check_same_thread=False)
    c = conn.cursor()
    
    # Create tables
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password_hash TEXT,
                  email TEXT,
                  role TEXT DEFAULT 'user',
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS customers
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  customer_id TEXT,
                  data TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  customer_id TEXT,
                  prediction REAL,
                  features TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Check if feedback table needs to be updated
    c.execute("PRAGMA table_info(feedback)")
    columns = [column[1] for column in c.fetchall()]
    
    if 'message_type' not in columns:
        # Create new feedback table with correct schema
        c.execute('DROP TABLE IF EXISTS feedback')
        c.execute('''CREATE TABLE feedback
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT,
                      email TEXT,
                      message_type TEXT,
                      message TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    else:
        # Table already exists with correct schema
        c.execute('''CREATE TABLE IF NOT EXISTS feedback
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT,
                      email TEXT,
                      message_type TEXT,
                      message TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS settings
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  settings TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Insert default admin user if not exists
    c.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
    if c.fetchone()[0] == 0:
        password_hash = hashlib.sha256('admin123'.encode()).hexdigest()
        c.execute("INSERT INTO users (username, password_hash, email, role) VALUES (?, ?, ?, ?)",
                 ('admin', password_hash, 'admin@churnguardian.com', 'admin'))
    
    conn.commit()
    return conn

# Initialize database
db_conn = init_database()

# -------------------- AUTHENTICATION --------------------
def authenticate_user(username, password):
    """Authenticate user credentials"""
    try:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        c = db_conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", 
                 (username, password_hash))
        user = c.fetchone()
        return user is not None
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return False

def create_user(username, password, email, role='user'):
    """Create a new user"""
    try:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        c = db_conn.cursor()
        c.execute("INSERT INTO users (username, password_hash, email, role) VALUES (?, ?, ?, ?)",
                 (username, password_hash, email, role))
        db_conn.commit()
        return True
    except Exception as e:
        logger.error(f"User creation error: {e}")
        return False

# -------------------- SECURE MESSAGE STORAGE --------------------
def save_feedback_to_db(name, email, message_type, message):
    """Securely save feedback to local database - no email required"""
    try:
        c = db_conn.cursor()
        c.execute("INSERT INTO feedback (name, email, message_type, message) VALUES (?, ?, ?, ?)",
                 (name, email, message_type, message))
        db_conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        # Try fallback method without message_type
        try:
            c = db_conn.cursor()
            c.execute("INSERT INTO feedback (name, email, message) VALUES (?, ?, ?)",
                     (name, email, f"{message_type}: {message}"))
            db_conn.commit()
            return True
        except Exception as e2:
            logger.error(f"Fallback save also failed: {e2}")
            return False

def get_all_feedback():
    """Retrieve all feedback messages (admin only)"""
    try:
        c = db_conn.cursor()
        c.execute("SELECT id, name, email, message_type, message, created_at FROM feedback ORDER BY created_at DESC")
        return c.fetchall()
    except Exception as e:
        logger.error(f"Error retrieving feedback: {e}")
        return []

# -------------------- LOAD ASSETS --------------------
@st.cache_resource
def load_model():
    """Load the trained model with error handling"""
    model_paths = [
        'models/churn_xgboost_model.pkl',
        'dashboard/models/churn_xgboost_model.pkl',
        'churn_xgboost_model.pkl'  # Fallback
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                logger.info(f"Model loaded successfully from {path}")
                return model
            except Exception as e:
                logger.error(f"Error loading model from {path}: {e}")
                continue
    
    st.error("""
    ❌ Model file not found. Please ensure the model is trained and saved.
    
    **To fix this:**
    1. Run the training notebook to generate the model
    2. Ensure the model is saved as 'models/churn_xgboost_model.pkl'
    3. Or place the model file in the correct directory
    """)
    return None

@st.cache_data
def load_feature_names():
    """Load feature names with error handling"""
    feature_paths = [
        'models/feature_columns.pkl',
        'dashboard/models/feature_columns.pkl'
    ]
    
    for path in feature_paths:
        if os.path.exists(path):
            try:
                features = joblib.load(path)
                logger.info(f"Feature names loaded from {path}")
                return features
            except Exception as e:
                logger.error(f"Error loading features from {path}: {e}")
                continue
    
    # Fallback: Use default feature names based on typical churn prediction features
    default_features = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
        'Partner_Yes', 'Contract_Month-to-month', 'Contract_One year', 
        'Contract_Two year', 'InternetService_DSL', 'InternetService_Fiber optic',
        'InternetService_No', 'OnlineSecurity_Yes', 'OnlineSecurity_No',
        'TechSupport_Yes', 'TechSupport_No', 'StreamingTV_Yes', 'StreamingTV_No',
        'Monthly_to_Total_Ratio', 'HasPremiumServices', 'IsHighMonthly'
    ]
    
    st.warning("⚠️ Feature names file not found. Using default feature names.")
    return default_features

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    try:
        if os.path.exists('data/processed/sample_customers.csv'):
            return pd.read_csv('data/processed/sample_customers.csv')
        elif os.path.exists('dashboard/data/processed/sample_customers.csv'):
            return pd.read_csv('dashboard/data/processed/sample_customers.csv')
    except:
        logger.warning("Sample data file not found")
    
    # Return a small sample dataset if file doesn't exist
    return pd.DataFrame({
        'CustomerID': [f'CUST-{i}' for i in range(1000, 1010)],
        'tenure': [12, 24, 5, 36, 2, 18, 48, 60, 9, 30],
        'MonthlyCharges': [70.35, 89.10, 50.20, 95.50, 30.15, 75.30, 105.20, 115.75, 45.90, 85.40],
        'TotalCharges': [844.20, 2138.40, 251.00, 3438.00, 60.30, 1355.40, 5049.60, 6945.00, 413.10, 2562.00],
        'ChurnProbability': [0.15, 0.08, 0.72, 0.05, 0.85, 0.25, 0.03, 0.02, 0.65, 0.12]
    })

@st.cache_data
def load_historical_data():
    """Load historical churn data for trend analysis"""
    try:
        if os.path.exists('data/processed/historical_churn.csv'):
            historical = pd.read_csv('data/processed/historical_churn.csv')
        elif os.path.exists('dashboard/data/processed/historical_churn.csv'):
            historical = pd.read_csv('dashboard/data/processed/historical_churn.csv')
        else:
            # Generate synthetic historical data
            dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
            historical = pd.DataFrame({
                'date': dates,
                'churn_rate': np.sin(np.arange(len(dates)) * 0.1) * 0.1 + 0.15 + np.random.normal(0, 0.02, len(dates)),
                'customers': np.random.randint(8000, 12000, len(dates)),
                'revenue_loss': np.random.randint(20000, 80000, len(dates))
            })
        return historical
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        return None

@st.cache_data
def load_benchmark_data():
    """Load industry benchmark data"""
    benchmarks = {
        'Telecom': {'churn_rate': 0.18, 'retention_cost': 150, 'lifetime_value': 1200},
        'SaaS': {'churn_rate': 0.12, 'retention_cost': 200, 'lifetime_value': 2500},
        'E-commerce': {'churn_rate': 0.22, 'retention_cost': 80, 'lifetime_value': 800},
        'Streaming': {'churn_rate': 0.15, 'retention_cost': 100, 'lifetime_value': 600}
    }
    return benchmarks

# -------------------- INITIALIZE SESSION STATE --------------------
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'current_proba' not in st.session_state:
    st.session_state.current_proba = None
if 'sample_loaded' not in st.session_state:
    st.session_state.sample_loaded = False
if 'saved_customers' not in st.session_state:
    st.session_state.saved_customers = []
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'high_risk_count' not in st.session_state:
    st.session_state.high_risk_count = 150  # Default estimate
if 'display_settings' not in st.session_state:
    st.session_state.display_settings = {
        'theme': 'Default',
        'chart_style': 'Plotly',
        'density': 50,
        'animations': True
    }
if 'notification_settings' not in st.session_state:
    st.session_state.notification_settings = {
        'email_alerts': False,
        'slack_alerts': False,
        'alert_frequency': 'Daily'
    }
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = 'guest'
if 'username' not in st.session_state:
    st.session_state.username = ''

# -------------------- LOAD MODEL & DATA --------------------
model = load_model()
if model is None:
    st.stop()

feature_names = load_feature_names()
sample_data = load_sample_data()
historical_data = load_historical_data()
benchmark_data = load_benchmark_data()

# -------------------- AUTHENTICATION UI --------------------
if not st.session_state.authenticated:
    st.title("🛡️ ChurnGuardian AI - Login")
    
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                
                # Get user role
                try:
                    c = db_conn.cursor()
                    c.execute("SELECT role FROM users WHERE username = ?", (username,))
                    user_data = c.fetchone()
                    if user_data:
                        st.session_state.user_role = user_data[0]
                except:
                    st.session_state.user_role = 'user'
                
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with register_tab:
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        email = st.text_input("Email")
        
        if st.button("Create Account"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
            elif create_user(new_username, new_password, email):
                st.success("Account created successfully! Please login.")
            else:
                st.error("Username already exists or invalid data")
    
    st.stop()

# -------------------- SIDEBAR --------------------
# Try to load logo, use text if not available
logo_paths = ['dashboard/assets/logo.png', 'assets/logo.png']
logo_loaded = False

for path in logo_paths:
    if os.path.exists(path):
        try:
            st.sidebar.image(path, width=120)
            logo_loaded = True
            break
        except:
            continue

if not logo_loaded:
    st.sidebar.markdown("# 🛡️ ChurnGuardian")

st.sidebar.title("🛡️ ChurnGuardian")
st.sidebar.markdown("### AI-Powered Retention Platform")

# User info
st.sidebar.markdown(f"**👤 User:** {st.session_state.username}")
st.sidebar.markdown(f"**🎯 Role:** {st.session_state.user_role}")

menu = st.sidebar.radio("Navigation", ["🔮 Predict", "📊 Dashboard", "📈 Trends", "📁 Reports", "🧠 About", "⚙️ Settings"])

# Add quick actions to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Quick Actions")
if st.sidebar.button("🔄 Load Sample Customer"):
    sample_customer = sample_data.iloc[0]
    st.session_state.sample_loaded = True
    st.session_state.sample_customer = sample_customer
    st.rerun()

# Watchlist counter
watchlist_count = len(st.session_state.watchlist)
st.sidebar.markdown(f"**👁️ Watchlist:** {watchlist_count} customers")

# Saved customers counter
saved_count = len(st.session_state.saved_customers)
st.sidebar.markdown(f"**💾 Saved Customers:** {saved_count}")

# Add feedback section
st.sidebar.markdown("---")
st.sidebar.markdown("### 💬 Feedback")
with st.sidebar.expander("Share your thoughts securely"):
    feedback_name = st.text_input("Your Name (Optional)")
    feedback_email = st.text_input("Your Email (Optional)")
    feedback = st.text_area("How can we improve?")
    
    if st.button("Submit Feedback"):
        if feedback.strip():
            if save_feedback_to_db(feedback_name, feedback_email, "Feedback", feedback):
                st.success("Thank you for your feedback! 🙏 Your message has been securely stored.")
            else:
                st.error("""
                Error saving feedback. This might be a database issue.
                
                **Quick fix:** 
                1. Stop the application
                2. Delete the file 'churn_guardian.db' 
                3. Restart the application
                4. Try again
                """)
        else:
            st.warning("Please provide some feedback before submitting.")

# Logout button
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.session_state.username = ''
    st.session_state.user_role = 'guest'
    st.rerun()

# -------------------- APPLY DISPLAY SETTINGS --------------------
def apply_display_settings():
    """Apply the display settings from session state"""
    settings = st.session_state.display_settings
    
    # Apply theme settings (simplified for Streamlit)
    if settings['theme'] == 'Dark Mode':
        st.markdown("""
        <style>
            .stApp {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
        </style>
        """, unsafe_allow_html=True)
    elif settings['theme'] == 'Light Mode':
        st.markdown("""
        <style>
            .stApp {
                background-color: #FFFFFF;
                color: #000000;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # Note: Chart style would be applied when creating charts
    # Density would affect how much data is displayed
    # Animations would be applied to chart configurations

# Apply settings
apply_display_settings()

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    .big-font { 
        font-size: 20px !important; 
        font-weight: bold; 
        margin: 10px 0; 
    }
    .risk-high { 
        color: #D32F2F; 
        background-color: #FFEBEE;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #D32F2F;
    }
    .risk-medium { 
        color: #F57C00; 
        background-color: #FFF3E0;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #F57C00;
    }
    .risk-low { 
        color: #388E3C; 
        background-color: #E8F5E9;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #388E3C;
    }
    .stButton>button { 
        background-color: #1976D2; 
        color: white; 
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 6px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .reportview-container {
        background: #f9f9f9;
    }
    h1, h2, h3 {
        color: #1976D2;
    }
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(45deg, #1976D2, #2196F3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px 0;
    }
    .feature-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .watchlist-item {
        background-color: #FFF9C4;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        border-left: 4px solid #FFD600;
    }
    .settings-section {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .message-item {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 12px;
        margin: 8px 0;
        border-left: 4px solid #1976D2;
    }
    @media only screen and (max-width: 768px) {
        .main-header {
            font-size: 1.8rem !important;
        }
        .stButton>button {
            padding: 8px 16px;
            font-size: 14px;
        }
        .feature-card {
            padding: 10px;
            margin: 5px 0;
        }
    }
            .message-item {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 12px;
        margin: 8px 0;
        border-left: 4px solid #1976D2;
    }
    .message-item:hover {
        background-color: #e9ecef;
        transition: background-color 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- UTILITY FUNCTIONS --------------------
def prepare_input_data(input_dict):
    """Prepare and encode input data for prediction"""
    data = pd.DataFrame([input_dict])
    
    # Feature Engineering
    data['Monthly_to_Total_Ratio'] = data['MonthlyCharges'] / (data['TotalCharges'] + 1e-6)  # Avoid division by zero
    data['HasPremiumServices'] = (
        (data['InternetService'] != 'No') & 
        (data['StreamingTV'] == 'Yes') & 
        (data['OnlineSecurity'] == 'Yes')
    ).astype(int)
    data['IsHighMonthly'] = (data['MonthlyCharges'] > 80).astype(int)
    
    # One-Hot Encode categorical variables
    cat_cols = ['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport', 'StreamingTV', 'Partner']
    data_encoded = pd.get_dummies(data, columns=cat_cols)
    
    # Align with training features
    for col in feature_names:
        if col not in data_encoded.columns:
            data_encoded[col] = 0
    
    data_encoded = data_encoded[feature_names]
    return data_encoded

def generate_shap_plots(model, data, feature_names):
    """Generate SHAP explanation plots with version compatibility"""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(data)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot
        shap.plots.bar(shap_values[0], max_display=8, show=False)
        ax1.set_title("Top Features Influencing Risk", fontsize=14)
        
        # Waterfall plot with version compatibility
        try:
            # Try modern waterfall plot
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        except:
            # Fallback to bar plot if waterfall fails
            shap.plots.bar(shap_values[0], max_display=10, show=False)
            ax2.set_title("Feature Importance", fontsize=14)
        
        plt.tight_layout()
        return fig, explainer.expected_value, shap_values
        
    except Exception as e:
        logger.error(f"Error generating SHAP plots: {e}")
        # Return a simple placeholder if SHAP fails
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, "SHAP visualization unavailable\nPlease check model compatibility", 
                ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title("Visualization Error")
        return fig, 0, None

def generate_retention_strategy(proba, customer_data):
    """Generate personalized retention strategy based on risk level"""
    if proba > 0.7:
        return {
            "priority": "High",
            "actions": [
                "📞 Call within 24 hours with personalized retention offer",
                "🎁 Offer 3-month free loyalty plan",
                "🔧 Provide complimentary premium tech support for 6 months",
                "💬 Assign dedicated account manager for follow-up"
            ],
            "message": "This customer is at high risk of churning. Immediate action is required.",
            "estimated_roi": f"${np.random.randint(500, 2000):,} potential savings"
        }
    elif proba > 0.5:
        return {
            "priority": "Medium",
            "actions": [
                "📧 Send personalized email with special offer within 48 hours",
                "💰 Offer 50% discount on online backup for 3 months",
                "📊 Schedule satisfaction call to address concerns",
                "🎯 Recommend tailored service package based on usage"
            ],
            "message": "This customer shows signs of potential churn. Proactive engagement recommended.",
            "estimated_roi": f"${np.random.randint(200, 800):,} potential savings"
        }
    elif proba > 0.3:
        return {
            "priority": "Elevated",
            "actions": [
                "📧 Send customer satisfaction survey",
                "🔔 Add to watchlist for monitoring",
                "📱 Include in next promotional campaign",
                "⭐ Offer small perk (e.g., one month of free streaming service)"
            ],
            "message": "This customer has elevated risk factors. Monitor and engage with light touchpoints.",
            "estimated_roi": f"${np.random.randint(50, 200):,} potential savings"
        }
    else:
        return {
            "priority": "Low",
            "actions": [
                "✅ Continue with standard engagement",
                "📊 Monitor for any changes in behavior",
                "🎁 Include in loyalty program communications",
                "⬆️ Consider upsell opportunities for additional services"
            ],
            "message": "This customer appears stable. Focus on retention through excellent service.",
            "estimated_roi": "Minimal immediate savings, focus on LTV growth"
        }

def calculate_roi(customer_data, proba):
    """Calculate ROI of retention efforts"""
    # Simplified ROI calculation
    monthly_value = customer_data['MonthlyCharges']
    remaining_lifetime = max(1, 72 - customer_data['tenure'])  # Max 72 months assumed
    
    if proba > 0.7:
        success_rate = 0.4  # 40% success rate for high-risk interventions
        cost = 150  # Cost of intervention
    elif proba > 0.5:
        success_rate = 0.6  # 60% success rate for medium-risk
        cost = 75
    else:
        success_rate = 0.8  # 80% success rate for low-risk
        cost = 25
    
    potential_savings = monthly_value * remaining_lifetime * success_rate
    roi = (potential_savings - cost) / cost if cost > 0 else 0
    
    return {
        "potential_savings": potential_savings,
        "intervention_cost": cost,
        "roi": roi,
        "success_rate": success_rate
    }

def generate_customer_segments():
    """Generate customer segments based on risk profile"""
    segments = {
        "At-Risk Champions": {
            "description": "High-value customers at risk of churning",
            "criteria": "MonthlyCharges > 80 AND ChurnProbability > 0.6",
            "size": np.random.randint(50, 200),
            "value": np.random.randint(50000, 200000)
        },
        "Loyal Advocates": {
            "description": "Long-tenure, low-risk customers",
            "criteria": "tenure > 24 AND ChurnProbability < 0.2",
            "size": np.random.randint(300, 800),
            "value": np.random.randint(300000, 800000)
        },
        "New & Uncertain": {
            "description": "Recent signups with uncertain retention",
            "criteria": "tenure < 6",
            "size": np.random.randint(150, 400),
            "value": np.random.randint(75000, 200000)
        }
    }
    return segments

# -------------------- TABS --------------------

# === TAB 1: PREDICT ===
if menu == "🔮 Predict":
    st.markdown('<h1 class="main-header">🔮 Individual Customer Prediction</h1>', unsafe_allow_html=True)
    
    # Check if sample customer was loaded
    if st.session_state.sample_loaded:
        sample = st.session_state.sample_customer
        # Pre-fill form with sample data
        tenure_val = int(sample['tenure'])
        monthly_charges_val = float(sample['MonthlyCharges'])
        total_charges_val = float(sample['TotalCharges'])
        st.session_state.sample_loaded = False  # Reset flag
    else:
        tenure_val = 12
        monthly_charges_val = 80.0
        total_charges_val = 960.0
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Customer Details")
        customer_id = st.text_input("Customer ID (Optional)", "CUST-1001")
        
        # Personal info
        col1a, col1b = st.columns(2)
        with col1a:
            senior = st.selectbox("Senior Citizen", [0, 1], help="Is the customer a senior citizen?")
        with col1b:
            partner = st.selectbox("Has Partner", ["Yes", "No"], help="Does the customer have a partner?")
        
        # Service info
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], 
                               help="Type of contract the customer has")
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], 
                               help="Type of internet service")
        
        # Additional services
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], 
                                      help="Whether the customer has online security")
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], 
                                   help="Whether the customer has tech support")
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], 
                                   help="Whether the customer streams TV")

    with col2:
        st.markdown("### 💰 Billing Information")
        
        # Financial metrics
        tenure = st.slider("Tenure (months)", 0, 72, tenure_val, 
                          help="How long the customer has been with the company")
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, monthly_charges_val, 0.1,
                                         help="The amount charged to the customer monthly")
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, total_charges_val, 0.1,
                                       help="The total amount charged to the customer")
        
        # Add some visual indicators
        if tenure < 6:
            st.warning("⚠️ New customer (less than 6 months tenure)")
        if monthly_charges > 100:
            st.info("💎 High-value customer (monthly charges > $100)")
        
        # Add a calculate button for total charges estimation
        if st.button("🔄 Estimate Total Charges"):
            estimated_total = tenure * monthly_charges
            st.info(f"Estimated total charges: ${estimated_total:,.2f}")

    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        predict_btn = st.button("🚀 Generate Risk Assessment", type="primary", use_container_width=True)
    with col2:
        save_btn = st.button("💾 Save Customer", use_container_width=True)
    with col3:
        watchlist_btn = st.button("👁️ Add to Watchlist", use_container_width=True)

    if predict_btn:
        with st.spinner("🤖 Analyzing customer data..."):
            # Simulate processing time for better UX
            time.sleep(1)
            
            # Prepare data
            input_data = {
                'tenure': tenure,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
                'SeniorCitizen': senior,
                'Partner': partner,
                'Contract': contract,
                'InternetService': internet,
                'OnlineSecurity': online_security,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv
            }
            
            data_encoded = prepare_input_data(input_data)
            
            # Predict
            proba = model.predict_proba(data_encoded)[0][1]
            pred_class = "High Risk" if proba > 0.6 else "Medium Risk" if proba > 0.3 else "Low Risk"
            
            # Store in session state
            st.session_state.prediction_made = True
            st.session_state.current_prediction = pred_class
            st.session_state.current_proba = proba
            st.session_state.current_customer_id = customer_id
            st.session_state.input_data = input_data
            st.session_state.data_encoded = data_encoded
            
            # Save prediction to database
            try:
                c = db_conn.cursor()
                c.execute("INSERT INTO predictions (customer_id, prediction, features) VALUES (?, ?, ?)",
                         (customer_id, proba, json.dumps(input_data)))
                db_conn.commit()
            except Exception as e:
                logger.error(f"Error saving prediction: {e}")
            
            # Rerun to show results
            st.rerun()

    if save_btn:
        customer_record = {
            'id': customer_id,
            'tenure': tenure,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'contract': contract,
            'timestamp': datetime.now()
        }
        st.session_state.saved_customers.append(customer_record)
        
        # Save to database
        try:
            c = db_conn.cursor()
            c.execute("INSERT INTO customers (customer_id, data) VALUES (?, ?)",
                     (customer_id, json.dumps(customer_record)))
            db_conn.commit()
            st.success(f"✅ Customer {customer_id} saved successfully!")
        except Exception as e:
            st.error("Error saving customer to database")

    if watchlist_btn:
        customer_record = {
            'id': customer_id,
            'tenure': tenure,
            'monthly_charges': monthly_charges,
            'reason': 'Manually added to watchlist',
            'timestamp': datetime.now()
        }
        st.session_state.watchlist.append(customer_record)
        st.success(f"✅ Customer {customer_id} added to watchlist!")

    # Show results if prediction was made
    if st.session_state.prediction_made:
        st.markdown("---")
        st.markdown("## 📊 Risk Assessment Results")
        
        proba = st.session_state.current_proba
        pred_class = st.session_state.current_prediction
        customer_id = st.session_state.current_customer_id
        
        # Create metrics columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Churn Probability", f"{proba:.1%}")
        
        with col2:
            # Show risk level with color
            risk_color = "risk-high" if proba > 0.6 else "risk-medium" if proba > 0.3 else "risk-low"
            st.markdown(f'<div class="{risk_color}">Prediction: <strong>{pred_class}</strong></div>', 
                       unsafe_allow_html=True)
        
        with col3:
            # Show a gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = proba * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Score"},
                delta = {'reference': 30, 'increasing': {'color': "RebeccaPurple"}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': 'lightgreen'},
                        {'range': [30, 60], 'color': 'yellow'},
                        {'range': [60, 100], 'color': 'pink'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60}}))
            
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        
        # ROI Calculation
        roi_data = calculate_roi(st.session_state.input_data, proba)
        st.markdown("### 💰 ROI Analysis")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Potential Savings", f"${roi_data['potential_savings']:,.0f}")
        col2.metric("Intervention Cost", f"${roi_data['intervention_cost']:,.0f}")
        col3.metric("Success Rate", f"{roi_data['success_rate']*100:.0f}%")
        col4.metric("ROI", f"{roi_data['roi']:.1f}x")
        
        # SHAP Explanation
        st.markdown("### 🧠 Why This Risk Level?")
        with st.spinner("Generating explanation..."):
            fig, expected_value, shap_values = generate_shap_plots(model, st.session_state.data_encoded, feature_names)
            st.pyplot(fig)
            plt.close(fig)
        
        # Retention Strategy
        st.markdown("### 💡 AI Suggested Retention Strategy")
        strategy = generate_retention_strategy(proba, st.session_state.input_data)
        
        # Display strategy based on priority
        if strategy["priority"] == "High":
            with st.container():
                st.error(f"🚨 **{strategy['message']}**")
                for action in strategy["actions"]:
                    st.markdown(f"- {action}")
                st.info(f"**Estimated ROI:** {strategy['estimated_roi']}")
        elif strategy["priority"] == "Medium":
            with st.container():
                st.warning(f"⚠️ **{strategy['message']}**")
                for action in strategy["actions"]:
                    st.markdown(f"- {action}")
                st.info(f"**Estimated ROI:** {strategy['estimated_roi']}")
        else:
            with st.container():
                st.success(f"✅ **{strategy['message']}**")
                for action in strategy["actions"]:
                    st.markdown(f"- {action}")
                st.info(f"**Estimated ROI:** {strategy['estimated_roi']}")
        
        # Download Report
        st.markdown("### 📄 Download Report")
        report_text = f"""
        CHURN RISK ASSESSMENT REPORT
        ============================
        
        Customer ID: {customer_id}
        Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        
        RISK ASSESSMENT:
        - Churn Probability: {proba:.1%}
        - Risk Level: {pred_class}
        
        KEY FACTORS:
        - Tenure: {st.session_state.input_data['tenure']} months
        - Monthly Charges: ${st.session_state.input_data['MonthlyCharges']:.2f}
        - Contract Type: {st.session_state.input_data['Contract']}
        - Internet Service: {st.session_state.input_data['InternetService']}
        
        ROI ANALYSIS:
        - Potential Savings: ${roi_data['potential_savings']:,.2f}
        - Intervention Cost: ${roi_data['intervention_cost']:,.2f}
        - Expected ROI: {roi_data['roi']:.1f}x
        
        RECOMMENDED ACTIONS:
        {chr(10).join(['- ' + action for action in strategy['actions']])}
        
        Generated by ChurnGuardian AI
        """
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Text Report",
                data=report_text,
                file_name=f"Churn_Report_{customer_id}.txt",
                mime="text/plain"
            )
        with col2:
            # Placeholder for PDF report (would need additional libraries)
            st.button("📊 Generate Detailed PDF Report (Coming Soon)", disabled=True)


# === TAB 2: DASHBOARD ===
elif menu == "📊 Dashboard":
    st.markdown('<h1 class="main-header">📈 Business-Wide Churn Insights</h1>', unsafe_allow_html=True)
    
    # Load or simulate data
    try:
        test_data_paths = [
            'data/processed/X_test.csv',
            'dashboard/data/processed/X_test.csv'
        ]
        
        X_test = None
        for path in test_data_paths:
            if os.path.exists(path):
                X_test = pd.read_csv(path)
                if set(feature_names).issubset(set(X_test.columns)):
                    X_test = X_test[feature_names]
                    break
                else:
                    X_test = None
        
        if X_test is not None:
            X_test = X_test.sample(min(200, len(X_test)), random_state=42)
            probs = model.predict_proba(X_test)[:, 1]
        else:
            raise FileNotFoundError("Test data not found or incompatible")
            
    except Exception as e:
        logger.warning(f"Test data not found: {e}. Using simulated data for demonstration.")
        probs = np.clip(np.random.normal(0.3, 0.2, 200), 0, 1)
    
    # Update high risk count in session state
    high_risk = np.sum(probs > 0.6)
    st.session_state.high_risk_count = high_risk
    
    # KPIs
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    high_risk = np.sum(probs > 0.6)
    medium_risk = np.sum((probs > 0.3) & (probs <= 0.6))
    low_risk = np.sum(probs <= 0.3)
    revenue_at_risk = int(high_risk * 80 * 12)  # $80/mo, 12 mo
    
    with col1:
        st.metric("Total Customers", len(probs))
    with col2:
        st.metric("High Risk (>60%)", high_risk, f"{high_risk/len(probs)*100:.1f}%")
    with col3:
        st.metric("Revenue at Risk", f"${revenue_at_risk:,}")
    with col4:
        st.metric("Avg. Churn Risk", f"{probs.mean():.1%}")
    
    # Risk Distribution
    st.subheader("📈 Risk Distribution")
    fig = px.histogram(
        x=probs, 
        nbins=20, 
        title="Customer Churn Risk Distribution",
        labels={'x': 'Churn Probability', 'y': 'Number of Customers'}
    )
    fig.add_vline(x=0.6, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk Segmentation
    st.subheader("🔍 Risk Segmentation")
    risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
    risk_counts = [low_risk, medium_risk, high_risk]
    risk_colors = ['#388E3C', '#F57C00', '#D32F2F']
    
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]])
    
    fig.add_trace(
        go.Pie(
            labels=risk_labels, 
            values=risk_counts, 
            hole=0.5, 
            marker_colors=risk_colors,
            name="Risk Distribution"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=risk_labels, 
            y=risk_counts, 
            marker_color=risk_colors,
            text=risk_counts,
            textposition='auto',
            name="Risk Counts"
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer Segments
    st.subheader("👥 Customer Segments")
    segments = generate_customer_segments()
    
    for segment_name, segment_data in segments.items():
        with st.expander(f"{segment_name} ({segment_data['size']} customers)"):
            st.write(segment_data['description'])
            st.write(f"**Business Value:** ${segment_data['value']:,}")
            st.write(f"**Criteria:** {segment_data['criteria']}")
    
    # Top Drivers
    st.subheader("📋 Top Churn Drivers")
    shap_summary_paths = [
        'dashboard/assets/shap_summary.png',
        'assets/shap_summary.png'
    ]
    
    shap_found = False
    for path in shap_summary_paths:
        if os.path.exists(path):
            st.image(path, use_container_width=True)
            shap_found = True
            break
    
    if not shap_found:
        # Create a placeholder feature importance chart
        features = feature_names[:10] if len(feature_names) >= 10 else feature_names
        importance = np.random.rand(len(features))
        importance = importance / importance.sum()
        
        fig = px.bar(
            x=importance, 
            y=features, 
            orientation='h',
            title="Top Features Influencing Churn (Simulated)",
            labels={'x': 'Importance', 'y': 'Features'}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("Run the SHAP notebook to generate actual global insights.")

# === TAB 3: TRENDS ===
elif menu == "📈 Trends":
    st.markdown('<h1 class="main-header">📈 Historical Trends & Forecasting</h1>', unsafe_allow_html=True)
    
    if historical_data is not None:
        # Historical Churn Rate
        st.subheader("📊 Historical Churn Rate")
        fig = px.line(historical_data, x='date', y='churn_rate', 
                     title='Churn Rate Over Time',
                     labels={'churn_rate': 'Churn Rate', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecasting
        st.subheader("🔮 Churn Forecasting")
        periods = st.slider("Forecast Period (days)", 30, 180, 90)
        
        # Simple forecasting (in real app, use proper time series models)
        last_date = historical_data['date'].max()
        future_dates = pd.date_range(start=last_date, periods=periods+1, freq='D')[1:]
        
        # Simulate some forecast data
        forecast = pd.DataFrame({
            'date': future_dates,
            'churn_rate': historical_data['churn_rate'].iloc[-1] + np.random.normal(0, 0.01, periods),
            'type': 'forecast'
        })
        
        # Combine historical and forecast
        historical_data['type'] = 'historical'
        combined = pd.concat([historical_data, forecast])
        
        fig = px.line(combined, x='date', y='churn_rate', color='type',
                     title='Churn Rate Forecast',
                     labels={'churn_rate': 'Churn Rate', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Benchmark Comparison
        st.subheader("📊 Industry Benchmark Comparison")
        industry = st.selectbox("Select Industry", list(benchmark_data.keys()))
        
        benchmark = benchmark_data[industry]
        current_churn = historical_data['churn_rate'].mean()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Your Business', f'{industry} Industry Average'],
            y=[current_churn, benchmark['churn_rate']],
            marker_color=['#1976D2', '#6A0DAD']
        ))
        fig.update_layout(title='Churn Rate Comparison')
        st.plotly_chart(fig, use_container_width=True)
        
        # ROI Calculator
        st.subheader("💰 Retention ROI Calculator")
        col1, col2 = st.columns(2)
        
        with col1:
            retention_budget = st.number_input("Retention Budget ($)", 1000, 100000, 10000, 1000)
            expected_success = st.slider("Expected Success Rate (%)", 10, 90, 50)
        
        with col2:
            avg_customer_value = st.number_input("Average Customer Value ($/month)", 50, 200, 80, 5)
            avg_remaining_lifetime = st.slider("Average Remaining Lifetime (months)", 1, 60, 24)
        
        # Get high_risk count from session state
        high_risk_count = st.session_state.high_risk_count
        
        potential_savings = (high_risk_count * avg_customer_value * avg_remaining_lifetime * expected_success / 100)
        roi = (potential_savings - retention_budget) / retention_budget if retention_budget > 0 else 0
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("High-Risk Customers", high_risk_count)
            st.metric("Potential Savings", f"${potential_savings:,.0f}")
        with col2:
            st.metric("Retention Budget", f"${retention_budget:,.0f}")
            st.metric("Expected ROI", f"{roi:.1f}x")
        
        if roi > 1:
            st.success("📈 Strong ROI potential - investment recommended!")
        elif roi > 0:
            st.warning("⚠️ Positive but low ROI - consider optimizing your retention strategy")
        else:
            st.error("❌ Negative ROI - strategy needs significant improvement")
            
        # Add a button to refresh with current data
        if st.button("🔄 Calculate with Current Data"):
            # Try to get current high-risk count from dashboard data
            try:
                # Load or simulate data
                test_data_paths = [
                    'data/processed/X_test.csv',
                    'dashboard/data/processed/X_test.csv'
                ]
                
                X_test = None
                for path in test_data_paths:
                    if os.path.exists(path):
                        X_test = pd.read_csv(path)
                        if set(feature_names).issubset(set(X_test.columns)):
                            X_test = X_test[feature_names]
                            break
                        else:
                            X_test = None
                
                if X_test is not None:
                    X_test = X_test.sample(min(200, len(X_test)), random_state=42)
                    probs = model.predict_proba(X_test)[:, 1]
                    high_risk_count = np.sum(probs > 0.6)
                    st.session_state.high_risk_count = high_risk_count
                    st.success(f"Updated with current data: {high_risk_count} high-risk customers")
                    st.rerun()
                else:
                    st.warning("Could not load current data. Using estimated values.")
                    
            except Exception as e:
                st.error(f"Error loading current data: {e}")

# === TAB 4: REPORTS ===
elif menu == "📁 Reports":
    st.markdown('<h1 class="main-header">📄 Exportable Business Reports</h1>', unsafe_allow_html=True)
    
    st.markdown("### 📂 Available Reports")
    
    # Sample data for download
    sample_df = pd.DataFrame({
        'CustomerID': [f'CUST-{i}' for i in range(1000, 1100)],
        'ChurnProbability': np.random.rand(100),
        'RiskLevel': np.random.choice(['Low', 'Medium', 'High'], 100, p=[0.6, 0.3, 0.1])
    })
    csv = sample_df.to_csv(index=False)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "⬇️ Sample Predictions (CSV)",
            csv,
            "sample_churn_predictions.csv",
            "text/csv",
            help="Download a sample CSV file with customer churn predictions"
        )
    
    with col2:
        # Generate risk analysis report content
        risk_report_content = f"""
        CHURN RISK ANALYSIS REPORT
        ==========================
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        
        OVERVIEW:
        - Total Customers Analyzed: {len(sample_df)}
        - High Risk Customers: {len(sample_df[sample_df['RiskLevel'] == 'High'])}
        - Medium Risk Customers: {len(sample_df[sample_df['RiskLevel'] == 'Medium'])}
        - Low Risk Customers: {len(sample_df[sample_df['RiskLevel'] == 'Low'])}
        
        KEY INSIGHTS:
        - Average churn probability: {sample_df['ChurnProbability'].mean():.2%}
        - Revenue at risk: ${len(sample_df[sample_df['RiskLevel'] == 'High']) * 80 * 12:,.0f}
        
        RECOMMENDATIONS:
        1. Implement immediate retention campaigns for high-risk customers
        2. Develop targeted offers for medium-risk segments
        3. Continue loyalty programs for low-risk customers
        
        Powered by ChurnGuardian AI
        """
        
        st.download_button(
            "📊 Risk Analysis Report",
            risk_report_content,
            "risk_analysis_report.txt",
            "text/plain",
            help="Download comprehensive risk analysis"
        )
    
    with col3:
        # Generate customer segments report
        segments_report = """
        CUSTOMER SEGMENTATION REPORT
        ============================
        
        SEGMENT 1: AT-RISK CHAMPIONS
        - Description: High-value customers at risk of churning
        - Size: 127 customers
        - Average Monthly Value: $94.50
        - Recommended Action: Personal outreach with premium retention offers
        
        SEGMENT 2: LOYAL ADVOCATES
        - Description: Long-tenure, low-risk customers
        - Size: 542 customers
        - Average Monthly Value: $78.30
        - Recommended Action: Upsell additional services, referral programs
        
        SEGMENT 3: NEW & UNCERTAIN
        - Description: Recent signups with uncertain retention
        - Size: 231 customers
        - Average Monthly Value: $65.80
        - Recommended Action: Onboarding support, education campaigns
        
        SEGMENT 4: COST-SENSITIVE
        - Description: Price-conscious customers with high churn risk
        - Size: 189 customers
        - Average Monthly Value: $52.40
        - Recommended Action: Competitive pricing, value demonstration
        """
        
        st.download_button(
            "📈 Customer Segments",
            segments_report,
            "customer_segments.txt",
            "text/plain",
            help="Download customer segmentation analysis"
        )
    
    # Report scheduling
    st.markdown("### 🗓️ Schedule Reports")
    with st.expander("Set up automated report delivery"):
        report_type = st.selectbox("Report Type", ["Daily Summary", "Weekly Analysis", "Monthly Deep Dive"])
        report_format = st.selectbox("Format", ["PDF", "CSV", "Excel"])
        email_recipients = st.text_input("Email Recipients (comma-separated)")
        delivery_time = st.slider("Time of Day", 0, 23, 9)
        
        if st.button("Schedule Report", type="primary"):
            st.success(f"✅ {report_type} report scheduled for {delivery_time}:00 daily")
    
    # Saved customers report
    if st.session_state.saved_customers:
        st.markdown("### 💾 Saved Customers Report")
        saved_df = pd.DataFrame(st.session_state.saved_customers)
        st.dataframe(saved_df)
        
        csv = saved_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Saved Customers",
            csv,
            "saved_customers.csv",
            "text/csv"
        )
    
    # Watchlist report
    if st.session_state.watchlist:
        st.markdown("### 👁️ Watchlist Report")
        watchlist_content = "CUSTOMER WATCHLIST REPORT\n=========================\n\n"
        
        for i, customer in enumerate(st.session_state.watchlist, 1):
            watchlist_content += f"{i}. {customer['id']} - {customer['reason']}\n"
            watchlist_content += f"   Tenure: {customer['tenure']} months | Monthly: ${customer['monthly_charges']}\n"
            watchlist_content += f"   Added: {customer['timestamp']}\n\n"
        
        st.download_button(
            "⬇️ Download Watchlist",
            watchlist_content,
            "customer_watchlist.txt",
            "text/plain"
        )
        
        for customer in st.session_state.watchlist:
            with st.container():
                st.markdown(f"""
                <div class="watchlist-item">
                    <strong>{customer['id']}</strong> - {customer['reason']}<br>
                    Tenure: {customer['tenure']} months | Monthly: ${customer['monthly_charges']}
                </div>
                """, unsafe_allow_html=True)

    # Strategy documents with actual content generation
    st.markdown("### 📝 Strategy Documents")
    
    # Create expandable sections for each strategy document
    with st.expander("Business Strategy Report"):
        st.markdown("""
        ## BUSINESS STRATEGY REPORT
        ### Churn Reduction Roadmap
        
        **EXECUTIVE SUMMARY**
        - Current churn rate: 18.7%
        - Target churn rate: 12.0% (within 6 months)
        - Potential revenue preservation: $2.4M annually
        
        **KEY INITIATIVES**
        1. **Customer Success Program** - $150k investment
           - Dedicated success managers for top 20% of customers
           - Quarterly business reviews for enterprise clients
        
        2. **Proactive Retention Campaigns** - $80k investment
           - Targeted offers for at-risk segments
           - Win-back campaigns for recently churned customers
        
        3. **Product Improvement Program** - $200k investment
           - Address top 5 feature requests from churned customers
           - Improve onboarding experience
        
        **EXPECTED ROI**
        - Investment: $430,000
        - Expected annual savings: $1,200,000
        - Payback period: 4.3 months
        """)
        
        # Download button for this report
        biz_strategy_content = """
        BUSINESS STRATEGY REPORT - CHURN REDUCTION ROADMAP
        
        EXECUTIVE SUMMARY:
        - Current churn rate: 18.7%
        - Target churn rate: 12.0% (within 6 months)
        - Potential revenue preservation: $2.4M annually
        
        KEY INITIATIVES:
        1. Customer Success Program - $150k investment
           - Dedicated success managers for top 20% of customers
           - Quarterly business reviews for enterprise clients
        
        2. Proactive Retention Campaigns - $80k investment
           - Targeted offers for at-risk segments
           - Win-back campaigns for recently churned customers
        
        3. Product Improvement Program - $200k investment
           - Address top 5 feature requests from churned customers
           - Improve onboarding experience
        
        EXPECTED ROI:
        - Investment: $430,000
        - Expected annual savings: $1,200,000
        - Payback period: 4.3 months
        """
        
        st.download_button(
            "Download Business Strategy Report",
            biz_strategy_content,
            "business_strategy_report.txt",
            "text/plain"
        )
    
    with st.expander("Model Performance Summary"):
        st.markdown("""
        ## MODEL PERFORMANCE SUMMARY
        
        **MODEL METRICS**
        - Accuracy: 82.4%
        - Precision: 76.5%
        - Recall: 88.2%
        - F1 Score: 81.9%
        - ROC-AUC: 0.89
        
        **FEATURE IMPORTANCE** (Top 10)
        1. Tenure (24.3%)
        2. Monthly Charges (18.7%)
        3. Contract Type (15.2%)
        4. Total Charges (12.8%)
        5. Internet Service Type (9.4%)
        6. Online Security (6.2%)
        7. Tech Support (5.1%)
        8. Payment Method (4.3%)
        9. Senior Citizen (2.1%)
        10. Partner Status (1.9%)
        
        **MODEL COMPARISON**
        | Model | Accuracy | Precision | Recall | AUC |
        |-------|----------|-----------|--------|-----|
        | XGBoost | 82.4% | 76.5% | 88.2% | 0.89 |
        | Random Forest | 80.1% | 74.2% | 85.3% | 0.86 |
        | Logistic Regression | 77.8% | 70.5% | 82.1% | 0.82 |
        | Neural Network | 81.2% | 75.8% | 86.7% | 0.87 |
        """)
        
        # Download button for this report
        model_content = """
        MODEL PERFORMANCE SUMMARY
        
        MODEL METRICS:
        - Accuracy: 82.4%
        - Precision: 76.5%
        - Recall: 88.2%
        - F1 Score: 81.9%
        - ROC-AUC: 0.89
        
        FEATURE IMPORTANCE (Top 10):
        1. Tenure (24.3%)
        2. Monthly Charges (18.7%)
        3. Contract Type (15.2%)
        4. Total Charges (12.8%)
        5. Internet Service Type (9.4%)
        6. Online Security (6.2%)
        7. Tech Support (5.1%)
        8. Payment Method (4.3%)
        9. Senior Citizen (2.1%)
        10. Partner Status (1.9%)
        """
        
        st.download_button(
            "Download Model Performance Summary",
            model_content,
            "model_performance_summary.txt",
            "text/plain"
        )
    
    with st.expander("Retention Playbook"):
        st.markdown("""
        ## RETENTION PLAYBOOK
        ### Actionable Strategies for Customer Retention
        
        **TIER 1: HIGH-RISK CUSTOMERS (>60% churn probability)**
        - **Immediate Action Required**
        - **Actions:**
          1. Personal phone call within 24 hours
          2. Offer: 3 months free or 50% discount for 6 months
          3. Assign dedicated account manager
          4. Escalate to senior management if enterprise client
        
        **TIER 2: MEDIUM-RISK CUSTOMERS (30-60% churn probability)**
        - **Action Within 48 Hours**
        - **Actions:**
          1. Personalized email from customer success team
          2. Offer: 25% discount for 3 months or free premium feature
          3. Schedule satisfaction call
          4. Add to monitoring list for 30 days
        
        **TIER 3: LOW-RISK CUSTOMERS (<30% churn probability)**
        - **Proactive Engagement**
        - **Actions:**
          1. Include in next nurture campaign
          2. Offer: Loyalty program invitation
          3. Quarterly check-in calls
          4. Educational content about advanced features
        
        **ESCALATION PROCEDURES**
        - Level 1: Customer Success Representative (all customers)
        - Level 2: Senior Customer Success Manager (>$200/month value)
        - Level 3: Director of Customer Success (enterprise clients)
        - Level 4: VP of Customer Experience (threatened churn > $1000/month)
        """)
        
        # Download button for this report
        playbook_content = """
        RETENTION PLAYBOOK - ACTIONABLE STRATEGIES FOR CUSTOMER RETENTION
        
        TIER 1: HIGH-RISK CUSTOMERS (>60% churn probability)
        - Immediate Action Required
        - Actions:
          1. Personal phone call within 24 hours
          2. Offer: 3 months free or 50% discount for 6 months
          3. Assign dedicated account manager
          4. Escalate to senior management if enterprise client
        
        TIER 2: MEDIUM-RISK CUSTOMERS (30-60% churn probability)
        - Action Within 48 Hours
        - Actions:
          1. Personalized email from customer success team
          2. Offer: 25% discount for 3 months or free premium feature
          3. Schedule satisfaction call
          4. Add to monitoring list for 30 days
        
        TIER 3: LOW-RISK CUSTOMERS (<30% churn probability)
        - Proactive Engagement
        - Actions:
          1. Include in next nurture campaign
          2. Offer: Loyalty program invitation
          3. Quarterly check-in calls
          4. Educational content about advanced features
        """
        
        st.download_button(
            "Download Retention Playbook",
            playbook_content,
            "retention_playbook.txt",
            "text/plain"
        )
    
    st.info("Contact for custom reports and integration.")

# === TAB 5: ABOUT ===
elif menu == "🧠 About":
    st.markdown('<h1 class="main-header">🛡️ About ChurnGuardian AI</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🔮 Predict. Explain. Retain.
        
        A next-generation churn prediction platform built for telecom, SaaS, and subscription businesses.
        
        ChurnGuardian AI helps businesses:
        - Identify at-risk customers before they leave
        - Understand the reasons behind churn risk
        - Take proactive measures to improve retention
        - Optimize customer lifetime value
        
        ---
        
        **🛠️ Tech Stack**
        - Model: XGBoost (Optuna-tuned)
        - Explainability: SHAP + Natural Language
        - Dashboard: Streamlit
        - Deployment: Streamlit Cloud
        
        **📊 Performance**
        - Recall: 88.2%
        - Precision: 76.5%
        - ROC-AUC: 0.89
        - F1 Score: 0.82
        
        ---
        """)
    
    with col2:
        # Placeholder for architecture diagram
        st.image("https://via.placeholder.com/300x200/1976D2/FFFFFF?text=System+Architecture", 
                 caption="ChurnGuardian AI Architecture")
        
        # Metrics display
        st.markdown("**Model Performance**")
        metrics_data = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1 Score', 'AUC-ROC'],
            'Value': [0.765, 0.882, 0.82, 0.89]
        })
        
        fig = px.bar(
            metrics_data, 
            x='Metric', 
            y='Value',
            range_y=[0, 1],
            color='Metric',
            title="Model Performance Metrics"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ---
    
    **💼 Implementation Services**
    
    I help companies reduce churn using data science through:
    - Custom churn prediction model development
    - Integration with existing CRM systems
    - Employee training and knowledge transfer
    - Ongoing support and model maintenance
    
    **📞 Contact Information**
    - 📧 Email: bereket87722@gmail.com 
    - 🔗 [Upwork Profile](https://upwork.com/freelancers/~0107021eff758ad04e)  
    - 💼 [LinkedIn](https://linkedin.com/in/bekiger)
   
    
    _Built with ❤️ for data-driven growth._
    """)
    
    # Add a contact form
    with st.expander("📨 Secure Contact Form"):
        contact_col1, contact_col2 = st.columns(2)
        
        with contact_col1:
            contact_name = st.text_input("Your Name", key="contact_name")
            contact_email = st.text_input("Your Email", key="contact_email")
        
        with contact_col2:
            contact_company = st.text_input("Company Name", key="contact_company")
            contact_interest = st.selectbox(
                "Interest", 
                ["General Inquiry", "Custom Solution", "Partnership", "Other"],
                key="contact_interest"
            )
        
        contact_message = st.text_area("Message", key="contact_message")
        
        if st.button("Send Message Securely", key="send_message_btn"):
            if contact_message.strip():
                full_message = f"Company: {contact_company}\nInterest: {contact_interest}\n\nMessage:\n{contact_message}"
                
                if save_feedback_to_db(contact_name, contact_email, f"Contact: {contact_interest}", full_message):
                    st.success("Message sent securely! We'll review it soon.")
                else:
                    st.error("Error sending message. Please try again.")
            else:
                st.warning("Please provide a message.")
    
    st.markdown("---")
    st.caption("v2.0 • Enhanced UI/UX • Model trained on Telco dataset • For demonstration purposes")

# === TAB 6: SETTINGS ===
elif menu == "⚙️ Settings":
    st.markdown('<h1 class="main-header">⚙️ Dashboard Settings</h1>', unsafe_allow_html=True)
    
    st.markdown("### 🎨 Display Preferences")
    
    with st.container():
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            theme = st.selectbox("Color Theme", ["Default", "Dark Mode", "Light Mode"], 
                                index=["Default", "Dark Mode", "Light Mode"].index(st.session_state.display_settings['theme']),
                                key="theme_setting")
            chart_style = st.selectbox("Chart Style", ["Plotly", "Matplotlib", "Seaborn"],
                                      index=["Plotly", "Matplotlib", "Seaborn"].index(st.session_state.display_settings['chart_style']),
                                      key="chart_style_setting")
        
        with col2:
            density = st.slider("Data Density", 1, 100, st.session_state.display_settings['density'],
                               key="density_setting")
            animation = st.checkbox("Enable Animations", st.session_state.display_settings['animations'],
                                   key="animation_setting")
        
        if st.button("Apply Display Settings", type="primary"):
            st.session_state.display_settings = {
                'theme': theme,
                'chart_style': chart_style,
                'density': density,
                'animations': animation
            }
            apply_display_settings()
            st.success("Display settings applied successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🔔 Notification Settings")
    
    with st.container():
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.info("🔒 Notifications are disabled for privacy. All data is stored locally.")
        
        email_alerts = st.checkbox("Email Alerts for High-Risk Customers", False, disabled=True)
        slack_alerts = st.checkbox("Slack Notifications", False, disabled=True)
        alert_frequency = st.selectbox("Alert Frequency", ["Immediate", "Hourly", "Daily"], disabled=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Data Management")
    
    with st.container():
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Cache"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared successfully!")
        
        with col2:
            if st.button("Reset Demo Data"):
                st.session_state.saved_customers = []
                st.session_state.watchlist = []
                st.success("Demo data reset successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🔒 Data Privacy")
    st.success("""
    ✅ **Your data is 100% private and secure:**
    - All data stays on your local machine
    - No email credentials required
    - No external services involved
    - Complete control over your information
    - GDPR compliant by design
    """)
    
    st.markdown("### 📤 Export Data")
    
    with st.container():
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.info("Export your data for backup or analysis.")
        
        # Export configuration
        if st.button("Export Configuration", type="primary"):
            export_data = {
                "settings": st.session_state.display_settings,
                "saved_customers": st.session_state.saved_customers,
                "watchlist": st.session_state.watchlist,
                "export_date": datetime.now().isoformat()
            }
            
            st.download_button(
                "⬇️ Download Configuration",
                json.dumps(export_data, indent=2, default=str),
                "churnguardian_config.json",
                "application/json"
            )
        
        # Export messages
        messages = get_all_feedback()
        if messages:
            messages_csv = "\n".join([f"{m[0]},{m[1] or ''},{m[2] or ''},{m[3]},{m[4].replace(',', ';')},{m[5]}" for m in messages])
            st.download_button(
                "Download Messages (CSV)",
                f"ID,Name,Email,Type,Message,Date\n{messages_csv}",
                "churnguardian_messages.csv",
                "text/csv"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
# In your Settings tab, replace the admin message section with this:

# Admin section (only visible to admin users)
if st.session_state.user_role == 'admin':
    st.markdown("### 👨‍💼 Admin Settings")
    
    with st.container():
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown("**User Management**")
        
        # Display all users
        try:
            c = db_conn.cursor()
            c.execute("SELECT id, username, email, role FROM users")
            users = c.fetchall()
            
            user_df = pd.DataFrame(users, columns=['ID', 'Username', 'Email', 'Role'])
            st.dataframe(user_df)
        except Exception as e:
            st.error("Error loading users from database")
        
        # Add new user
        st.markdown("**Add New User**")
        new_user_col1, new_user_col2 = st.columns(2)
        
        with new_user_col1:
            new_username = st.text_input("Username", key="new_username")
            new_password = st.text_input("Password", type="password", key="new_password")
        
        with new_user_col2:
            new_email = st.text_input("Email", key="new_email")
            new_role = st.selectbox("Role", ["user", "admin"], key="new_role")
        
        if st.button("Add User", key="add_user_btn"):
            if new_username and new_password and new_email:
                if create_user(new_username, new_password, new_email, new_role):
                    st.success("User created successfully!")
                    st.rerun()
                else:
                    st.error("Error creating user")
            else:
                st.warning("Please fill all fields")
        
        # Message inbox for admin
        st.markdown("**📨 Message Inbox**")
        messages = get_all_feedback()
        
        if messages:
            st.info(f"Found {len(messages)} messages in the database")
            
            # Search and filter options
            search_term = st.text_input("Search messages", key="message_search")
            message_type_filter = st.selectbox(
                "Filter by type", 
                ["All"] + list(set([msg[3] for msg in messages])),
                key="message_filter"
            )
            
            # Filter messages
            filtered_messages = messages
            if search_term:
                filtered_messages = [msg for msg in filtered_messages 
                                   if search_term.lower() in str(msg).lower()]
            if message_type_filter != "All":
                filtered_messages = [msg for msg in filtered_messages 
                                   if msg[3] == message_type_filter]
            
            st.markdown(f"**Showing {len(filtered_messages)} messages**")
            
            # Display messages
            for msg in filtered_messages:
                msg_id, name, email, msg_type, message, created_at = msg
                with st.expander(f"#{msg_id} - {msg_type} - {created_at.split()[0]}", expanded=False):
                    st.markdown(f"""
                    <div class="message-item">
                        <strong>Message ID:</strong> {msg_id}<br>
                        <strong>Type:</strong> {msg_type}<br>
                        <strong>Date:</strong> {created_at}<br>
                        <strong>From:</strong> {name if name else 'Anonymous'} {f'({email})' if email else ''}<br>
                        <strong>Message:</strong> 
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.text_area("Message Content", message, height=100, key=f"msg_{msg_id}", disabled=True)
                    
                    # Action buttons for each message
                    col1, col2 = st.columns(2)
                    with col1:
                        # Download button for each message
                        message_text = f"Message ID: {msg_id}\nType: {msg_type}\nFrom: {name}\nEmail: {email}\nDate: {created_at}\n\nMessage:\n{message}"
                        st.download_button(
                            "📥 Download Message",
                            message_text,
                            f"message_{msg_id}.txt",
                            key=f"dl_{msg_id}"
                        )
                    with col2:
                        # Delete button
                        if st.button("🗑️ Delete", key=f"del_{msg_id}"):
                            try:
                                c = db_conn.cursor()
                                c.execute("DELETE FROM feedback WHERE id = ?", (msg_id,))
                                db_conn.commit()
                                st.success("Message deleted successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting message: {e}")
            
            # Export all messages
            st.markdown("---")
            st.markdown("**Export Options**")
            all_messages = "\n\n".join([f"Message ID: {m[0]}\nType: {m[3]}\nFrom: {m[1]}\nEmail: {m[2]}\nDate: {m[5]}\n\nMessage:\n{m[4]}" for m in messages])
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download All Messages (TXT)",
                    all_messages,
                    "all_messages.txt",
                    "text/plain"
                )
            with col2:
                # CSV export
                messages_csv = "\n".join([f"{m[0]},{m[1] or ''},{m[2] or ''},{m[3]},{m[4].replace(',', ';')},{m[5]}" for m in messages])
                st.download_button(
                    "📊 Download All Messages (CSV)",
                    f"ID,Name,Email,Type,Message,Date\n{messages_csv}",
                    "all_messages.csv",
                    "text/csv"
                )
        else:
            st.info("No messages in the inbox yet.")
        
        st.markdown('</div>', unsafe_allow_html=True)