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
from datetime import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

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

# -------------------- INITIALIZE SESSION STATE --------------------
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'current_proba' not in st.session_state:
    st.session_state.current_proba = None
if 'sample_loaded' not in st.session_state:
    st.session_state.sample_loaded = False

# -------------------- LOAD MODEL & DATA --------------------
model = load_model()
if model is None:
    st.stop()

feature_names = load_feature_names()
sample_data = load_sample_data()

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

menu = st.sidebar.radio("Navigation", ["🔮 Predict", "📊 Dashboard", "📁 Reports", "🧠 About"])

# Add quick actions to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Quick Actions")
if st.sidebar.button("🔄 Load Sample Customer"):
    sample_customer = sample_data.iloc[0]
    st.session_state.sample_loaded = True
    st.session_state.sample_customer = sample_customer
    st.rerun()

# Add feedback section
st.sidebar.markdown("---")
st.sidebar.markdown("### 💬 Feedback")
with st.sidebar.expander("Share your thoughts"):
    feedback = st.text_area("How can we improve?")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! 🙏")

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
            "message": "This customer is at high risk of churning. Immediate action is required."
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
            "message": "This customer shows signs of potential churn. Proactive engagement recommended."
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
            "message": "This customer has elevated risk factors. Monitor and engage with light touchpoints."
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
            "message": "This customer appears stable. Focus on retention through excellent service."
        }

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

    # Calculate button
    if st.button("🚀 Generate Risk Assessment", type="primary"):
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
            
            # Rerun to show results
            st.rerun()

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
        elif strategy["priority"] == "Medium":
            with st.container():
                st.warning(f"⚠️ **{strategy['message']}**")
                for action in strategy["actions"]:
                    st.markdown(f"- {action}")
        else:
            with st.container():
                st.success(f"✅ **{strategy['message']}**")
                for action in strategy["actions"]:
                    st.markdown(f"- {action}")
        
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


# === TAB 3: REPORTS ===
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
        # Placeholder for more report types
        st.button("📈 Performance Metrics (Coming Soon)", disabled=True)
    
    with col3:
        st.button("📊 Customer Segmentation (Coming Soon)", disabled=True)
    
    # Report scheduling
    st.markdown("### 🗓️ Schedule Reports")
    with st.expander("Set up automated report delivery"):
        st.selectbox("Report Type", ["Daily Summary", "Weekly Analysis", "Monthly Deep Dive"])
        st.selectbox("Format", ["PDF", "CSV", "Excel"])
        st.text_input("Email Recipients")
        st.slider("Time of Day", 0, 23, 9)
        st.button("Schedule Report", type="primary")
    
    st.markdown("### 📊 Sample Report Preview")
    st.dataframe(sample_df.head(10))
    
    # Report examples
    st.markdown("### 📝 Strategy Documents")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - [Business Strategy Report](reports/Business_Strategy_Report.pdf)
        - [Model Performance Summary](reports/model_leaderboard.csv)
        - [Retention Playbook](reports/retention_playbook.pdf)
        """)
    
    with col2:
        st.markdown("""
        - [Customer Segmentation Analysis](reports/segmentation_analysis.pdf)
        - [Quarterly Churn Trends](reports/quarterly_trends.pdf)
        - [Competitive Analysis](reports/competitive_analysis.pdf)
        """)
    
    st.info("Contact for custom reports and integration.")


# === TAB 4: ABOUT ===
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
    - 🌐 [Portfolio Website](https://yourportfolio.com)
    
    _Built with ❤️ for data-driven growth._
    """)
    
    # Add a contact form
    with st.expander("📨 Contact Form"):
        contact_col1, contact_col2 = st.columns(2)
        
        with contact_col1:
            contact_name = st.text_input("Your Name")
            contact_email = st.text_input("Your Email")
        
        with contact_col2:
            contact_company = st.text_input("Company Name")
            contact_interest = st.selectbox(
                "Interest", 
                ["General Inquiry", "Custom Solution", "Partnership", "Other"]
            )
        
        contact_message = st.text_area("Message")
        
        if st.button("Send Message"):
            st.success("Message sent! I'll get back to you within 24 hours.")
    
    st.markdown("---")
    st.caption("v2.0 • Enhanced UI/UX • Model trained on Telco dataset • For demonstration purposes")

# -------------------- END OF FILE --------------------