# dashboard/app.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import os

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="ChurnGuardian AI",
    page_icon="🛡️",
    layout="wide"
)

# -------------------- LOAD ASSETS --------------------
@st.cache_resource
def load_model():
    return joblib.load('models/churn_xgboost_model.pkl')

@st.cache_resource
def load_encoder():
    return joblib.load('models/onehot_encoder.pkl')

@st.cache_data
def load_feature_names():
    return joblib.load('models/feature_columns.pkl')

model = load_model()
encoder = load_encoder()
feature_names = load_feature_names()

# -------------------- HEADER --------------------
st.title("🛡️ ChurnGuardian AI")
st.markdown("### Predict churn risk & get AI-powered retention strategies")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Insights", "📄 Report"])

# -------------------- TAB 1: PREDICT --------------------
with tab1:
    st.header("Predict Churn for a Customer")

    col1, col2 = st.columns(2)

    with col1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])

    with col2:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 80.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 960.0)
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner", ["Yes", "No"])

    if st.button("🔮 Predict Churn Risk"):
        # Create DataFrame
        data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'SeniorCitizen': [senior],
            'Partner': [partner],
            'Contract': [contract],
            'InternetService': [internet],
            'OnlineSecurity': [online_security],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv]
        })

        # Feature engineering
        data['Monthly_to_Total_Ratio'] = data['MonthlyCharges'] / (data['TotalCharges'] + 1)
        data['HasPremiumServices'] = ((data['InternetService'] != 'No') & 
                                      (data['StreamingTV'] == 'Yes') & 
                                      (data['OnlineSecurity'] == 'Yes')).astype(int)
        data['IsHighMonthly'] = (data['MonthlyCharges'] > 80).astype(int)

        # One-hot encode
        cat_cols = ['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport', 'StreamingTV', 'Partner']
        data_encoded = pd.get_dummies(data, columns=cat_cols)
        
        # Align with training columns
        for col in feature_names:
            if col not in data_encoded.columns:
                data_encoded[col] = 0
        data_encoded = data_encoded[feature_names]

        # Predict
        proba = model.predict_proba(data_encoded)[0][1]
        prediction = "🚨 High Risk" if proba > 0.5 else "✅ Low Risk"

        # Display
        st.metric("Churn Probability", f"{proba:.1%}")
        st.success(f"Prediction: **{prediction}**")

        # SHAP Explanation
        st.subheader("🧠 Why This Prediction? (Model Explanation)")
       # ✅ CORRECTED: Modern SHAP-compatible code
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(data_encoded)

        # Now shap_values is an Explanation object — safe to plot
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.bar(shap_values[0], ax=ax)
        plt.title("SHAP: Feature Impact on Prediction")
        st.pyplot(fig)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, ax=ax2)
        plt.title("How Features Push Prediction")
        st.pyplot(fig2)
        plt.close(fig2)

        # AI-Powered Insight (Simulated)
        st.subheader("💡 Retention Strategy (AI Suggestion)")
        if proba > 0.7:
            st.warning("""
            **Recommended Action:**  
            Offer a 3-month free loyalty plan with tech support.  
            This customer is on month-to-month and pays >$80 — high flight risk.
            """)
        elif proba > 0.5:
            st.info("Consider bundling online backup at 50% off for 3 months.")
        else:
            st.success("No action needed. Customer is stable.")

# -------------------- TAB 2: INSIGHTS --------------------
with tab2:
    st.header("Business Insights")
    st.image("dashboard/assets/shap_summary.png", caption="Top Churn Drivers")
    st.image("dashboard/assets/roc_curve.png", caption="Model Performance")
    st.image("dashboard/assets/dependence_monthly_contract.png", caption="Monthly Charges × Contract Interaction")

# -------------------- TAB 3: REPORT --------------------
with tab3:
    st.header("Download Business Report")
    st.markdown("""
    📄 [Download Business Strategy Report (PDF)](reports/Business_Strategy_Report.pdf)  
    📈 Includes:  
    - Top 3 churn drivers  
    - $1.2M/year savings estimate  
    - 3 retention strategies  
    - Contact for consulting
    """)