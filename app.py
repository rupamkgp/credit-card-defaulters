import os
os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config at the very beginning
st.set_page_config(
    page_title="Credit Default Risk Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern, premium custom CSS for styling, gradients, cards, and glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B, #8A2387, #E94057, #F27121);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF4B4B;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 20px;
    }
    
    .recommendation-item {
        padding: 10px 15px;
        border-left: 4px solid #FF4B4B;
        background: rgba(255, 75, 75, 0.05);
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
    
    .recommendation-item-low {
        padding: 10px 15px;
        border-left: 4px solid #2ECC71;
        background: rgba(46, 204, 113, 0.05);
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
    
    .recommendation-item-med {
        padding: 10px 15px;
        border-left: 4px solid #F39C12;
        background: rgba(243, 156, 18, 0.05);
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load prediction artifacts
@st.cache_resource
def load_artifacts():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    from src.predict import load_prediction_artifacts
    return load_prediction_artifacts(models_dir)

try:
    model, scaler, feature_names = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please run the pipeline script first.")
    artifacts_loaded = False

# Sidebar navigation
st.sidebar.title("💳 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home & Project Overview", "Predict Default Risk", "Model Insights & SHAP"]
)

# Static Model comparison table from pipeline outputs
model_comparison_data = pd.DataFrame([
    {"Model": "Logistic Regression", "Accuracy": "92.0%", "ROC-AUC": "0.700"},
    {"Model": "Random Forest", "Accuracy": "96.0%", "ROC-AUC": "0.730"},
    {"Model": "CatBoost", "Accuracy": "96.8%", "ROC-AUC": "0.740"},
    {"Model": "LightGBM", "Accuracy": "96.9%", "ROC-AUC": "0.740"},
    {"Model": "XGBoost (Tuned & Best Model)", "Accuracy": "97.0%", "ROC-AUC": "0.750"}
])

if page == "Home & Project Overview":
    st.markdown('<div class="main-title">Credit Card Default Risk Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A machine learning solution for predicting defaults and analyzing customer financial behaviors.</div>', unsafe_allow_html=True)
    
    # Core Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">96,000+</div>
            <div class="metric-label">Credit Accounts Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">25</div>
            <div class="metric-label">Features Tracked</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">97.0%</div>
            <div class="metric-label">Best Accuracy (XGBoost)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0.750</div>
            <div class="metric-label">Max ROC-AUC (XGBoost)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Project details layout
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("Project Overview")
        st.write("""
        This project performs deep Exploratory Data Analysis (EDA) and builds predictive models to assess credit default probability. Using historical payment data of credit card accounts, we address the challenge of class imbalance using SMOTE and identify top predictors via Random Forest Feature Importances.
        
        Our tuned classifiers achieve high discriminative ability, explaining credit defaults utilizing **SHAP (SHapley Additive exPlanations)** values for complete transparency and compliance.
        """)
        
        st.subheader("Model Comparison Summary")
        st.dataframe(model_comparison_data, width="stretch")
        
    with right_col:
        st.subheader("Dataset Attributes")
        st.write("""
        The system relies on the Default of Credit Card Clients Dataset containing demographic information and payment history:
        
        *   **LIMIT_BAL**: Credit limit in NT dollars.
        *   **SEX, EDUCATION, MARRIAGE, AGE**: Client demographic features.
        *   **PAY_0 to PAY_6**: Monthly payment delay status (April to September 2005).
        *   **BILL_AMT1 to BILL_AMT6**: Monthly statement bill amounts.
        *   **PAY_AMT1 to PAY_AMT6**: Monthly amount paid in previous billing cycles.
        """)
        
        # Display sample EDA image
        img_path = "reports/images/target_distribution.png"
        if os.path.exists(img_path):
            st.image(img_path, caption="Imbalance in Target Class (Default vs No Default)", width="stretch")

elif page == "Predict Default Risk":
    st.markdown('<div class="main-title">Default Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Input credit card account indicators to evaluate defaults probabilities in real-time.</div>', unsafe_allow_html=True)
    
    if not artifacts_loaded:
        st.warning("Prediction interface disabled due to missing model artifacts. Run the pipeline first.")
    else:
        # Create input layout
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Demographic & Limit Profile")
                limit_bal = st.number_input("Credit Limit (LIMIT_BAL in NT$)", min_value=1000, max_value=1000000, value=50000, step=1000)
                age = st.slider("Client Age (AGE)", min_value=18, max_value=85, value=35)
                sex = st.radio("Gender (SEX)", options=["Male", "Female"])
                education = st.selectbox("Education Level (EDUCATION)", options=["Graduate School", "University", "High School", "Others"])
                marriage = st.selectbox("Marital Status (MARRIAGE)", options=["Married", "Single", "Others"])
                
                st.subheader("Repayment History Status")
                st.caption("Status: -2=No consumption, -1=Paid duly, 0=Revolving credit, 1-8=Payment delays")
                pay_0 = st.slider("Payment Status (Sept - PAY_0)", -2, 8, 0)
                pay_2 = st.slider("Payment Status (Aug - PAY_2)", -2, 8, 0)
                pay_3 = st.slider("Payment Status (Jul - PAY_3)", -2, 8, 0)
                pay_4 = st.slider("Payment Status (Jun - PAY_4)", -2, 8, 0)
                pay_5 = st.slider("Payment Status (May - PAY_5)", -2, 8, 0)
                pay_6 = st.slider("Payment Status (Apr - PAY_6)", -2, 8, 0)
                
            with col2:
                st.subheader("Statement Bill Amount (Sept - Apr)")
                bill_amt1 = st.number_input("Bill Amount Sept (BILL_AMT1 in NT$)", value=20000)
                bill_amt2 = st.number_input("Bill Amount Aug (BILL_AMT2 in NT$)", value=19000)
                bill_amt3 = st.number_input("Bill Amount Jul (BILL_AMT3 in NT$)", value=18000)
                bill_amt4 = st.number_input("Bill Amount Jun (BILL_AMT4 in NT$)", value=15000)
                bill_amt5 = st.number_input("Bill Amount May (BILL_AMT5 in NT$)", value=14000)
                bill_amt6 = st.number_input("Bill Amount Apr (BILL_AMT6 in NT$)", value=13000)
                
                st.subheader("Previous Payments Amount (Sept - Apr)")
                pay_amt1 = st.number_input("Paid Amount Sept (PAY_AMT1 in NT$)", value=15000)
                pay_amt2 = st.number_input("Paid Amount Aug (PAY_AMT2 in NT$)", value=1000)
                pay_amt3 = st.number_input("Paid Amount Jul (PAY_AMT3 in NT$)", value=1000)
                pay_amt4 = st.number_input("Paid Amount Jun (PAY_AMT4 in NT$)", value=1000)
                pay_amt5 = st.number_input("Paid Amount May (PAY_AMT5 in NT$)", value=1000)
                pay_amt6 = st.number_input("Paid Amount Apr (PAY_AMT6 in NT$)", value=1000)
                
            submit_btn = st.form_submit_button("Predict Default Probability", width="stretch")
            
        if submit_btn:
            # Map categorical variables
            sex_val = 1 if sex == "Male" else 2
            edu_map = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
            mar_map = {"Married": 1, "Single": 2, "Others": 3}
            
            input_dict = {
                "LIMIT_BAL": float(limit_bal),
                "SEX": sex_val,
                "EDUCATION": edu_map[education],
                "MARRIAGE": mar_map[marriage],
                "AGE": float(age),
                "PAY_0": float(pay_0),
                "PAY_2": float(pay_2),
                "PAY_3": float(pay_3),
                "PAY_4": float(pay_4),
                "PAY_5": float(pay_5),
                "PAY_6": float(pay_6),
                "BILL_AMT1": float(bill_amt1),
                "BILL_AMT2": float(bill_amt2),
                "BILL_AMT3": float(bill_amt3),
                "BILL_AMT4": float(bill_amt4),
                "BILL_AMT5": float(bill_amt5),
                "BILL_AMT6": float(bill_amt6),
                "PAY_AMT1": float(pay_amt1),
                "PAY_AMT2": float(pay_amt2),
                "PAY_AMT3": float(pay_amt3),
                "PAY_AMT4": float(pay_amt4),
                "PAY_AMT5": float(pay_amt5),
                "PAY_AMT6": float(pay_amt6)
            }
            
            # Predict
            from src.predict import predict_default
            res = predict_default(input_dict, model, scaler, feature_names)
            
            # Show output beautiful cards
            st.markdown("<br><hr>", unsafe_allow_html=True)
            
            out_col1, out_col2 = st.columns([1, 1])
            with out_col1:
                st.subheader("Model Prediction Output")
                
                # Determine colors based on risk
                prob = res["probability"]
                risk_lvl = res["risk_level"]
                
                if risk_lvl == "High Risk":
                    color = "#FF4B4B"
                    text_class = "recommendation-item"
                elif risk_lvl == "Medium Risk":
                    color = "#F39C12"
                    text_class = "recommendation-item-med"
                else:
                    color = "#2ECC71"
                    text_class = "recommendation-item-low"
                    
                st.markdown(f"""
                <div style="padding: 24px; background: rgba(255,255,255,0.03); border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);">
                    <h3 style="margin-top: 0; color: {color};">{risk_lvl}</h3>
                    <p style="font-size: 1.1rem;">Estimated Probability of Default next month:</p>
                    <div style="font-size: 3.5rem; font-weight: 700; color: {color}; margin: 10px 0;">{prob * 100:.1f}%</div>
                    <div style="background: rgba(255,255,255,0.1); border-radius: 8px; height: 12px; width: 100%;">
                        <div style="background: {color}; width: {prob * 100}%; height: 100%; border-radius: 8px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with out_col2:
                st.subheader("Actionable Recommendations")
                for rec in res["recommendations"]:
                    st.markdown(f'<div class="{text_class}">{rec}</div>', unsafe_allow_html=True)

elif page == "Model Insights & SHAP":
    st.markdown('<div class="main-title">Model Insights & Explanations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">SHAP Interpretability and overall classification curves.</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["SHAP Explanations", "Feature Importances", "Model Performance Curves"])
    
    with tab1:
        st.subheader("Model Interpretability using SHAP Values")
        st.write("""
        **SHAP (SHapley Additive exPlanations)** values measure the contribution of each feature to the model's final default risk prediction.
        Below are the generated SHAP summary and dependence plots for the XGBoost model.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            img_shap_sum = "reports/images/shap_summary.png"
            if os.path.exists(img_shap_sum):
                st.image(img_shap_sum, caption="SHAP Summary Plot", width="stretch")
            else:
                st.warning("SHAP Summary Plot not found.")
                
        with col2:
            img_shap_water = "reports/images/shap_waterfall.png"
            if os.path.exists(img_shap_water):
                st.image(img_shap_water, caption="SHAP Waterfall Plot for a sample customer", width="stretch")
            else:
                st.warning("SHAP Waterfall Plot not found.")
                
        # Dependence plot
        img_shap_dep = "reports/images/shap_dependence.png"
        if os.path.exists(img_shap_dep):
            st.image(img_shap_dep, caption="SHAP Dependence Plot: limit balance impact on default risk", width=600)
            
    with tab2:
        st.subheader("Random Forest Feature Importance")
        st.write("""
        Global feature importance metrics obtained from training a Random Forest model on the credit default dataset. 
        Engineered variables such as average delay count, utilization rates, and repayment history (`PAY_0`) rank as key predictors.
        """)
        img_fi = "reports/images/feature_importance.png"
        if os.path.exists(img_fi):
            st.image(img_fi, caption="Top 20 Features Importance", width="stretch")
        else:
            st.warning("Feature Importance plot not found.")
            
    with tab3:
        st.subheader("Performance Curves & Confusion Matrix")
        st.write("Validation charts for the tuned XGBoost classifier showing discriminative accuracy.")
        
        col1, col2 = st.columns(2)
        with col1:
            img_cm = "reports/images/confusion_matrix.png"
            if os.path.exists(img_cm):
                st.image(img_cm, caption="Confusion Matrix", width="stretch")
            else:
                st.warning("Confusion Matrix image not found.")
                
        with col2:
            img_roc = "reports/images/roc_curve.png"
            if os.path.exists(img_roc):
                st.image(img_roc, caption="ROC Curve", width="stretch")
            else:
                st.warning("ROC Curve image not found.")
