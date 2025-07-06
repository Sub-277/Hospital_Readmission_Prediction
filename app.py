import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from transformers import FeatureTransformer

st.set_page_config(
    page_title="🏥 Hospital Readmission Risk",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BBE;
        text-align: center;
        font-weight: 700;
        padding: 0.5rem;
    }
    .prediction-container {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    .success {
        background: linear-gradient(135deg, #a8e063 0%, #56ab2f 100%);
    }
    .danger {
        background: linear-gradient(135deg, #ff758c 0%, #ff7eb3 100%);
    }
    .feature-box {
        background: #f9f9f9;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #4B8BBE;
    }
    .group-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #4B8BBE;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load("hospital_readmission_pipeline_v3.0.pkl")

pipeline = load_pipeline()

# Title
st.markdown("""<h1 class='main-header'>🏥 Hospital Readmission Risk Predictor</h1>""", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar Input Groups ---
with st.sidebar:
    st.markdown("## 🩺 Patient Admission Info")

    # Group 1
    st.markdown('<div class="group-title">Patient Personal Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (in years)", min_value=0, max_value=100, value=60)
        diag_1 = st.number_input("First Diagnosis Code (1 - 999)", min_value=1, max_value=999, value=276)
        diag_3 = st.number_input("Third Diagnosis Code (1 - 999)", min_value=1, max_value=999, value=255)
    with col2:
        race = st.selectbox("Ethnicity", options=['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'])
        diag_2 = st.number_input("Second Diagnosis Code (1 - 999)", min_value=1, max_value=999, value=250)
        discharge_disposition_id = st.selectbox("Discharge Disposition ID", options=[1, 2, 3, 4, 5, 6])

    # Group 2
    st.markdown('<div class="group-title">Patient Medical Details</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        num_medications = st.number_input("Number of Medications", min_value=0, max_value=50, value=18)
    with col4:
        time_in_hospital = st.selectbox("Time in Hospital (days)", options=list(range(1, 15)), index=2)

    # Group 3
    st.markdown('<div class="group-title">Patient Diagnosis Details</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        number_diagnoses = st.number_input("Number of Diagnoses", min_value=1, max_value=20, value=6)
        num_procedures = st.number_input("Number of Procedures", min_value=0, max_value=6, value=1)
        number_diabetes_meds = st.number_input("Number of Diabetes Medications", min_value=0, max_value=20, value=2)
    with col6:
        number_preceding_year_visits = st.number_input("Number of Visits in Preceding Year", min_value=0, max_value=20, value=0)
        num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=100, value=40)

    # Group 4
    st.markdown('<div class="group-title">Other Details</div>', unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    with col7:
        insulin = st.selectbox("Insulin", options=['No', 'Steady', 'Down', 'Up'])
    with col8:
        admission_type_id = st.selectbox("Admission Type ID", options=[1, 2, 3, 4, 6, 7, 8])

# Predict button
if st.sidebar.button("🔍 Predict Readmission", type="primary"):
    input_data = pd.DataFrame([{
        'num_medications': num_medications,
        'diag_1': str(diag_1),
        'time_in_hospital': time_in_hospital,
        'diag_2': str(diag_2),
        'diag_3': str(diag_3),
        'age': age,
        'discharge_disposition_id': discharge_disposition_id,
        'number_diagnoses': number_diagnoses,
        'number_preceding_year_visits': number_preceding_year_visits,
        'num_procedures': num_procedures,
        'num_lab_procedures': num_lab_procedures,
        'number_diabetes_meds': number_diabetes_meds,
        'race': race,
        'insulin': insulin,
        'admission_type_id': admission_type_id
    }])

    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-container danger">
            <h2>🔴 Patient is likely to be readmitted</h2>
            <h3>Risk Probability: {probability:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        st.error("⚠️ Patient has moderate to high readmission risk.")
    else:
        st.markdown(f"""
        <div class="prediction-container success">
            <h2>🟢 Patient unlikely to be readmitted</h2>
            <h3>Risk Probability: {probability:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        st.success("✅ Patient has low readmission risk.")

    # Feature Importance
    with st.expander("📊 Feature Importance", expanded=False):
        model = pipeline.named_steps['classifier']
        importances = model.feature_importances_
        feature_names = [
            'num_medications', 'diag_1', 'time_in_hospital', 'diag_2', 'diag_3', 'age',
            'discharge_disposition_id', 'number_diagnoses', 'number_preceding_year_visits',
            'num_procedures', 'num_lab_procedures', 'number_diabetes_meds',
            'race', 'insulin', 'admission_type_id'
        ]

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by="Importance", ascending=True)

        bar_colors = ["green" if prediction == 0 else "red"] * len(importance_df)
        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance",
            color_discrete_sequence=bar_colors
        )
        fig.update_layout(height=400, font_size=12)
        st.plotly_chart(fig, use_container_width=True)

    # Model Information Section
    st.markdown("---")
    st.markdown("## 📊 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Model Performance")

        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Accuracy", "62.3.2%", "1.5%")
        with metrics_col2:
            st.metric("Precision", "17.0%", "1.5%")
        with metrics_col3:
            st.metric("Recall", "56.1%", "0.8%")

        st.markdown("### 📈 Model Details")
        st.info("""
        **Algorithm:** LightGBM Classifier

        **Features:** 15 medical, admission, and demographic predictors

        **Training Data:** ~100K historical hospital records with known readmission outcomes

        **Use Case:** Predict probability of patient readmission within 30 days
        """)

    with col2:
        st.markdown("### 🔍 Feature Summary")
        st.dataframe(input_data.T.rename(columns={0: "Input Values"}))