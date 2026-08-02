"""
About Project Page - Used Car Price Category Prediction
Pure White Theme.
"""

import streamlit as st  # type: ignore # noqa
import pandas as pd

st.set_page_config(
    page_title="About Project - Used Car Price Category Prediction",
    page_icon="ℹ️",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(16px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .main .block-container {
        animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        background-color: #FFFFFF !important;
    }
    
    .page-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #000000;
        letter-spacing: -0.03em;
        margin-bottom: 8px;
    }
    .page-subtitle {
        color: #475569;
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 28px;
    }
    .workflow-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 20px;
        padding: 20px 24px;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.04);
        margin-bottom: 16px;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .workflow-box:hover {
        transform: translateX(4px);
        border-color: #CBD5E1;
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.06);
    }
    .workflow-title {
        font-weight: 800;
        color: #000000;
        font-size: 1.05rem;
    }
    .tech-badge {
        display: inline-block;
        background: #FFFFFF;
        color: #000000;
        padding: 10px 20px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.9rem;
        margin: 6px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.25s ease;
    }
    .tech-badge:hover {
        border-color: #2563EB;
        color: #2563EB;
        transform: translateY(-2px);
    }
    .stat-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 22px;
        border: 1px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.04);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .stat-card:hover {
        transform: translateY(-4px);
        border-color: #CBD5E1;
        box-shadow: 0 12px 24px -4px rgba(0, 0, 0, 0.08);
    }
    .stat-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #000000;
        letter-spacing: -0.03em;
    }
    .stat-lbl {
        font-size: 0.82rem;
        font-weight: 700;
        color: #64748B;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="page-title">ℹ️ About Project & Dataset Information</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Detailed documentation of the end-to-end Machine Learning pipeline and specifications.</div>', unsafe_allow_html=True)

# --- Dataset Summary Stat Cards ---
st.markdown("### 📊 Dataset Overview Stats")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-val">38,000</div>
            <div class="stat-lbl">Original Listings</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-val">13,577</div>
            <div class="stat-lbl">Cleaned Records</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-val">140</div>
            <div class="stat-lbl">Features Before</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-val">65</div>
            <div class="stat-lbl">Features After</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col5:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-val">Price Category</div>
            <div class="stat-lbl">Target Variable</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- Machine Learning Workflow Timeline ---
st.markdown("### 🔄 Machine Learning Workflow Timeline")

steps = [
    ("1. Dataset Acquisition", "Collected ~38,000 multi-city raw vehicle listings from CarDekho containing text, numerical, and structural attributes."),
    ("2. Data Cleaning", "Removed metadata, redundant URL tags, duplicates, and cleaned text-based numeric fields like mileage, power, torque, and displacement."),
    ("3. Exploratory Data Analysis (EDA)", "Analyzed distributions, correlations, outliers, missingness patterns, and vehicle pricing relationships across brands."),
    ("4. Feature Engineering", "Created target vehicle price tiers (Budget, Mid-Range, Premium, Luxury) and engineered interaction terms for engine capacity and power."),
    ("5. Missing Value Handling", "Imputed missing numerical features using medians and categorical variables using domain-specific modes."),
    ("6. Encoding Categorical Variables", "Applied Label Encoding across categorical variables (Brand, Model, Transmission, Fuel Type, Body Type, State)."),
    ("7. Train-Test Split", "Partitioned dataset into 80% Training (10,861 records) and 20% Testing (2,716 records) with target stratification."),
    ("8. Model Training", "Trained multiple classification models including Logistic Regression, Decision Tree Classifier, and Random Forest Classifier."),
    ("9. Model Evaluation", "Evaluated models using Accuracy, Confusion Matrix, Precision, Recall, and 5-fold Cross-Validation."),
    ("10. Web Application & Prediction", "Built a production-ready Streamlit web application providing instant category predictions and feature insights.")
]

for title, desc in steps:
    st.markdown(
        f"""
        <div class="workflow-box">
            <div class="workflow-title">{title}</div>
            <div style="font-size:0.95rem; color:#475569; margin-top:4px; line-height:1.6;">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# --- Deep-Dive Details ---
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📌 Problem Statement & Objective")
    st.markdown(
        """
        Used car dealerships handle thousands of vehicles with diverse specifications. Manually pricing or categorizing vehicles is slow and prone to inconsistency.
        
        **Goal:** Develop an automated machine learning classification model that predicts the vehicle **Price Category** based on specifications like Brand, Model Year, Kilometers Driven, Engine CC, BHP, Torque, and Body Type.
        """
    )
    
    st.markdown("### 🛠 Data Cleaning & Feature Engineering")
    st.markdown(
        """
        - Cleaned text fields to extract clean numbers for engine displacement (CC), max power (BHP), and max torque (Nm).
        - Created target price buckets: Budget, Mid-Range, Premium, and Luxury.
        - Handled missing values using median/mode imputation.
        - Standardized numerical features with `StandardScaler` and applied `LabelEncoder` for high-cardinality categorical variables.
        """
    )

with col_right:
    st.markdown("### 🤖 Model Selection & Benchmark")
    st.markdown(
        """
        - **Random Forest Classifier**: **83.87% Accuracy** (Selected champion model)
        - **Decision Tree Classifier**: **79.79% Accuracy**
        - **Logistic Regression**: **75.63% Accuracy**
        
        Random Forest achieved superior precision by combining ensemble decision trees, capturing complex interactions without overfitting.
        """
    )
    
    st.markdown("### 💻 Technologies Used")
    tech_stack = ["Python", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Scikit-learn", "Streamlit", "Joblib"]
    badges_html = "".join([f'<span class="tech-badge">{tech}</span>' for tech in tech_stack])
    st.markdown(f'<div style="margin-top:14px;">{badges_html}</div>', unsafe_allow_html=True)
