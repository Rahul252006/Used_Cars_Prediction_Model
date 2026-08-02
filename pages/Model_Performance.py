"""
Model Performance Page - Used Car Price Category Prediction
Pure White Theme.
"""

import streamlit as st  # type: ignore # noqa
import pandas as pd
from utils.helper import (
    load_model_scaler_and_encoders,
    load_dataset,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_price_category_distribution,
)

st.set_page_config(
    page_title="Model Performance - Used Car Price Category Prediction",
    page_icon="📈",
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
    .highlight-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.04);
        margin-bottom: 22px;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .highlight-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px -4px rgba(0, 0, 0, 0.08);
        border-color: #CBD5E1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="page-title">📈 Model Performance & Evaluation</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Comparative evaluation of machine learning models trained on the CarDekho dataset.</div>', unsafe_allow_html=True)

# --- Model Performance Overview ---
rf_model, scaler, label_encoders = load_model_scaler_and_encoders()
df_cleaned = load_dataset()

col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.markdown("### 🏆 Model Comparison Benchmark")
    st.markdown(
        """
        The project evaluated three distinct classification models: Logistic Regression, Decision Tree Classifier, and Random Forest Classifier.
        The **Random Forest Classifier** achieved the highest test accuracy of **83.87%**.
        """
    )

    perf_data = {
        "Model": ["Random Forest Classifier (Selected)", "Decision Tree Classifier", "Logistic Regression"],
        "Accuracy Score": ["83.87%", "79.79%", "75.63%"],
        "Cross-Val Accuracy": ["83.45%", "78.90%", "74.80%"],
        "Status": ["🏅 Best Model", "⚡ Benchmark", "Baseline"]
    }
    df_perf = pd.DataFrame(perf_data)

    st.table(df_perf)

    st.markdown(
        """
        <div class="highlight-card">
            <h4 style="margin:0 0 8px 0; color:#000000; font-weight:800;">🌟 Champion Model: Random Forest</h4>
            <p style="margin:0; font-size:0.98rem; color:#475569; line-height:1.6;">
                Random Forest handles non-linear interactions between vehicle specifications (power, engine displacement, model year) without overfitting.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown("### 📊 Accuracy Comparison")
    fig_comp = plot_model_comparison()
    st.pyplot(fig_comp)

st.markdown("---")

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.markdown("### 🎯 Random Forest Confusion Matrix")
    st.caption("Distribution of actual vs. predicted price category classifications.")
    fig_cm = plot_confusion_matrix()
    st.pyplot(fig_cm)

with row1_col2:
    st.markdown("### 📌 Top Feature Importances")
    st.caption("Key vehicle specifications influencing price category predictions.")
    fig_fi = plot_feature_importance(rf_model, top_n=10)
    st.pyplot(fig_fi)

st.markdown("---")

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.markdown("### 🏷 Price Category Distribution")
    st.caption("Distribution of vehicles across price tiers in the processed dataset.")
    fig_dist = plot_price_category_distribution(df_cleaned)
    st.pyplot(fig_dist)

with row2_col2:
    st.markdown("### 📋 Key Performance Insights")
    st.markdown(
        """
        - **Model Year & Power**: Model Year (`myear`) and Maximum Power (`Max Power`) are the top two strongest predictors of vehicle price.
        - **Engine & Dimensions**: Engine Displacement (`engine_cc`), Vehicle Width, and Wheel Base length contribute significantly to separating Mid-Range from Premium/Luxury tiers.
        - **Robust Precision**: Random Forest achieved **83.87%** test accuracy and high F1-scores across all price tiers.
        """
    )
