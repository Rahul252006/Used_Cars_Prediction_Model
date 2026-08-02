"""
Used Car Price Category Prediction - Production Streamlit Web Application
Minimalist Monochrome Portfolio Application (Vercel / Linear / Notion / Apple Inspired)
Developer: Rahul Simhadri
Model: Random Forest Classifier (83.87% Accuracy)
"""

import os
import streamlit as st  # type: ignore # noqa
import numpy as np
import pandas as pd
from PIL import Image

from utils.helper import (
    load_model_scaler_and_encoders_v3 as load_model_scaler_and_encoders,
    load_dataset,
    get_brand_model_mapping,
    get_brand_spec_defaults,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_model_comparison,
)
from utils.preprocessing import validate_inputs, prepare_feature_vector

# --- Page Configuration ---
st.set_page_config(
    page_title="Used Car Price Category Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Modern Minimalist Black & White Design System (12px Borders, Clean Typography) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }
    
    /* Smooth Entrance Transition */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(8px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .main .block-container {
        animation: fadeIn 0.35s ease-out;
        max-width: 1140px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        background-color: #FFFFFF !important;
    }
    
    /* Hide Streamlit Header & Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Minimal Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5E7EB !important;
    }
    section[data-testid="stSidebar"] * {
        color: #111111 !important;
    }
    .sidebar-header {
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 14px;
        margin-bottom: 18px;
    }
    .sidebar-stat {
        background: #FAFAFA;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 10px 14px;
        margin-top: 8px;
    }
    .sidebar-stat-lbl {
        font-size: 0.76rem;
        color: #6B7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .sidebar-stat-val {
        font-size: 0.92rem;
        font-weight: 700;
        color: #111111;
        margin-top: 2px;
    }
    
    /* Minimal Header */
    .app-header {
        margin-bottom: 24px;
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 16px;
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #111111;
        letter-spacing: -0.03em;
        margin-bottom: 6px;
    }
    .app-subtitle {
        font-size: 1.02rem;
        color: #4B5563;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Clean Card Container */
    .card-container {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        transition: border-color 0.2s ease;
    }
    .card-container:hover {
        border-color: #9CA3AF;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #111111;
        margin-bottom: 16px;
        letter-spacing: -0.01em;
    }

    /* Result Display Card */
    .result-card {
        background: #FFFFFF;
        border: 1px solid #111111;
        border-radius: 12px;
        padding: 28px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
        animation: fadeIn 0.3s ease-out;
    }
    .result-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .result-val {
        font-size: 2rem;
        font-weight: 800;
        color: #111111;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
    }
    .result-confidence {
        font-size: 1rem;
        font-weight: 600;
        color: #374151;
    }

    /* Solid Black Action Button */
    .stButton>button, .stFormSubmitButton>button {
        background: #111111 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        border: 1px solid #111111 !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        background: #000000 !important;
        border-color: #000000 !important;
        opacity: 0.92 !important;
    }
    
    /* Clean Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 4px;
        margin-bottom: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #4B5563;
        background-color: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        color: #111111 !important;
        background-color: #F3F4F6 !important;
        font-weight: 700 !important;
    }
    
    /* Simple Footer */
    .app-footer {
        border-top: 1px solid #E5E7EB;
        padding-top: 20px;
        margin-top: 40px;
        text-align: center;
        font-size: 0.88rem;
        color: #6B7280;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Core Resources ---
try:
    rf_model, scaler, label_encoders = load_model_scaler_and_encoders()
    df_cleaned = load_dataset()
    brand_model_map = get_brand_model_mapping(df_cleaned)
except Exception as e:
    st.error(f"⚠️ Error loading core resources: {e}")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header">
            <h3 style="margin:0; font-size:1.15rem; font-weight:800; color:#111111;">🚗 Price Predictor</h3>
            <p style="margin:2px 0 0 0; font-size:0.8rem; color:#6B7280;">Machine Learning Web App</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="sidebar-stat">
            <div class="sidebar-stat-lbl">Project Name</div>
            <div class="sidebar-stat-val">Used Car Price Prediction</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-lbl">Model Used</div>
            <div class="sidebar-stat-val">Random Forest Classifier</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-lbl">Dataset Name</div>
            <div class="sidebar-stat-val">CarDekho Used Cars</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-lbl">Model Accuracy</div>
            <div class="sidebar-stat-val">83.87%</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-lbl">Developer Name</div>
            <div class="sidebar-stat-val">Rahul Simhadri</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Portfolio Application • Rahul Simhadri")

# --- App Header ---
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">Used Car Price Category Prediction</div>
        <div class="app-subtitle">
            Predict the price category of a used vehicle using a Random Forest Classification model trained on the CarDekho Used Cars Dataset.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- 3-Tab Architecture (Predict, Model Performance, About Project) ---
tab_predict, tab_performance, tab_about = st.tabs(["🎯 Predict", "📈 Model Performance", "ℹ️ About Project"])

# ==========================================
# TAB 1: PREDICT
# ==========================================
with tab_predict:
    brand_options = [
        "Maruti", "Hyundai", "Honda", "Toyota", "Mahindra", "Tata", "BMW",
        "Mercedes-Benz", "Audi", "Volkswagen", "Skoda", "Renault", "Ford",
        "Nissan", "Jeep", "Kia", "MG", "Volvo", "Jaguar", "Lexus",
        "Land Rover", "Mini", "Porsche", "Chevrolet", "Datsun", "Fiat"
    ]
    for b in brand_model_map.keys():
        if b not in brand_options:
            brand_options.append(b)

    brand_col, model_col = st.columns(2)

    with brand_col:
        selected_brand = st.selectbox("Brand (OEM)", options=brand_options, index=0)

    available_models = brand_model_map.get(selected_brand, [])
    if not available_models:
        available_models = [f"{selected_brand} Standard", f"{selected_brand} LX", f"{selected_brand} VXI", f"{selected_brand} Highline"]

    with model_col:
        selected_model = st.selectbox("Model", options=available_models, index=0)

    # Compute intelligent default engine specs for selected brand/model
    spec_defaults = get_brand_spec_defaults(selected_brand, selected_model, df_cleaned)

    with st.form(key="vehicle_prediction_form"):
        st.markdown('<div class="card-title">🚘 Vehicle Specifications</div>', unsafe_allow_html=True)

        col_yr, col_km = st.columns(2)
        with col_yr:
            selected_year = st.slider("Manufacturing Year", min_value=2000, max_value=2025, value=2018, step=1)
        with col_km:
            km_driven = st.number_input("Kilometers Driven", min_value=100, max_value=1000000, value=45000, step=1000)

        fuel_opts = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
        fuel_idx = fuel_opts.index(spec_defaults["ft"]) if spec_defaults["ft"] in fuel_opts else 0

        trans_opts = ["Manual", "Automatic"]
        trans_idx = trans_opts.index(spec_defaults["tt"]) if spec_defaults["tt"] in trans_opts else 0

        body_opts = ["Hatchback", "Sedan", "SUV", "MUV", "Coupe", "Convertible", "Pickup", "Van"]
        body_idx = body_opts.index(spec_defaults["bt"]) if spec_defaults["bt"] in body_opts else 0

        col3, col4, col5 = st.columns(3)
        with col3:
            fuel_type = st.selectbox("Fuel Type", options=fuel_opts, index=fuel_idx)
        with col4:
            transmission = st.selectbox("Transmission", options=trans_opts, index=trans_idx)
        with col5:
            body_type = st.selectbox("Body Type", options=body_opts, index=body_idx)

        col6, col7, col8 = st.columns(3)
        with col6:
            owner_type = st.selectbox("Owner Type", options=["First Owner", "Second Owner", "Third Owner", "Fourth Owner"], index=0)
        with col7:
            seating_cap = st.selectbox("Seating Capacity", options=[2, 4, 5, 6, 7, 8, 9], index=2)
        with col8:
            indian_states = [
                "maharashtra", "delhi", "karnataka", "tamil nadu", "telangana",
                "gujarat", "uttar pradesh", "west bengal", "kerala", "punjab",
                "haryana", "rajasthan", "andhra pradesh", "madhya pradesh"
            ]
            selected_state = st.selectbox("State", options=indian_states, index=0)

        col9, col10, col11 = st.columns(3)
        with col9:
            engine_cc = st.number_input("Engine Capacity (CC)", min_value=100, max_value=10000, value=int(max(100, min(10000, spec_defaults["cc"]))), step=50)
        with col10:
            max_power = st.number_input("Maximum Power (BHP)", min_value=5.0, max_value=2000.0, value=float(max(5.0, min(2000.0, spec_defaults["power"]))), step=1.0)
        with col11:
            max_torque = st.number_input("Maximum Torque (Nm)", min_value=5.0, max_value=3000.0, value=float(max(5.0, min(3000.0, spec_defaults["torque"]))), step=1.0)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_clicked = st.form_submit_button("Predict Price Category")

    # --- Prediction Execution ---
    if submit_clicked:
        user_inputs = {
            "oem": selected_brand,
            "model": selected_model,
            "myear": selected_year,
            "km_driven": km_driven,
            "ft": fuel_type,
            "tt": transmission,
            "bt": body_type,
            "owner_type": owner_type,
            "seating_capacity": seating_cap,
            "engine_cc": engine_cc,
            "max_power": max_power,
            "max_torque": max_torque,
            "state": selected_state,
        }

        is_valid, validation_errors = validate_inputs(user_inputs)

        if not is_valid:
            for err in validation_errors:
                st.error(f"⚠️ {err}")
        else:
            with st.spinner("Analyzing vehicle specifications..."):
                try:
                    X_vector = prepare_feature_vector(user_inputs, df_cleaned, rf_model, scaler, label_encoders)

                    pred_idx = rf_model.predict(X_vector)[0]
                    proba = rf_model.predict_proba(X_vector)[0]
                    confidence_pct = int(np.max(proba) * 100)

                    target_le = label_encoders.get("__target__", None)
                    if target_le and hasattr(target_le, "inverse_transform"):
                        pred_category = str(target_le.inverse_transform([pred_idx])[0])
                    else:
                        category_names = {
                            0: "Budget (Below 4 Lakhs)",
                            1: "Luxury (15+ Lakhs)",
                            2: "Mid-Range (4–8 Lakhs)",
                            3: "Premium (8–15 Lakhs)"
                        }
                        pred_category = category_names.get(pred_idx, "Mid-Range (4–8 Lakhs)")

                    st.session_state["last_prediction"] = {
                        "category": pred_category,
                        "confidence": confidence_pct,
                        "brand": selected_brand,
                        "model": selected_model,
                        "year": selected_year,
                    }
                except Exception as ex:
                    st.error(f"❌ Error during model inference: {ex}")

    # --- Display Prediction Result Card ---
    if "last_prediction" in st.session_state:
        res = st.session_state["last_prediction"]

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-lbl">Prediction</div>
                <div style="font-size:0.9rem; color:#4B5563; margin-bottom:4px;">Price Category</div>
                <div class="result-val">{res['category']}</div>
                <div class="result-confidence">Prediction Confidence: {res['confidence']}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- Model Information Section ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">📌 Model Information</div>', unsafe_allow_html=True)

    info_col1, info_col2, info_col3, info_col4, info_col5 = st.columns(5)

    with info_col1:
        st.markdown(
            """
            <div class="sidebar-stat">
                <div class="sidebar-stat-lbl">Algorithm Used</div>
                <div class="sidebar-stat-val">Random Forest Classifier</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with info_col2:
        st.markdown(
            """
            <div class="sidebar-stat">
                <div class="sidebar-stat-lbl">Dataset</div>
                <div class="sidebar-stat-val">CarDekho Used Cars</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with info_col3:
        st.markdown(
            """
            <div class="sidebar-stat">
                <div class="sidebar-stat-lbl">Training Samples</div>
                <div class="sidebar-stat-val">13,577</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with info_col4:
        st.markdown(
            """
            <div class="sidebar-stat">
                <div class="sidebar-stat-lbl">Original Dataset</div>
                <div class="sidebar-stat-val">~38,000 Listings</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with info_col5:
        st.markdown(
            """
            <div class="sidebar-stat">
                <div class="sidebar-stat-lbl">Features Used</div>
                <div class="sidebar-stat-val">65</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# TAB 2: MODEL PERFORMANCE
# ==========================================
with tab_performance:
    st.markdown('<div class="card-title">📊 Model Performance Benchmark</div>', unsafe_allow_html=True)

    perf_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy Score": ["75.63%", "79.79%", "83.87%"]
    })

    st.table(perf_df)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">📈 Model Visualizations</div>', unsafe_allow_html=True)

    vis_col1, vis_col2 = st.columns(2)

    with vis_col1:
        st.markdown("##### Model Comparison")
        fig_comp = plot_model_comparison()
        st.pyplot(fig_comp)

    with vis_col2:
        st.markdown("##### Confusion Matrix")
        fig_cm = plot_confusion_matrix()
        st.pyplot(fig_cm)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Feature Importance")
    fig_fi = plot_feature_importance(rf_model, top_n=10)
    st.pyplot(fig_fi)


# ==========================================
# TAB 3: ABOUT PROJECT
# ==========================================
with tab_about:
    st.markdown('<div class="card-title">📌 Problem Statement</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Used car dealerships handle thousands of vehicles with diverse specifications. Manually pricing or categorizing vehicles is slow and prone to inconsistency.
        
        **Objective:** Develop an automated machine learning classification model that predicts vehicle **Price Category** based on specifications like Brand, Model Year, Kilometers Driven, Engine CC, BHP, Torque, and Body Type.
        """
    )

    st.markdown("---")
    st.markdown('<div class="card-title">📁 Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown(
        """
        The dataset was sourced from CarDekho, containing ~38,000 original listings. After data cleaning and handling missing values, **13,577 records** and **65 features** were selected for training and evaluation.
        """
    )

    st.markdown("---")
    st.markdown('<div class="card-title">🔄 End-to-End Machine Learning Workflow</div>', unsafe_allow_html=True)

    workflow_steps = [
        "1. Data Cleaning",
        "2. Missing Value Handling",
        "3. Feature Engineering",
        "4. Feature Selection",
        "5. Encoding Categorical Variables",
        "6. Train-Test Split",
        "7. Model Training",
        "8. Model Evaluation"
    ]
    for step in workflow_steps:
        st.markdown(f"- **{step}**")

    st.markdown("---")
    st.markdown('<div class="card-title">💻 Technologies Used</div>', unsafe_allow_html=True)

    techs = ["Python", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Streamlit", "Joblib"]
    st.markdown(" • ".join([f"**{t}**" for t in techs]))


# --- Footer ---
st.markdown(
    """
    <div class="app-footer">
        Developed by <strong>Rahul Simhadri</strong>
    </div>
    """,
    unsafe_allow_html=True
)
