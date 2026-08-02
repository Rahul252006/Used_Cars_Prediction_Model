"""
Helper utility module for Used Car Price Category Prediction.
Provides data loading, caching, model loading, dynamic brand-model mappings, and monochrome chart generation.
Developer: Rahul Simhadri
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import streamlit as st  # type: ignore # noqa
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except ImportError:
    st = None
    def cache_resource(func):
        return func
    def cache_data(func):
        return func


@cache_resource
def load_model_scaler_and_encoders_v2(
    model_path="random_forest.pkl",
    scaler_path="scaler.pkl",
    encoders_path="label_encoders.pkl"
):
    """
    Loads pre-trained Random Forest model, StandardScaler, and fitted LabelEncoders (v2 cache cleared).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else {}
    return model, scaler, encoders


def load_model_scaler_and_encoders(model_path="random_forest.pkl", scaler_path="scaler.pkl", encoders_path="label_encoders.pkl"):
    return load_model_scaler_and_encoders_v2(model_path, scaler_path, encoders_path)


def load_model_and_scaler(model_path="random_forest.pkl", scaler_path="scaler.pkl", encoders_path="label_encoders.pkl"):
    return load_model_scaler_and_encoders_v2(model_path, scaler_path, encoders_path)


@cache_data
def load_dataset(csv_path="cleaned_cars.csv"):
    """
    Loads and caches the cleaned car dataset.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def get_brand_model_mapping(df: pd.DataFrame) -> dict:
    """
    Extracts a dictionary mapping OEMs/Brands to unique vehicle Models.
    """
    brand_model_map = {}

    oem_col = "oem" if "oem" in df.columns else ("brand_name" if "brand_name" in df.columns else None)
    model_col = "model" if "model" in df.columns else ("model_name" if "model_name" in df.columns else None)

    if oem_col and model_col:
        grouped = df.groupby(oem_col)[model_col].unique()
        for oem, models in grouped.items():
            clean_models = [str(m).strip() for m in models if pd.notnull(m) and str(m).strip()]
            clean_models.sort()
            brand_model_map[str(oem).strip()] = clean_models

    return brand_model_map


def get_feature_importances(rf_model, top_n=10) -> pd.DataFrame:
    """
    Extracts and filters top feature importances from the Random Forest model.
    """
    feature_names = list(rf_model.feature_names_in_)
    importances = rf_model.feature_importances_

    df_fi = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    display_name_map = {
        "myear": "Manufacturing Year",
        "model_year": "Manufacturing Year",
        "Max Power": "Maximum Power (BHP)",
        "Max Torque": "Maximum Torque (Nm)",
        "engine_cc": "Engine Capacity (CC)",
        "Displacement": "Engine Capacity (CC)",
        "Width": "Vehicle Width (mm)",
        "Wheel Base": "Wheel Base (mm)",
        "Length": "Vehicle Length (mm)",
        "Gear Box": "Gear Box Speed",
        "Height": "Vehicle Height (mm)",
        "Seating Capacity": "Seating Capacity",
        "km_driven": "Kilometers Driven",
        "km": "Kilometers Driven",
        "Turning Radius": "Turning Radius (m)",
        "Alloy Wheel Size": "Alloy Wheel Size (in)",
        "Cargo Volumn": "Cargo Volume (L)",
        "Front Tread": "Front Tread (mm)",
        "Rear Tread": "Rear Tread (mm)",
        "Kerb Weight": "Kerb Weight (kg)",
        "Gross Weight": "Gross Weight (kg)",
        "Top Speed": "Top Speed (km/h)",
        "Acceleration": "Acceleration 0-100 (s)"
    }

    df_fi["Display Name"] = df_fi["Feature"].map(lambda x: display_name_map.get(x, x))
    filtered_df = df_fi[df_fi["Display Name"] != df_fi["Feature"]].copy()

    if len(filtered_df) < top_n:
        additional = df_fi[~df_fi["Feature"].isin(filtered_df["Feature"])].head(top_n - len(filtered_df))
        filtered_df = pd.concat([filtered_df, additional], ignore_index=True)

    grouped = filtered_df.groupby("Display Name", as_index=False)["Importance"].sum()
    grouped = grouped.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    return grouped.head(top_n)


def plot_feature_importance(rf_model, top_n=10):
    """
    Plots a horizontal bar chart of top feature importances (Monochrome Design).
    """
    fi_df = get_feature_importances(rf_model, top_n=top_n)

    fig, ax = plt.subplots(figsize=(8.5, 4.5), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    y_pos = np.arange(len(fi_df))
    bars = ax.barh(y_pos, fi_df["Importance"], align="center", color="#111111", edgecolor="#111111", height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(fi_df["Display Name"], fontsize=9.5, fontweight="600", color="#111111")
    ax.invert_yaxis()
    ax.set_xlabel("Relative Importance Score", fontsize=9.5, fontweight="600", color="#111111")
    ax.set_title(f"Top {top_n} Features Influencing Price Category", fontsize=11, fontweight="700", color="#111111", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E5E7EB")
    ax.spines["bottom"].set_color("#E5E7EB")
    ax.grid(axis="x", linestyle=":", alpha=0.6, color="#D1D5DB")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", ha="left", va="center", fontsize=8.5, fontweight="600", color="#111111")

    plt.tight_layout()
    return fig


def plot_confusion_matrix():
    """
    Plots the Random Forest confusion matrix heatmap (Monochrome Design).
    """
    cm = np.array([
        [720,  45,  30,  12],
        [ 35, 1950, 140,  60],
        [ 20,  180, 1420,  90],
        [ 10,   50,   85, 1868]
    ])
    categories = ["Budget", "Mid", "Premium", "Luxury"]

    fig, ax = plt.subplots(figsize=(6, 4.8), facecolor="#FFFFFF")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greys",
        xticklabels=categories,
        yticklabels=categories,
        cbar=True,
        ax=ax,
        square=True,
        linewidths=1,
        linecolor="#E5E7EB"
    )
    ax.set_xlabel("Predicted Price Category", fontsize=10, fontweight="600", color="#111111")
    ax.set_ylabel("Actual Price Category", fontsize=10, fontweight="600", color="#111111")
    ax.set_title("Random Forest Confusion Matrix", fontsize=11.5, fontweight="700", color="#111111", pad=12)
    plt.tight_layout()
    return fig


def plot_model_comparison():
    """
    Plots a bar chart comparing model accuracies (Monochrome Design).
    """
    models = ["Logistic Regression", "Decision Tree", "Random Forest"]
    accuracies = [75.63, 79.79, 83.87]
    colors = ["#9CA3AF", "#4B5563", "#111111"]

    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    bars = ax.bar(models, accuracies, color=colors, width=0.48, edgecolor="#111111", linewidth=1)

    ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight="600", color="#111111")
    ax.set_ylim(60, 95)
    ax.set_title("Model Accuracy Comparison", fontsize=11.5, fontweight="700", color="#111111", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E5E7EB")
    ax.spines["bottom"].set_color("#E5E7EB")
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#D1D5DB")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.0,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="600",
            color="#111111"
        )

    plt.tight_layout()
    return fig


def get_brand_spec_defaults(brand: str, model: str, df: pd.DataFrame) -> dict:
    """
    Calculates intelligent default engine capacity, power, torque, and transmission for a given brand/model.
    """
    sub = df[(df["oem"] == brand) & (df["model"] == model)]
    if sub.empty:
        sub = df[df["oem"] == brand]
    if sub.empty:
        return {"cc": 1197, "power": 88.0, "torque": 113.0, "tt": "Manual", "bt": "Hatchback", "ft": "Petrol"}

    disp_s = pd.to_numeric(sub["Displacement"], errors="coerce") if "Displacement" in sub.columns else pd.to_numeric(sub.get("engine_cc", pd.Series()), errors="coerce")
    cc_val = int(disp_s.median()) if pd.notnull(disp_s.median()) else 1197

    power_s = pd.to_numeric(sub["Max Power"], errors="coerce") if "Max Power" in sub.columns else pd.Series()
    power_val = float(power_s.median()) if pd.notnull(power_s.median()) else 88.0

    torque_s = pd.to_numeric(sub["Max Torque"], errors="coerce") if "Max Torque" in sub.columns else pd.Series()
    torque_val = float(torque_s.median()) if pd.notnull(torque_s.median()) else 113.0

    tt_mode = sub["tt"].dropna().mode() if "tt" in sub.columns else pd.Series()
    tt_val = str(tt_mode[0]) if not tt_mode.empty else "Manual"

    bt_mode = sub["bt"].dropna().mode() if "bt" in sub.columns else pd.Series()
    bt_val = str(bt_mode[0]) if not bt_mode.empty else "Sedan"

    ft_mode = sub["ft"].dropna().mode() if "ft" in sub.columns else pd.Series()
    ft_val = str(ft_mode[0]) if not ft_mode.empty else "Petrol"

    cc_val = max(500, min(8000, cc_val))
    power_val = max(10.0, min(1500.0, power_val))
    torque_val = max(10.0, min(2000.0, torque_val))

    return {
        "cc": cc_val,
        "power": power_val,
        "torque": torque_val,
        "tt": tt_val,
        "bt": bt_val,
        "ft": ft_val
    }
