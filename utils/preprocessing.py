"""
Preprocessing utility module for Used Car Price Category Prediction.
Handles input validation, brand-aware feature vector construction, categorical encoding using fitted LabelEncoders,
and numerical scaling.
Developer: Rahul Simhadri
"""

import pandas as pd
import numpy as np


def validate_inputs(inputs: dict) -> tuple[bool, list[str]]:
    """
    Validates user input values from the prediction form.
    Returns (is_valid, list_of_error_messages).
    """
    errors = []

    km = inputs.get("km_driven", 0)
    if km is None or km <= 0:
        errors.append("Kilometers Driven must be a positive number greater than 0.")
    elif km > 1_000_000:
        errors.append("Kilometers Driven seems unusually high (max 1,000,000 km).")

    engine_cc = inputs.get("engine_cc", 0)
    if engine_cc is None or engine_cc <= 0:
        errors.append("Engine Capacity (CC) must be greater than 0.")
    elif engine_cc > 10_000:
        errors.append("Engine Capacity (CC) seems invalid (max 10,000 CC).")

    power = inputs.get("max_power", 0)
    if power is None or power <= 0:
        errors.append("Maximum Power (BHP) must be greater than 0.")
    elif power > 2000:
        errors.append("Maximum Power (BHP) seems invalid (max 2000 BHP).")

    torque = inputs.get("max_torque", 0)
    if torque is None or torque <= 0:
        errors.append("Maximum Torque (Nm) must be greater than 0.")
    elif torque > 3000:
        errors.append("Maximum Torque (Nm) seems invalid (max 3000 Nm).")

    year = inputs.get("myear", 2020)
    if year < 2000 or year > 2025:
        errors.append("Manufacturing Year must be between 2000 and 2025.")

    return len(errors) == 0, errors


def prepare_feature_vector(inputs: dict, df: pd.DataFrame, rf_model, scaler, label_encoders: dict = None) -> pd.DataFrame:
    """
    Constructs a 65-feature input vector matching the clean RandomForest model requirements.
    Uses brand-aware baseline feature initialization, normalizes string values to dataset formats,
    applies fitted LabelEncoders, and scales numerical features.
    """
    rf_features = list(rf_model.feature_names_in_)
    scaler_features = list(scaler.feature_names_in_)

    oem = inputs.get("oem", "Maruti")
    
    # Filter dataset for brand-aware baseline feature medians
    sub_df = df[df["oem"] == oem]
    if sub_df.empty:
        sub_df = df

    # Normalize categorical string inputs to match dataset unique values
    owner_raw = str(inputs.get("owner_type", "first")).lower().replace(" owner", "").strip()
    if owner_raw not in ["first", "second", "third", "fourth", "fifth"]:
        owner_raw = "first"

    bt_raw = inputs.get("bt", "Hatchback")
    bt_map = {
        "Convertible": "Convertibles",
        "Pickup": "Pickup Trucks",
        "Van": "Minivans"
    }
    bt_raw = bt_map.get(bt_raw, bt_raw)

    input_mapping = {
        "oem": oem,
        "brand_name": oem,
        "model": inputs.get("model", "Maruti Alto"),
        "model_name": inputs.get("model", "Maruti Alto"),
        "myear": inputs.get("myear", 2018),
        "model_year": inputs.get("myear", 2018),
        "km": inputs.get("km_driven", 45000),
        "km_driven": inputs.get("km_driven", 45000),
        "ft": inputs.get("ft", "Petrol"),
        "fuel_type": inputs.get("ft", "Petrol"),
        "tt": inputs.get("tt", "Manual"),
        "transmission_type": inputs.get("tt", "Manual"),
        "bt": bt_raw,
        "body_type_new": bt_raw,
        "owner_type": owner_raw,
        "engine_cc": inputs.get("engine_cc", 1197),
        "Displacement": inputs.get("engine_cc", 1197),
        "Max Power": inputs.get("max_power", 88.0),
        "Max Torque": inputs.get("max_torque", 113.0),
        "Seating Capacity": inputs.get("seating_capacity", 5),
        "state": str(inputs.get("state", "maharashtra")).lower(),
    }

    row = {}
    for col in rf_features:
        if col in input_mapping:
            row[col] = input_mapping[col]
        elif col in sub_df.columns:
            if pd.api.types.is_numeric_dtype(sub_df[col]):
                num_s = pd.to_numeric(sub_df[col], errors="coerce")
                val = num_s.median()
                row[col] = float(val) if pd.notnull(val) else 0.0
            else:
                mode_vals = sub_df[col].dropna().mode()
                row[col] = str(mode_vals[0]) if not mode_vals.empty else "Unknown"
        else:
            row[col] = 0.0

    sample_df = pd.DataFrame([row])

    # Apply fitted LabelEncoders
    for col in sample_df.columns:
        val = str(sample_df[col].iloc[0])
        if label_encoders and col in label_encoders and col != "__target__":
            le = label_encoders[col]
            if val in le.classes_:
                sample_df[col] = le.transform([val])[0]
            else:
                # Fallback to model's first class index if unseen string
                sample_df[col] = le.transform([le.classes_[0]])[0]
        elif not isinstance(sample_df[col].iloc[0], (int, float, np.integer, np.floating)):
            sample_df[col] = 0

    sample_df = sample_df.astype(float)

    # Standardize numerical features using scaler.pkl
    sample_df[scaler_features] = scaler.transform(sample_df[scaler_features])

    return sample_df
