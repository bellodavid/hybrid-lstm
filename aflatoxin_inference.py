#!/usr/bin/env python3
"""
Run aflatoxin risk inference for a given location and date.

This script:
1. Takes a Nigerian state as input.
2. Maps it to a representative city (agro-ecological zone) for the model.
3. Fetches the 30-day weather history ending on the specified date.
4. Loads pre-trained scaler, encoder, and ONNX model.
5. Preprocesses the weather data.
6. Runs inference and prints the predicted aflatoxin risk.

Requirements:
    pip install requests pandas python-dateutil numpy onnxruntime scikit-learn

Example Usage (in notebook):
    predict_aflatoxin(location="Lagos")
    predict_aflatoxin(location="Kaduna", date="2025-10-01")
"""

import sys
import argparse
import json
import requests
import pandas as pd
import numpy as np
import onnxruntime as ort
from datetime import datetime, timedelta
from dateutil import parser
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- Constants ---

# Model and Preprocessor Paths
ONNX_MODEL_PATH = "aflatoxin_model.onnx"
SCALER_PATH = "standard_scaler_params.json"
ENCODER_PATH = "label_encoder_classes.json"

# Model Parameters
SEQ_LEN = 30  # Sequence length matches the 30-day weather fetch
FEATURES_LIST = [
    "Max_Temp", "Min_Temp", "Rainfall", "Relative_Humidity",
    "dayofyear_sin", "dayofyear_cos", "month_sin", "month_cos"
]

# Weather API
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_mean",
]

# TODO: User MUST complete this mapping.
# Map user-input states to the 9 representative locations your model was trained on.
# The keys should be lowercase state names for easy lookup.
# The values must match the exact strings the LabelEncoder was trained on.
STATE_TO_REPRESENTATIVE_CITY = {
    # North-West (Reps: Kaduna, Kebbi)
    'kaduna': 'Kaduna',
    'kano': 'Kaduna',
    'katsina': 'Kaduna',
    'jigawa': 'Kaduna',
    'kebbi': 'Kebbi',
    'sokoto': 'Kebbi',
    'zamfara': 'Kebbi',

    # North-East (Rep: Maiduguri)
    'borno': 'Maiduguri',
    'yobe': 'Maiduguri',
    'adamawa': 'Maiduguri',
    'gombe': 'Maiduguri',
    'bauchi': 'Maiduguri',
    # Taraba is geographically closer to Makurdi
    'taraba': 'Makurdi',

    # North-Central (Reps: Abuja, Jos, Makurdi, Niger)
    'fct': 'Abuja',
    'abuja': 'Abuja',
    'nasarawa': 'Abuja',
    'kwara': 'Abuja',
    'kogi': 'Makurdi',
    'benue': 'Makurdi',
    'plateau': 'Jos',
    'niger': 'Niger',

    # South-West (Rep: Ikeja)
    'lagos': 'Ikeja',
    'ogun': 'Ikeja',
    'oyo': 'Ikeja',
    'osun': 'Ikeja',
    'ondo': 'Ikeja',
    'ekiti': 'Ikeja',

    # South-South (Rep: Port Harcourt)
    'rivers': 'Port Harcourt',
    'akwa ibom': 'Port Harcourt',
    'bayelsa': 'Port Harcourt',
    'cross river': 'Port Harcourt',
    'delta': 'Port Harcourt',
    'edo': 'Port Harcourt',

    # South-East (Reps: Port Harcourt, Makurdi)
    'imo': 'Port Harcourt',
    'abia': 'Port Harcourt',
    'anambra': 'Port Harcourt',
    # Enugu and Ebonyi are geographically closer to Makurdi
    'enugu': 'Makurdi',
    'ebonyi': 'Makurdi',
}


# --- Weather Fetching Functions ---

def geocode(location_name):
    """Get lat/lon for a location name."""
    params = {"name": location_name, "count": 1, "language": "en", "format": "json"}
    r = requests.get(GEOCODE_URL, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    results = js.get("results")
    if not results:
        raise ValueError(f"Location not found: {location_name}")
    best = results[0]
    display_name = best.get("name", location_name)
    country = best.get("country")
    if country:
        display_name = f"{display_name}, {country}"
    return display_name, float(best["latitude"]), float(best["longitude"])


def fetch_daily_archive(lat, lon, start_date, end_date, timezone="Africa/Lagos"):
    """Fetch weather archive data."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARS),
        "timezone": timezone,
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def daily_json_to_table(js, location_label):
    """Convert Open-Meteo JSON response to a DataFrame."""
    daily = js.get("daily", {})
    times = daily.get("time", [])
    if not times:
        raise ValueError("No daily data returned for this period/location.")

    df = pd.DataFrame({
        "time": times,
        "temperature_2m_max": daily.get("temperature_2m_max", [None] * len(times)),
        "temperature_2m_min": daily.get("temperature_2m_min", [None] * len(times)),
        "precipitation_sum": daily.get("precipitation_sum", [None] * len(times)),
        "relative_humidity_2m_mean": daily.get("relative_humidity_2m_mean", [None] * len(times)),
    })

    def fmt_date(iso_str):
        dt = pd.to_datetime(iso_str).to_pydatetime()
        return f"{dt.month}/{dt.day}/{dt.year}"

    df["Date"] = df["time"].apply(fmt_date)

    out = df[["Date", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "relative_humidity_2m_mean"]].copy()
    out = out.rename(columns={
        "temperature_2m_max": "Max Temp",
        "temperature_2m_min": "Min Temp",
        "precipitation_sum": "Rainfall",
        "relative_humidity_2m_mean": "Relative Humidity",
    })

    for c in ["Max Temp", "Min Temp", "Rainfall", "Relative Humidity"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out[["Max Temp", "Min Temp", "Rainfall", "Relative Humidity"]] = out[["Max Temp", "Min Temp", "Rainfall", "Relative Humidity"]].round(8)
    out["Location"] = location_label
    out = out[["Date", "Max Temp", "Min Temp", "Rainfall", "Relative Humidity", "Location"]]
    return out


def get_30_day_weather(location_name, date_str, include_end_date=True, timezone="Africa/Lagos"):
    """Main weather fetching workflow."""
    target = parser.parse(date_str).date()
    end_date = target if include_end_date else (target - timedelta(days=1))
    start_date = end_date - timedelta(days=29)  # 30 days total

    print(f"Geocoding '{location_name}'...", file=sys.stderr)
    loc_label, lat, lon = geocode(location_name)
    print(f"-> Found {loc_label} @ {lat},{lon}", file=sys.stderr)

    print(f"Fetching weather from {start_date} to {end_date}...", file=sys.stderr)
    js = fetch_daily_archive(lat, lon, start_date.isoformat(), end_date.isoformat(), timezone=timezone)
    table = daily_json_to_table(js, loc_label)

    if len(table) != SEQ_LEN:
        print(f"Warning: API returned {len(table)} days, but model expects {SEQ_LEN}. "
              "This may be due to missing data for the requested period.", file=sys.stderr)
        if len(table) == 0:
             raise ValueError("API returned no data.")
        # Attempt to proceed, but model will likely fail at reshape step.
        # A more robust solution would pad or error out here.

    return table


# --- Preprocessing Functions ---

def load_scaler(path):
    """
    Loads a StandardScaler's parameters from a JSON file.
    Assumes JSON structure: {"mean": [..], "scale": [..]}
    """
    print(f"Loading scaler from {path}...", file=sys.stderr)
    with open(path, 'r') as f:
        data = json.load(f)

    if 'mean' not in data or 'scale' not in data:
        raise ValueError(f"Scaler JSON {path} is missing 'mean' or 'scale' keys.")

    scaler = StandardScaler()
    scaler.mean_ = np.array(data['mean'])
    scaler.scale_ = np.array(data['scale'])
    # Need to set n_features_in_ for transform to work
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler


def load_encoder(path):
    """
    Loads a LabelEncoder's parameters from a JSON file.
    Assumes JSON structure: {"classes": [..]}
    """
    print(f"Loading encoder from {path}...", file=sys.stderr)
    with open(path, 'r') as f:
        data = json.load(f)

    if 'classes' not in data:
        raise ValueError(f"Encoder JSON {path} is missing 'classes' key.")

    encoder = LabelEncoder()
    encoder.classes_ = np.array(data['classes'])
    return encoder


def preprocess_data(df, scaler, encoder):
    """
    Applies all feature engineering and preprocessing steps.
    Returns the two NumPy arrays ready for the ONNX model.
    """
    print("Applying feature engineering...", file=sys.stderr)
    df_proc = df.copy()

    # 1. Standardize column names
    df_proc.columns = df_proc.columns.str.replace(' ', '_')
    df_proc.rename(columns={
        'Max_Temp': 'Max_Temp',
        'Min_Temp': 'Min_Temp',
        'Relative_Humidity': 'Relative_Humidity'
    }, inplace=True)

    # 2. Ensure datetime for 'Date'
    df_proc["Date"] = pd.to_datetime(df_proc["Date"])

    # 3. Time features
    df_proc["dayofyear"] = df_proc["Date"].dt.dayofyear
    df_proc["month"] = df_proc["Date"].dt.month
    df_proc["dayofyear_sin"] = np.sin(2 * np.pi * df_proc["dayofyear"] / 365)
    df_proc["dayofyear_cos"] = np.cos(2 * np.pi * df_proc["dayofyear"] / 365)
    df_proc["month_sin"] = np.sin(2 * np.pi * df_proc["month"] / 12)
    df_proc["month_cos"] = np.cos(2 * np.pi * df_proc["month"] / 12)

    # 4. Encode regions
    # Extract the city name (e.g., "Kaduna" from "Kaduna, Nigeria")
    df_proc["region_name"] = df_proc["Location"].apply(lambda x: x.split(',')[0])
    df_proc["region_id"] = encoder.transform(df_proc["region_name"])

    # 5. Handle missing values
    # IMPORTANT: This uses the median of the *new* data.
    # A more robust method would use medians saved from the *training* data.
    for col in FEATURES_LIST:
        if col not in df_proc.columns:
            raise ValueError(f"Missing expected feature column: {col}")
        df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
        if df_proc[col].isnull().any():
            median_val = df_proc[col].median()
            print(f"Warning: Filling {df_proc[col].isnull().sum()} missing value(s) in '{col}' with median {median_val}", file=sys.stderr)
            df_proc[col] = df_proc[col].fillna(median_val)

    # 6. Scale features
    df_proc[FEATURES_LIST] = scaler.transform(df_proc[FEATURES_LIST])

    # 7. Extract NumPy arrays
    x_num_np = df_proc[FEATURES_LIST].values
    x_region_np = df_proc['region_id'].values

    # 8. Reshape for ONNX model (batch_size, seq_len, num_features)
    if x_num_np.shape[0] != SEQ_LEN:
         raise ValueError(f"Data shape mismatch. Expected {SEQ_LEN} days, but got {x_num_np.shape[0]}. Cannot reshape.")

    batch_size = 1
    num_features = len(FEATURES_LIST)

    x_num_reshaped = x_num_np.reshape(batch_size, SEQ_LEN, num_features)
    x_region_reshaped = x_region_reshaped = x_region_np.reshape(batch_size, SEQ_LEN)


    # 9. Ensure correct data types
    x_num_reshaped = x_num_reshaped.astype(np.float32)
    x_region_reshaped = x_region_reshaped.astype(np.int64)

    print(f"Num features shape: {x_num_reshaped.shape}", file=sys.stderr)
    print(f"Region IDs shape:   {x_region_reshaped.shape}", file=sys.stderr)

    return x_num_reshaped, x_region_reshaped


# --- Main Function for Notebook Use ---

def predict_aflatoxin(location, date=None):
    """
    Predict aflatoxin risk based on location and date.

    Args:
        location (str): The name of the state (e.g., 'Kaduna', 'Lagos').
        date (str, optional): The end date for the 30-day weather period (YYYY-MM-DD).
                              Defaults to today.
    Returns:
        float: The predicted aflatoxin risk.
    """
    if date is None:
        date = datetime.now().isoformat()

    try:
        # 1. Load Preprocessors
        scaler = load_scaler(SCALER_PATH)
        encoder = load_encoder(ENCODER_PATH)

        # 2. Map Location
        user_location = location.strip().lower()
        if user_location not in STATE_TO_REPRESENTATIVE_CITY:
            print(f"Error: Location '{location}' not found in STATE_TO_REPRESENTATIVE_CITY map.", file=sys.stderr)
            print("Please add the mapping in the script and try again.", file=sys.stderr)
            return None

        representative_city = STATE_TO_REPRESENTATIVE_CITY[user_location]
        print(f"Mapping user location '{location}' to model zone '{representative_city}'", file=sys.stderr)

        # 3. Fetch Weather
        df_weather = get_30_day_weather(representative_city, date)

        # 4. Preprocess Data
        x_num, x_region = preprocess_data(df_weather, scaler, encoder)

        # 5. Load ONNX Model
        print(f"Loading ONNX model from {ONNX_MODEL_PATH}...", file=sys.stderr)
        ort_session = ort.InferenceSession(ONNX_MODEL_PATH)

        # Verify model input names (optional but good practice)
        input_names = [inp.name for inp in ort_session.get_inputs()]
        print(f"Model expects inputs: {input_names}", file=sys.stderr)
        if "x_num" not in input_names or "x_region" not in input_names:
             print("Warning: Model input names 'x_num' or 'x_region' not found. "
                   "Using them anyway as per script.", file=sys.stderr)

        # 6. Run Inference
        inputs = {
            "x_num": x_num,
            "x_region": x_region
        }

        print("Running inference...", file=sys.stderr)
        outputs = ort_session.run(
            None,  # Get all outputs
            inputs # Pass the dictionary of inputs
        )

        aflatoxin_prediction = outputs[0]

        # 7. Print Result
        print("\n--- Aflatoxin Risk Prediction ---")
        # The output shape is likely (1, 1) or (1,). We extract the scalar value.
        prediction_value = aflatoxin_prediction.item()
        print(f"Location:           {location} (Mapped to {representative_city})")
        print(f"Weather Period:     {df_weather['Date'].iloc[0]} to {df_weather['Date'].iloc[-1]}")
        print(f"Predicted Risk:     {prediction_value:.4f}")
        print("-----------------------------------")

        return prediction_value

    except FileNotFoundError as e:
        print(f"\nError: A required file was not found.", file=sys.stderr)
        print(f"{e}", file=sys.stderr)
        print("Please ensure aflatoxin_model.onnx, standard_scaler_params.json, and label_encoder_classes.json are in the same directory.", file=sys.stderr)
        return None
    except (requests.RequestException, ValueError) as e:
        print(f"\nError during data fetching or processing:", file=sys.stderr)
        print(f"{e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\nAn unexpected error occurred:", file=sys.stderr)
        print(f"{e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None
    
    
    
predict_aflatoxin(location="Lagos", date="2024-08-01")