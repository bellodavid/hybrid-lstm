import json
import os
import sys
import requests
import pandas as pd
import numpy as np
import onnxruntime as ort
from datetime import datetime, timedelta
from dateutil import parser
from sklearn.preprocessing import StandardScaler, LabelEncoder
from app.models import AflatoxinResponse
from typing import Optional

class AflatoxinPredictor:
    def __init__(self):
        self.model_session = None
        self.scaler = None
        self.encoder = None
        self.model_loaded = False
        
        # Model paths
        self.model_dir = os.path.dirname(__file__)
        self.onnx_path = os.path.join(self.model_dir, "aflatoxin_model.onnx")
        self.scaler_path = os.path.join(self.model_dir, "standard_scaler_params.json")
        self.encoder_path = os.path.join(self.model_dir, "label_encoder_classes.json")
        
        # Constants
        self.SEQ_LEN = 30
        self.FEATURES_LIST = [
            "Max_Temp", "Min_Temp", "Rainfall", "Relative_Humidity",
            "dayofyear_sin", "dayofyear_cos", "month_sin", "month_cos"
        ]
        
        # State mapping
        self.STATE_TO_CITY = {
            'kaduna': 'Kaduna', 'kano': 'Kaduna', 'katsina': 'Kaduna', 'jigawa': 'Kaduna',
            'kebbi': 'Kebbi', 'sokoto': 'Kebbi', 'zamfara': 'Kebbi',
            'borno': 'Maiduguri', 'yobe': 'Maiduguri', 'adamawa': 'Maiduguri', 
            'gombe': 'Maiduguri', 'bauchi': 'Maiduguri', 'taraba': 'Makurdi',
            'fct': 'Abuja', 'abuja': 'Abuja', 'nasarawa': 'Abuja', 'kwara': 'Abuja',
            'kogi': 'Makurdi', 'benue': 'Makurdi', 'plateau': 'Jos', 'niger': 'Niger',
            'lagos': 'Ikeja', 'ogun': 'Ikeja', 'oyo': 'Ikeja', 'osun': 'Ikeja', 
            'ondo': 'Ikeja', 'ekiti': 'Ikeja',
            'rivers': 'Port Harcourt', 'akwa ibom': 'Port Harcourt', 'bayelsa': 'Port Harcourt',
            'cross river': 'Port Harcourt', 'delta': 'Port Harcourt', 'edo': 'Port Harcourt',
            'imo': 'Port Harcourt', 'abia': 'Port Harcourt', 'anambra': 'Port Harcourt',
            'enugu': 'Makurdi', 'ebonyi': 'Makurdi'
        }
    
    def load_model(self):
        """Load ONNX model and preprocessors"""
        try:
            # Load scaler
            with open(self.scaler_path, 'r') as f:
                scaler_data = json.load(f)
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(scaler_data['mean'])
            self.scaler.scale_ = np.array(scaler_data['scale'])
            self.scaler.n_features_in_ = len(self.scaler.mean_)
            
            # Load encoder
            with open(self.encoder_path, 'r') as f:
                encoder_data = json.load(f)
            self.encoder = LabelEncoder()
            self.encoder.classes_ = np.array(encoder_data['classes'])
            
            # Load ONNX model
            self.model_session = ort.InferenceSession(self.onnx_path)
            
            self.model_loaded = True
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            raise
    
    def is_model_loaded(self) -> bool:
        return self.model_loaded
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to categorical level"""
        if risk_score < 0.3:
            return "Low"
        elif risk_score < 0.7:
            return "Medium"
        else:
            return "High"
    
    def _fetch_weather_data(self, city: str, end_date: str) -> pd.DataFrame:
        """Fetch 30-day weather data"""
        # Geocoding
        geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": city, "count": 1, "language": "en", "format": "json"}
        r = requests.get(geocode_url, params=params, timeout=20)
        r.raise_for_status()
        
        results = r.json().get("results")
        if not results:
            raise ValueError(f"Location not found: {city}")
        
        lat, lon = results[0]["latitude"], results[0]["longitude"]
        
        # Weather data
        target_date = parser.parse(end_date).date()
        start_date = target_date - timedelta(days=29)
        
        archive_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": target_date.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean",
            "timezone": "Africa/Lagos"
        }
        
        r = requests.get(archive_url, params=params, timeout=30)
        r.raise_for_status()
        
        daily = r.json().get("daily", {})
        times = daily.get("time", [])
        
        if len(times) != self.SEQ_LEN:
            raise ValueError(f"Expected {self.SEQ_LEN} days of data, got {len(times)}")
        
        df = pd.DataFrame({
            "Date": [datetime.fromisoformat(t).strftime("%m/%d/%Y") for t in times],
            "Max Temp": daily.get("temperature_2m_max", []),
            "Min Temp": daily.get("temperature_2m_min", []),
            "Rainfall": daily.get("precipitation_sum", []),
            "Relative Humidity": daily.get("relative_humidity_2m_mean", []),
            "Location": city
        })
        
        return df
    
    def _preprocess_data(self, df: pd.DataFrame) -> tuple:
        """Preprocess weather data for model input"""
        df_proc = df.copy()
        df_proc.columns = df_proc.columns.str.replace(' ', '_')
        df_proc["Date"] = pd.to_datetime(df_proc["Date"])
        
        # Time features
        df_proc["dayofyear"] = df_proc["Date"].dt.dayofyear
        df_proc["month"] = df_proc["Date"].dt.month
        df_proc["dayofyear_sin"] = np.sin(2 * np.pi * df_proc["dayofyear"] / 365)
        df_proc["dayofyear_cos"] = np.cos(2 * np.pi * df_proc["dayofyear"] / 365)
        df_proc["month_sin"] = np.sin(2 * np.pi * df_proc["month"] / 12)
        df_proc["month_cos"] = np.cos(2 * np.pi * df_proc["month"] / 12)
        
        # Region encoding
        region_name = df_proc["Location"].iloc[0].split(',')[0]
        df_proc["region_id"] = self.encoder.transform([region_name] * len(df_proc))
        
        # Handle missing values and scale
        for col in self.FEATURES_LIST:
            df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce').fillna(df_proc[col].median())
        
        df_proc[self.FEATURES_LIST] = self.scaler.transform(df_proc[self.FEATURES_LIST])
        
        # Prepare model inputs
        x_num_np = df_proc[self.FEATURES_LIST].values
        x_region_np = df_proc['region_id'].values
        
        # Reshape for ONNX model
        x_num = x_num_np.reshape(1, self.SEQ_LEN, len(self.FEATURES_LIST)).astype(np.float32)
        x_region = x_region_np.reshape(1, self.SEQ_LEN).astype(np.int64)
        
        return x_num, x_region
    
    def predict(self, location: str, date: Optional[str] = None) -> Optional[AflatoxinResponse]:
        """Make aflatoxin risk prediction"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        if date is None:
            date = datetime.now().isoformat()
        
        # Map location to representative city
        user_location = location.strip().lower()
        if user_location not in self.STATE_TO_CITY:
            raise ValueError(f"Location '{location}' not supported")
        
        representative_city = self.STATE_TO_CITY[user_location]
        
        # Fetch and preprocess data
        df_weather = self._fetch_weather_data(representative_city, date)
        x_num, x_region = self._preprocess_data(df_weather)
        
        # Run inference
        inputs = {"x_num": x_num, "x_region": x_region}
        outputs = self.model_session.run(None, inputs)
        prediction_value = float(outputs[0].item())
        
        # Format response
        weather_period = f"{df_weather['Date'].iloc[0]} to {df_weather['Date'].iloc[-1]}"
        
        return AflatoxinResponse(
            location=location,
            representative_city=representative_city,
            weather_period=weather_period,
            predicted_risk=round(prediction_value, 4),
            risk_level=self._get_risk_level(prediction_value)
        )