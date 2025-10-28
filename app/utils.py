import json
import numpy as np
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_scaler(path):
    with open(path, 'r') as f:
        data = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(data['mean'])
    scaler.scale_ = np.array(data['scale'])
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler

def load_encoder(path):
    with open(path, 'r') as f:
        data = json.load(f)
    encoder = LabelEncoder()
    encoder.classes_ = np.array(data['classes'])
    return encoder

def load_model(path):
    return ort.InferenceSession(path)