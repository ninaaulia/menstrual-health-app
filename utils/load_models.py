
import os
import joblib
import json
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMAResults

# ── Base path (selalu absolute, tidak peduli dari mana Streamlit dijalankan) ───
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))   # → .../utils/
APP_DIR    = os.path.dirname(BASE_DIR)                    # → .../menstrual_health_app/
MODEL_DIR  = os.path.join(APP_DIR, 'models')
DATA_DIR   = os.path.join(APP_DIR, 'data')

@st.cache_resource
def load_classifier():
    return joblib.load(os.path.join(MODEL_DIR, 'best_model_logistic_regression.pkl'))

@st.cache_resource
def load_clf_scaler():
    return joblib.load(os.path.join(MODEL_DIR, 'clf_scaler.pkl'))

@st.cache_data
def load_clf_columns():
    with open(os.path.join(MODEL_DIR, 'clf_feature_columns.json')) as f:
        return json.load(f)

@st.cache_resource
def load_arima():
    try:
        model = ARIMAResults.load(os.path.join(MODEL_DIR, 'arima_model.pkl'))
    except Exception:
        from statsmodels.tsa.statespace.sarimax import SARIMAXResults
        model = SARIMAXResults.load(os.path.join(MODEL_DIR, 'arima_model.pkl'))
    with open(os.path.join(MODEL_DIR, 'arima_meta.json')) as f:
        meta = json.load(f)
    return model, meta

@st.cache_resource
def load_anomaly_pipeline():
    pipeline = joblib.load(os.path.join(MODEL_DIR, 'anomaly_pipeline.pkl'))
    with open(os.path.join(MODEL_DIR, 'anomaly_meta.json')) as f:
        meta = json.load(f)
    return pipeline, meta

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'menstrual_cycle_dataset.csv'))
