import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_and_save_model(file_dir):
    """Train and save machining time prediction model"""
    # Load data
    data_path = os.path.join(file_dir, 'data', 'processed', 'combined_filtered_data.csv')
    df = pd.read_csv(data_path)
    
    # Prepare features
    feature_columns = ['espesor', 'cutting_length']
    X = df[feature_columns].dropna()
    y = df.loc[X.index, 'tiempo_real_minutos']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Save model and scaler
    joblib.dump(model, 'machining_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return model, scaler, feature_columns, metrics

def predict_machining_time(espesor, cutting_length, model_path='machining_model.pkl', scaler_path='scaler.pkl'):
    """Predict machining time for given parameters"""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        raise FileNotFoundError("Model files not found. Run train_and_save_model() first.")
    
    # Prepare input with feature names
    X = pd.DataFrame([[espesor, cutting_length]], columns=['espesor', 'cutting_length'])
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    
    # Estimate range (±20%)
    range_min = prediction * 0.8
    range_max = prediction * 1.2
    
    return {
        'predicted_time_minutes': round(prediction, 2),
        'estimated_range_min': round(range_min, 2),
        'estimated_range_max': round(range_max, 2)
    }

def predict_multiple_jobs(jobs, model_path='machining_model.pkl', scaler_path='scaler.pkl'):
    """Predict machining times for multiple jobs"""
    results = []
    for espesor, cutting_length in jobs:
        result = predict_machining_time(espesor, cutting_length, model_path, scaler_path)
        results.append({
            'espesor': espesor,
            'cutting_length': cutting_length,
            **result
        })
    return results