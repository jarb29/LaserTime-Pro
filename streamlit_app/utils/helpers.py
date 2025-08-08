"""
Helper Functions
================

Utility functions for the Streamlit app.
"""

import pandas as pd
import streamlit as st
from pathlib import Path

def format_time(seconds):
    """Format seconds to human readable time"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def validate_input_data(espesor, cutting_length):
    """Validate input data for predictions"""
    errors = []
    
    if espesor <= 0:
        errors.append("Material thickness must be positive")
    if espesor > 50:
        errors.append("Material thickness seems too large (>50mm)")
    
    if cutting_length <= 0:
        errors.append("Cutting length must be positive")
    if cutting_length > 1000:
        errors.append("Cutting length seems too large (>1000m)")
    
    return errors

def load_sample_data():
    """Load sample data for demonstration"""
    return pd.DataFrame({
        'Espesor': [3.0, 5.0, 8.0, 10.0, 12.0],
        'Cutting_Length': [5.0, 10.0, 15.0, 20.0, 25.0],
        'Expected_Time': [45.2, 89.5, 156.8, 234.1, 325.7]
    })

def export_results_to_csv(results_df, filename="prediction_results.csv"):
    """Export results to CSV"""
    csv = results_df.to_csv(index=False)
    return csv

def check_model_files():
    """Check if model files exist"""
    sklearn_dir = Path("models/sklearn")
    onnx_dir = Path("models/onnx")
    
    sklearn_files = {
        'model': sklearn_dir / "machining_time_model.pkl",
        'scaler': sklearn_dir / "feature_scaler.pkl"
    }
    
    onnx_files = {
        'model': onnx_dir / "machining_time_model.onnx",
        'scaler': onnx_dir / "feature_scaler.onnx"
    }
    
    sklearn_available = all(f.exists() for f in sklearn_files.values())
    onnx_available = all(f.exists() for f in onnx_files.values())
    
    return {
        'sklearn': sklearn_available,
        'onnx': onnx_available,
        'sklearn_files': sklearn_files,
        'onnx_files': onnx_files
    }

def display_model_status():
    """Display current model status"""
    status = check_model_files()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if status['sklearn']:
            st.success("✅ sklearn Models")
        else:
            st.error("❌ sklearn Models")
    
    with col2:
        if status['onnx']:
            st.success("✅ ONNX Models")
        else:
            st.warning("⚠️ ONNX Models")
    
    with col3:
        if 'model_service' in st.session_state:
            model_type = st.session_state.get('model_type', 'Unknown')
            st.info(f"🤖 Using: {model_type}")
        else:
            st.error("❌ No Service")