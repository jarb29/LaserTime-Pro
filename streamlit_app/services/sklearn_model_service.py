"""
Sklearn Model Service
====================

Fallback service for sklearn models when ONNX is not available.
"""

import numpy as np
import joblib
from pathlib import Path
import streamlit as st

from config.settings import SKLEARN_MODELS_DIR

class SklearnModelService:
    def __init__(self, model_dir=None):
        if model_dir is None:
            self.model_dir = SKLEARN_MODELS_DIR
        else:
            self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.metadata = None
        self._load_models()
    
    def _load_models(self):
        """Load sklearn models with Streamlit caching"""
        try:
            # Load model
            model_path = self.model_dir / "machining_time_model.pkl"
            if model_path.exists():
                self.model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = self.model_dir / "feature_scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            # Load info if available
            info_path = self.model_dir / "model_info.txt"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    self.metadata = f.read()
            
            if self.model and self.scaler:
                return True
            else:
                raise Exception("Failed to load sklearn models")
                
        except Exception as e:
            st.error(f"Error loading sklearn models: {e}")
            return False
    
    def predict_single(self, espesor, cutting_length, enable_validation=True):
        """Single prediction with sklearn and optional validation"""
        if not self.model or not self.scaler:
            raise Exception("Models not loaded")
        
        # Create feature vector with proper column names
        import pandas as pd
        feature_names = [
            'Espesor',
            'Longitude Corte (m)',
            'Cutting_Length_Squared',
            'Espesor_Squared',
            'Length_Thickness_Interaction'
        ]
        
        features_df = pd.DataFrame([[
            float(espesor),
            float(cutting_length),
            float(cutting_length) ** 2,
            float(espesor) ** 2,
            float(cutting_length) * float(espesor)
        ]], columns=feature_names)
        
        # Scale features
        scaled_features = self.scaler.transform(features_df)
        
        # Make prediction (model predicts in seconds)
        prediction_seconds = self.model.predict(scaled_features)[0]
        prediction_minutes = prediction_seconds / 60
        
        # Always apply validation
        from .validation_service import ValidationService
        validator = ValidationService()
        validation_result = validator.validate_prediction(
            espesor, cutting_length, prediction_minutes, True
        )
        
        return {
            'predicted_time_minutes': round(float(validation_result['validated_result']), 1),
            'predicted_time_seconds': round(float(validation_result['validated_result']) * 60, 1),
            'model_type': 'sklearn',
            'confidence': 'High',
            'validation': validation_result
        }
    
    def predict_batch(self, jobs_df, enable_validation=True):
        """Batch prediction with sklearn and optional validation"""
        if not self.model or not self.scaler:
            raise Exception("Models not loaded")
        
        results = []
        for _, row in jobs_df.iterrows():
            try:
                result = self.predict_single(row['Espesor'], row['Cutting_Length'], enable_validation)
                result.update({
                    'Espesor (mm)': row['Espesor'],
                    'Cutting Length (m)': row['Cutting_Length']
                })
                results.append(result)
            except Exception as e:
                st.error(f"Error predicting for row: {e}")
        
        return results
    
    def get_model_info(self):
        """Get model information"""
        return {
            'type': 'sklearn',
            'status': 'Ready' if self.model else 'Not Loaded',
            'model_class': type(self.model).__name__ if self.model else 'Unknown',
            'scaler_class': type(self.scaler).__name__ if self.scaler else 'Unknown'
        }