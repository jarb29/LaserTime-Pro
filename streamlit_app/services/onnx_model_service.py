"""
ONNX Model Service
==================

Handles ONNX model loading and inference.
"""

import json
import numpy as np
from pathlib import Path
import streamlit as st

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

class ONNXModelService:
    def __init__(self, model_dir=None):
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX runtime not available")
        
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            try:
                from config.settings import ONNX_MODELS_DIR
                self.model_dir = ONNX_MODELS_DIR
            except ImportError:
                # Fallback to relative path
                self.model_dir = Path(__file__).parent.parent.parent / "notebooks" / "analysis" / "saved_models"
        self.session = None
        self.scaler_session = None
        self.metadata = None
        self._load_models()
    
    def _load_models(self):
        """Load ONNX models with Streamlit caching"""
        try:
            try:
                from config.settings import ONNX_CONFIG
            except ImportError:
                # Fallback ONNX config
                ONNX_CONFIG = {"providers": ["CPUExecutionProvider"]}
            
            # Load model
            model_path = self.model_dir / "machining_time_model.onnx"
            if model_path.exists():
                self.session = ort.InferenceSession(
                    str(model_path),
                    providers=ONNX_CONFIG["providers"]
                )
            
            # Load scaler
            scaler_path = self.model_dir / "feature_scaler.onnx"
            if scaler_path.exists():
                self.scaler_session = ort.InferenceSession(
                    str(scaler_path),
                    providers=ONNX_CONFIG["providers"]
                )
            
            # Load metadata
            metadata_path = self.model_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            if self.session and self.scaler_session:
                return True
            else:
                raise Exception("Failed to load ONNX models")
                
        except Exception as e:
            st.error(f"Error loading ONNX models: {e}")
            return False
    
    def predict_single(self, espesor, cutting_length, enable_validation=True):
        """Single prediction with ONNX and optional validation"""
        if not self.session or not self.scaler_session:
            raise Exception("Models not loaded")
        
        # Create feature vector
        features = np.array([[
            float(espesor),
            float(cutting_length),
            float(cutting_length) ** 2,
            float(espesor) ** 2,
            float(cutting_length) * float(espesor)
        ]], dtype=np.float32)
        
        # Scale features
        scaled_features = self.scaler_session.run(None, {'float_input': features})[0]
        
        # Make prediction (model predicts in seconds)
        prediction_seconds = self.session.run(None, {'float_input': scaled_features})[0][0][0]
        prediction_minutes = float(prediction_seconds) / 60
        
        # Always apply validation
        from .validation_service import ValidationService
        validator = ValidationService()
        validation_result = validator.validate_prediction(
            espesor, cutting_length, prediction_minutes, True
        )
        
        return {
            'predicted_time_minutes': round(float(validation_result['validated_result']), 1),
            'predicted_time_seconds': round(float(validation_result['validated_result']) * 60, 1),
            'model_type': 'ONNX',
            'confidence': 'High',
            'validation': validation_result
        }
    
    def predict_batch(self, jobs_df, enable_validation=True):
        """Batch prediction with ONNX and optional validation"""
        if not self.session or not self.scaler_session:
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
            'type': 'ONNX',
            'status': 'Ready' if self.session else 'Not Loaded',
            'metadata': self.metadata,
            'features': self.metadata.get('features', []) if self.metadata else []
        }