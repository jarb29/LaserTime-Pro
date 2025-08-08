"""
ONNX Model Conversion Script
===========================

Converts sklearn models to ONNX format for faster inference.
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

class ModelConverter:
    def __init__(self, sklearn_dir="models/sklearn", onnx_dir="models/onnx"):
        self.sklearn_dir = Path(sklearn_dir)
        self.onnx_dir = Path(onnx_dir)
        self.onnx_dir.mkdir(exist_ok=True)
        
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX libraries not available. Install with: pip install onnx onnxruntime skl2onnx")
    
    def convert_all_models(self, force=False):
        """Convert all sklearn models to ONNX"""
        try:
            # Load sklearn models
            model_path = self.sklearn_dir / "machining_time_model.pkl"
            scaler_path = self.sklearn_dir / "feature_scaler.pkl"
            
            if not model_path.exists() or not scaler_path.exists():
                print("❌ sklearn model files not found")
                return False
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            print(f"📦 Converting {type(model).__name__} model...")
            
            # Convert main model
            success_model = self._convert_model(model, "machining_time_model.onnx")
            
            # Convert scaler
            success_scaler = self._convert_scaler(scaler, "feature_scaler.onnx")
            
            if success_model and success_scaler:
                # Create metadata
                self._create_metadata(model, scaler)
                print("✅ All models converted successfully")
                return True
            else:
                print("❌ Model conversion failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during conversion: {e}")
            return False
    
    def _convert_model(self, model, filename):
        """Convert main prediction model"""
        try:
            # Define input shape (5 features)
            initial_type = [('float_input', FloatTensorType([None, 5]))]
            
            # Convert to ONNX
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            # Save ONNX model
            onnx_path = self.onnx_dir / filename
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"✅ Model converted: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Model conversion failed: {e}")
            return False
    
    def _convert_scaler(self, scaler, filename):
        """Convert feature scaler"""
        try:
            # Define input shape (5 features)
            initial_type = [('float_input', FloatTensorType([None, 5]))]
            
            # Convert to ONNX
            onnx_scaler = convert_sklearn(scaler, initial_types=initial_type)
            
            # Save ONNX scaler
            onnx_path = self.onnx_dir / filename
            with open(onnx_path, "wb") as f:
                f.write(onnx_scaler.SerializeToString())
            
            print(f"✅ Scaler converted: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Scaler conversion failed: {e}")
            return False
    
    def _create_metadata(self, model, scaler):
        """Create metadata file for converted models"""
        metadata = {
            "model_type": type(model).__name__,
            "scaler_type": type(scaler).__name__,
            "features": [
                "Espesor",
                "Longitude Corte (m)",
                "Cutting_Length_Squared",
                "Espesor_Squared",
                "Length_Thickness_Interaction"
            ],
            "conversion_info": {
                "conversion_date": datetime.now().isoformat(),
                "sklearn_version": "1.3.0",
                "onnx_version": "1.14.0"
            }
        }
        
        metadata_path = self.onnx_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Metadata created")
    
    def validate_conversion(self):
        """Validate ONNX models against sklearn models"""
        try:
            # Load original models
            model = joblib.load(self.sklearn_dir / "machining_time_model.pkl")
            scaler = joblib.load(self.sklearn_dir / "feature_scaler.pkl")
            
            # Load ONNX models
            ort_session = ort.InferenceSession(str(self.onnx_dir / "machining_time_model.onnx"))
            scaler_session = ort.InferenceSession(str(self.onnx_dir / "feature_scaler.onnx"))
            
            # Test data
            test_data = np.array([[
                5.0,    # Espesor
                10.0,   # Longitude Corte (m)
                100.0,  # Cutting_Length_Squared
                25.0,   # Espesor_Squared
                50.0    # Length_Thickness_Interaction
            ]], dtype=np.float32)
            
            # sklearn prediction
            scaled_data = scaler.transform(test_data)
            sklearn_pred = model.predict(scaled_data)
            
            # ONNX prediction
            onnx_scaled = scaler_session.run(None, {'float_input': test_data})[0]
            onnx_pred = ort_session.run(None, {'float_input': onnx_scaled})[0]
            
            # Compare results
            diff = abs(sklearn_pred[0] - onnx_pred[0][0])
            if diff < 0.01:  # Allow small numerical differences
                print(f"✅ Validation passed (diff: {diff:.6f})")
                return True
            else:
                print(f"❌ Validation failed (diff: {diff:.6f})")
                return False
                
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False