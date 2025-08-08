"""
Model Validation Script
=======================

Validates ONNX models against sklearn models.
"""

import numpy as np
import joblib
from pathlib import Path

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

def validate_onnx_models():
    """Validate ONNX models against sklearn models"""
    if not ONNX_AVAILABLE:
        print("⚠️ ONNX not available, skipping validation")
        return False
    
    sklearn_dir = Path("models/sklearn")
    onnx_dir = Path("models/onnx")
    
    try:
        # Check if files exist
        sklearn_model_path = sklearn_dir / "machining_time_model.pkl"
        sklearn_scaler_path = sklearn_dir / "feature_scaler.pkl"
        onnx_model_path = onnx_dir / "machining_time_model.onnx"
        onnx_scaler_path = onnx_dir / "feature_scaler.onnx"
        
        if not all(p.exists() for p in [sklearn_model_path, sklearn_scaler_path, onnx_model_path, onnx_scaler_path]):
            print("⚠️ Some model files missing, skipping validation")
            return False
        
        # Load models
        sklearn_model = joblib.load(sklearn_model_path)
        sklearn_scaler = joblib.load(sklearn_scaler_path)
        
        onnx_model_session = ort.InferenceSession(str(onnx_model_path))
        onnx_scaler_session = ort.InferenceSession(str(onnx_scaler_path))
        
        # Test with multiple data points
        test_cases = [
            [3.0, 5.0, 25.0, 9.0, 15.0],
            [5.0, 10.0, 100.0, 25.0, 50.0],
            [8.0, 15.0, 225.0, 64.0, 120.0],
            [10.0, 20.0, 400.0, 100.0, 200.0]
        ]
        
        all_passed = True
        for i, test_case in enumerate(test_cases):
            test_data = np.array([test_case], dtype=np.float32)
            
            # sklearn prediction
            scaled_data = sklearn_scaler.transform(test_data)
            sklearn_pred = sklearn_model.predict(scaled_data)[0]
            
            # ONNX prediction
            onnx_scaled = onnx_scaler_session.run(None, {'float_input': test_data})[0]
            onnx_pred = onnx_model_session.run(None, {'float_input': onnx_scaled})[0][0]
            
            # Compare
            diff = abs(sklearn_pred - onnx_pred)
            if diff < 0.1:  # Allow small numerical differences
                print(f"✅ Test case {i+1}: PASSED (diff: {diff:.6f})")
            else:
                print(f"❌ Test case {i+1}: FAILED (diff: {diff:.6f})")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

if __name__ == "__main__":
    validate_onnx_models()