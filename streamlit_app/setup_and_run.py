#!/usr/bin/env python3
"""
Machining Time Analysis - Setup and Run Script
==============================================

This script handles:
1. Model conversion from sklearn to ONNX
2. Model validation
3. Streamlit app launch

Usage:
    python setup_and_run.py
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.settings import MODELS_DIR

# Constants
REQUIRED_MODEL_FILES = ["machining_time_model.pkl", "feature_scaler.pkl"]
FEATURE_COLUMNS = [
    'Espesor', 'Longitude Corte (m)', 'Cutting_Length_Squared',
    'Espesor_Squared', 'Length_Thickness_Interaction'
]
DATA_FILE_PATH = "../data/processed/clean_data/clean_full_tiempo_final.xlsx"
REQUIRED_DATA_COLUMNS = ['Espesor', 'Longitude Corte (m)', 'Tiempo']


def check_models_exist() -> bool:
    """Check if required sklearn model files exist.

    Returns:
        bool: True if all required files exist, False otherwise
    """
    if not MODELS_DIR.exists():
        print("❌ Models directory not found!")
        print("Please train the model first using train_model.py")
        return False

    missing_files = [
        file for file in REQUIRED_MODEL_FILES
        if not (MODELS_DIR / file).exists()
    ]

    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        return False

    print("✅ Sklearn models found")
    return True


def convert_models(force: bool = False) -> bool:
    """Convert sklearn models to ONNX format.

    Args:
        force: Whether to force conversion even if ONNX files exist

    Returns:
        bool: True if conversion successful, False otherwise
    """
    model_onnx_file = MODELS_DIR / "machining_time_model.onnx"
    scaler_onnx_file = MODELS_DIR / "feature_scaler.onnx"

    if not force and model_onnx_file.exists() and scaler_onnx_file.exists():
        print("✅ ONNX models already exist (use --force-conversion to reconvert)")
        return True

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        # Load sklearn components
        model = joblib.load(MODELS_DIR / "machining_time_model.pkl")
        scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")

        # Define input type for 5 features
        initial_type = [('float_input', FloatTensorType([None, 5]))]

        # Convert to ONNX
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        onnx_scaler = convert_sklearn(scaler, initial_types=initial_type)

        # Save ONNX models
        model_onnx_file.write_bytes(onnx_model.SerializeToString())
        scaler_onnx_file.write_bytes(onnx_scaler.SerializeToString())

        print("✅ Models converted to ONNX successfully")
        return True

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        print("Will use sklearn models as fallback")
        return False


def validate_models() -> bool:
    """Validate that models can make predictions.

    Returns:
        bool: True if validation successful, False otherwise
    """
    try:
        sys.path.append('..')
        from services.sklearn_model_service import SklearnModelService

        service = SklearnModelService()
        result = service.predict_single(3.0, 10.5)

        if result and 'predicted_time_minutes' in result:
            print("✅ Model validation passed")
            return True
        else:
            print("⚠️ Model validation failed")
            return False
    except Exception as e:
        print(f"⚠️ Validation error: {e}")
        return False


def run_streamlit_app() -> None:
    """Launch the Streamlit application."""
    try:
        print("🌟 Starting Streamlit app...")
        print("📱 App will be available at: http://localhost:8501")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")


def _prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target from raw data.

    Args:
        df: Raw dataframe with machining data

    Returns:
        Tuple of (features, target)
    """
    df = df[REQUIRED_DATA_COLUMNS].copy().dropna()
    df = df[(df['Longitude Corte (m)'] > 0) & (df['Tiempo'] > 0)]

    df['Cutting_Length_Squared'] = df['Longitude Corte (m)'] ** 2
    df['Espesor_Squared'] = df['Espesor'] ** 2
    df['Length_Thickness_Interaction'] = df['Longitude Corte (m)'] * df['Espesor']

    return df[FEATURE_COLUMNS], df['Tiempo']


def _train_and_select_best_model(X_train: np.ndarray, X_test: np.ndarray,
                                 y_train: pd.Series, y_test: pd.Series) -> Tuple[object, float]:
    """Train models and select the best one.

    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets

    Returns:
        Tuple of (best_model, best_r2_score)
    """
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'LinearRegression': LinearRegression()
    }

    best_model, best_r2 = None, 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)

        if r2 > best_r2:
            best_model, best_r2 = model, r2

    return best_model, best_r2


def main() -> None:
    """Main entry point for the setup and run script."""
    parser = argparse.ArgumentParser(
        description='Setup and run Machining Time Analysis app',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--skip-conversion',
        action='store_true',
        help='Skip ONNX conversion if models already exist'
    )
    parser.add_argument(
        '--force-conversion',
        action='store_true',
        help='Force ONNX conversion even if models exist'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate models, do not run app'
    )

    args = parser.parse_args()

    print("🚀 Machining Time Analysis - Setup and Run")
    print("=" * 50)

    # Step 1: Check sklearn models exist
    if not check_models_exist():
        print("\n📋 To get started:")
        print("1. Run train_model.py from the JKI root directory")
        print(f"2. Required files: {', '.join(REQUIRED_MODEL_FILES)} in {MODELS_DIR.relative_to(Path.cwd().parent)}/")
        return

    # Step 2: (Removed retraining step)

    # Step 3: Convert models to ONNX
    if not args.skip_conversion:
        print("\n📦 Converting models to ONNX...")
        convert_models(force=True)  # Always force conversion for consistency

    # Step 4: Validate models
    print("\n✅ Validating models...")
    validate_models()

    # Step 5: Run app (unless validate-only)
    if not args.validate_only:
        print("\n🌟 Starting Streamlit app...")
        run_streamlit_app()
    else:
        print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()