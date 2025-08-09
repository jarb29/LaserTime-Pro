"""
Machining Time Analysis - Streamlit App
=======================================

Main entry point for the Streamlit application.
"""

import streamlit as st
from pathlib import Path
import sys
import os

# Add project root to path for Streamlit Cloud compatibility
project_root = Path(__file__).parent
app_root = project_root
repo_root = project_root.parent

# Add all necessary paths
sys.path.insert(0, str(app_root))
sys.path.insert(0, str(repo_root))

# Ensure config can be found
if str(app_root) not in sys.path:
    sys.path.append(str(app_root))
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

# Robust imports with fallbacks
try:
    from services.onnx_model_service import ONNXModelService
except ImportError:
    ONNXModelService = None

try:
    from services.sklearn_model_service import SklearnModelService
except ImportError:
    SklearnModelService = None

try:
    from components.sidebar import create_sidebar
except ImportError:
    def create_sidebar():
        return "Dashboard"

try:
    from components.batch_processor import show_batch_prediction
except ImportError:
    def show_batch_prediction():
        st.error("Procesador por lotes no disponible")

try:
    from components.single_predictor import show_single_prediction
except ImportError:
    def show_single_prediction():
        st.error("Predictor individual no disponible")

# Utils now imported within components as needed

import pandas as pd

def main():
    # Robust config import with fallback
    try:
        from config.settings import APP_CONFIG
    except ImportError:
        # Fallback config if import fails
        APP_CONFIG = {
            "title": "LaserTime Pro",
            "icon": "⚡",
            "layout": "wide",
            "theme": "light"
        }
    
    st.set_page_config(
        page_title=APP_CONFIG["title"],
        page_icon=APP_CONFIG["icon"],
        layout=APP_CONFIG["layout"],
        initial_sidebar_state="expanded"
    )
    
    # Initialize services
    initialize_services()
    
    # Create sidebar
    selected_page = create_sidebar()
    
    # Show main content
    show_main_page()

def show_main_page():
    """Show the main application page"""
    # Clean header without redundant subtitle
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='color: #2c3e50; margin-bottom: 0.5rem;'>Análisis de Tiempo de Mecanizado</h1>
        <p style='color: #6c757d; font-size: 1.1rem; margin: 0;'>Estimación de tiempo de producción con IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs with better visual distinction
    tab1, tab2 = st.tabs(["Análisis Individual", "Procesamiento por Lotes"])
    
    # Add custom CSS for better tab styling
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0px 0px;
        color: #6c757d;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with tab1:
        show_single_prediction()
    
    with tab2:
        show_batch_prediction()

# Single prediction now handled by single_predictor component

# Batch prediction now handled by batch_processor component

@st.cache_resource
def get_model_service():
    """Get cached model service"""
    try:
        # Try ONNX first
        if ONNXModelService:
            return ONNXModelService(), "ONNX"
    except Exception as e:
        pass
    
    try:
        # Fallback to sklearn
        if SklearnModelService:
            return SklearnModelService(), "sklearn"
    except Exception as e2:
        pass
    
    raise Exception("Failed to load any models - check model files and dependencies")

def initialize_services():
    """Initialize model services with proper caching"""
    if 'model_service' not in st.session_state:
        try:
            model_service, model_type = get_model_service()
            st.session_state.model_service = model_service
            st.session_state.model_type = model_type
        except Exception as e:
            st.error(f"Error al cargar modelo: {e}")
            st.error("Por favor asegúrate de que los modelos estén disponibles en el directorio models/")

if __name__ == "__main__":
    main()