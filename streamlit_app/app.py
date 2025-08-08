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

from services.onnx_model_service import ONNXModelService
from services.sklearn_model_service import SklearnModelService
from components.sidebar import create_sidebar
from components.forms import create_single_prediction_form, create_batch_prediction_form
from components.charts import create_prediction_gauge
from utils.helpers import format_time, validate_input_data, export_results_to_csv
import pandas as pd
from config.settings import APP_CONFIG

def main():
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
        <h1 style='color: #2c3e50; margin-bottom: 0.5rem;'>Machining Time Analysis</h1>
        <p style='color: #6c757d; font-size: 1.1rem; margin: 0;'>AI-powered production time estimation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs with better visual distinction
    tab1, tab2 = st.tabs(["Individual Analysis", "Batch Processing"])
    
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

def show_single_prediction():
    """Show single prediction interface"""
    # Better balanced columns - input slightly smaller
    col1, col2 = st.columns([0.4, 0.6])
    
    with col1:
        st.markdown("<h4 style='color: #2c3e50; margin-bottom: 1.5rem;'>Job Parameters</h4>", unsafe_allow_html=True)
        
        # Input form with consistent formatting
        espesor = st.number_input(
            "Material Thickness (mm)",
            min_value=0.1,
            max_value=50.0,
            value=5.0,
            step=0.1,
            format="%.1f",
            help="Thickness of the material in millimeters"
        )
        
        cutting_length = st.number_input(
            "Cutting Length (m)",
            min_value=0.1,
            max_value=1000.0,
            value=10.0,
            step=0.1,
            format="%.1f",
            help="Total cutting length in meters"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Professional calculate button
        predict_button = st.button(
            "Calculate Estimate", 
            type="primary", 
            use_container_width=True,
            help="Generate machining time estimate"
        )
    
    with col2:
        st.markdown("<h3 style='text-align: center; color: #2c3e50; margin-bottom: 1.5rem;'>Estimation Results</h3>", unsafe_allow_html=True)
        
        if predict_button:
            # Validate inputs
            errors = validate_input_data(espesor, cutting_length)
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Make prediction
                if 'model_service' in st.session_state:
                    try:
                        # Validation always enabled
                        enable_validation = True
                        
                        with st.spinner("Calculating estimate..."):
                            result = st.session_state.model_service.predict_single(
                                espesor, cutting_length, enable_validation
                            )
                        
                        # Main result - user-friendly
                        minutes = int(result['predicted_time_minutes'])
                        seconds = int((result['predicted_time_minutes'] - minutes) * 60)
                        
                        st.markdown(f"""
                        <div style='text-align: center; margin-bottom: 1.5rem;'>
                            <div style='
                                background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
                                color: #2d5a2d;
                                padding: 1rem;
                                border-radius: 12px;
                                border: 1px solid #c3e6c3;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            '>
                                <div style='font-size: 1.2rem; font-weight: 600;'>Estimated Machining Time</div>
                                <div style='font-size: 2rem; font-weight: 700; margin: 0.5rem 0;'>{minutes} min {seconds} sec</div>
                                <div style='font-size: 0.9rem; opacity: 0.8;'>High-performance AI prediction</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Two columns: metrics and chart
                        col2a, col2b = st.columns([1, 1])
                        
                        with col2a:
                            # Consistent metric cards with professional styling
                            total_seconds = int(result['predicted_time_seconds'])
                            material_thickness = int(espesor) if espesor == int(espesor) else espesor
                            
                            st.markdown(f"""
                            <div style="text-align: center;">
                                <div style="
                                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                                    border-radius: 12px;
                                    padding: 1rem;
                                    margin-bottom: 0.8rem;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                    border: 1px solid #dee2e6;
                                ">
                                    <div style="font-size: 0.875rem; color: #6c757d; font-weight: 500;">Total Seconds</div>
                                    <div style="font-size: 1.6rem; font-weight: 600; color: #495057;">{total_seconds:,}</div>
                                </div>
                                <div style="
                                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                                    border-radius: 12px;
                                    padding: 1rem;
                                    margin-bottom: 0.8rem;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                    border: 1px solid #dee2e6;
                                ">
                                    <div style="font-size: 0.875rem; color: #6c757d; font-weight: 500;">Material Thickness</div>
                                    <div style="font-size: 1.6rem; font-weight: 600; color: #495057;">{material_thickness}mm</div>
                                </div>
                                <div style="
                                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                                    border-radius: 12px;
                                    padding: 1rem;
                                    margin-bottom: 0.8rem;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                    border: 1px solid #dee2e6;
                                ">
                                    <div style="font-size: 0.875rem; color: #6c757d; font-weight: 500;">Cutting Length</div>
                                    <div style="font-size: 1.6rem; font-weight: 600; color: #495057;">{cutting_length}m</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2b:
                            # Simple progress indicator instead of complex gauge
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                                border-radius: 12px;
                                padding: 1.5rem;
                                text-align: center;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                border: 1px solid #dee2e6;
                            ">
                                <div style="font-size: 0.875rem; color: #6c757d; margin-bottom: 1rem;">Processing Time Breakdown</div>
                                <div style="font-size: 2.5rem; font-weight: 700; color: #28a745; margin-bottom: 0.5rem;">{result['predicted_time_minutes']:.1f}</div>
                                <div style="font-size: 1rem; color: #6c757d; margin-bottom: 1rem;">minutes</div>
                                <div style="
                                    background: #e9ecef;
                                    height: 8px;
                                    border-radius: 4px;
                                    overflow: hidden;
                                    margin-bottom: 0.5rem;
                                ">
                                    <div style="
                                        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
                                        height: 100%;
                                        width: {min(100, (result['predicted_time_minutes']/60)*100):.0f}%;
                                        border-radius: 4px;
                                    "></div>
                                </div>
                                <div style="font-size: 0.75rem; color: #6c757d;">Estimated completion time</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Estimation failed: {e}")
                else:
                    st.error("System not ready. Please check model status.")
        else:
            # Better placeholder without empty feeling
            st.markdown("""
            <div style='
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                border: 1px solid #dee2e6;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            '>
                <div style='font-size: 1.2rem; color: #6c757d; margin-bottom: 1rem;'>Ready for Analysis</div>
                <div style='font-size: 0.9rem; color: #6c757d; line-height: 1.5;'>
                    Enter your material parameters and click<br>
                    <strong>Calculate Estimate</strong> to get instant results
                </div>
                <div style='margin-top: 1.5rem; font-size: 3rem; opacity: 0.3;'>⚡</div>
            </div>
            """, unsafe_allow_html=True)

def show_batch_prediction():
    """Show batch prediction interface"""
    # Better balanced columns for batch processing
    col1, col2 = st.columns([0.4, 0.6])
    
    with col1:
        st.markdown("<h4 style='color: #2c3e50; margin-bottom: 1.5rem;'>Data Input</h4>", unsafe_allow_html=True)
        
        # Clean file upload section
        st.markdown("**Upload File**")
        uploaded_file = st.file_uploader(
            "Choose CSV or Excel file",
            type=['csv', 'xlsx'],
            help="File should contain columns: Espesor, Cutting_Length"
        )
        
        st.markdown("<br>**Manual Entry**", unsafe_allow_html=True)
        
        # Initialize session state for batch data
        if 'batch_jobs' not in st.session_state:
            st.session_state.batch_jobs = pd.DataFrame({
                'Espesor': [5.0, 8.0, 10.0],
                'Cutting_Length': [10.0, 15.0, 20.0]
            })
        
        # Data editor
        edited_df = st.data_editor(
            st.session_state.batch_jobs,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Espesor": st.column_config.NumberColumn(
                    "Thickness (mm)",
                    min_value=0.1,
                    max_value=50.0,
                    step=0.1,
                    format="%.1f"
                ),
                "Cutting_Length": st.column_config.NumberColumn(
                    "Length (m)",
                    min_value=0.1,
                    max_value=1000.0,
                    step=0.1,
                    format="%.1f"
                )
            }
        )
        
        # Update session state
        st.session_state.batch_jobs = edited_df
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Clean calculate button
        predict_button = st.button(
            "Process Batch", 
            type="primary", 
            use_container_width=True,
            help="Process all jobs and generate estimates"
        )
    
    with col2:
        st.markdown("<h4 style='color: #2c3e50; margin-bottom: 1.5rem;'>Batch Results</h4>", unsafe_allow_html=True)
        
        # Determine data source
        data_df = None
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data_df = pd.read_csv(uploaded_file)
                else:
                    data_df = pd.read_excel(uploaded_file)
                st.success(f"✅ File loaded: {len(data_df)} jobs")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            data_df = edited_df
        
        if predict_button and data_df is not None and not data_df.empty:
            try:
                with st.spinner(f"Processing {len(data_df)} jobs..."):
                    # Validate data
                    required_cols = ['Espesor', 'Cutting_Length']
                    if not all(col in data_df.columns for col in required_cols):
                        st.error(f"Missing required columns: {required_cols}")
                        return
                    
                    # Validation always enabled
                    enable_validation = True
                    
                    # Make batch predictions
                    results = st.session_state.model_service.predict_batch(data_df, enable_validation)
                    
                    # Format results with validation info
                    from utils.validation_helpers import format_batch_results_with_validation, display_validation_summary, create_validation_explanation
                    results_df = format_batch_results_with_validation(results)
                    
                    # Clean success message
                    st.success(f"✅ Processed {len(results_df)} jobs successfully")
                    
                    # Validation summary (always shown)
                    st.markdown("**Validation Summary**")
                    display_validation_summary(results_df)
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Summary statistics in clean cards
                    st.markdown("**Summary Statistics**")
                    col2a, col2b = st.columns(2)
                    
                    with col2a:
                        st.metric("Total Jobs", len(results_df))
                        avg_time = results_df['Predicted Time (min)'].mean()
                        st.metric("Average Time", f"{avg_time:.1f} min")
                    
                    with col2b:
                        total_time = results_df['Predicted Time (min)'].sum()
                        st.metric("Total Time", f"{total_time:.1f} min")
                        max_time = results_df['Predicted Time (min)'].max()
                        st.metric("Max Time", f"{max_time:.1f} min")
                    
                    st.markdown("<br>**Detailed Results**", unsafe_allow_html=True)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Add validation explanation
                    create_validation_explanation()
                    
                    # Clean download button
                    csv = export_results_to_csv(results_df)
                    st.download_button(
                        label="Download Results (CSV)",
                        data=csv,
                        file_name="batch_estimates.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"Batch processing failed: {e}")
        else:
            # Better placeholder for batch results
            st.markdown("""
            <div style='
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                border: 1px solid #dee2e6;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            '>
                <div style='font-size: 1.2rem; color: #6c757d; margin-bottom: 1rem;'>Ready for Batch Processing</div>
                <div style='font-size: 0.9rem; color: #6c757d; line-height: 1.5;'>
                    Upload a file or enter data manually,<br>
                    then click <strong>Process Batch</strong> to analyze multiple jobs
                </div>
                <div style='margin-top: 1.5rem; font-size: 3rem; opacity: 0.3;'>📊</div>
            </div>
            """, unsafe_allow_html=True)

@st.cache_resource
def get_model_service():
    """Get cached model service"""
    try:
        # Try ONNX first
        return ONNXModelService(), "ONNX"
    except Exception as e:
        try:
            # Fallback to sklearn
            return SklearnModelService(), "sklearn"
        except Exception as e2:
            raise Exception(f"Failed to load any models: {e2}")

def initialize_services():
    """Initialize model services with proper caching"""
    if 'model_service' not in st.session_state:
        try:
            model_service, model_type = get_model_service()
            st.session_state.model_service = model_service
            st.session_state.model_type = model_type
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            st.error("Please ensure models are available in models/ directory")

if __name__ == "__main__":
    main()