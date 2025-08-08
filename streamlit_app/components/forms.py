"""
Form Components
===============

Reusable form components for the Streamlit app.
"""

import streamlit as st
import pandas as pd

def create_single_prediction_form():
    """Create form for single prediction"""
    st.subheader("🔮 Single Job Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        espesor = st.number_input(
            "Material Thickness (mm)",
            min_value=0.1,
            max_value=50.0,
            value=5.0,
            step=0.1,
            help="Thickness of the material in millimeters"
        )
    
    with col2:
        cutting_length = st.number_input(
            "Cutting Length (m)",
            min_value=0.1,
            max_value=1000.0,
            value=10.0,
            step=0.1,
            help="Total cutting length in meters"
        )
    
    predict_button = st.button("🚀 Predict Time", type="primary")
    
    return espesor, cutting_length, predict_button

def create_batch_prediction_form():
    """Create form for batch prediction"""
    st.subheader("📋 Batch Job Prediction")
    
    # Option 1: File upload
    st.write("**Option 1: Upload CSV/Excel File**")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx'],
        help="File should contain columns: Espesor, Cutting_Length"
    )
    
    # Option 2: Manual entry
    st.write("**Option 2: Manual Entry**")
    
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
                step=0.1
            ),
            "Cutting_Length": st.column_config.NumberColumn(
                "Length (m)",
                min_value=0.1,
                max_value=1000.0,
                step=0.1
            )
        }
    )
    
    # Update session state
    st.session_state.batch_jobs = edited_df
    
    predict_batch_button = st.button("🚀 Predict Batch", type="primary")
    
    return uploaded_file, edited_df, predict_batch_button

