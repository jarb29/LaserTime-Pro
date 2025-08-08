"""
Validation Helper Functions
Helper functions for displaying validation results
"""

import pandas as pd
import streamlit as st


def format_batch_results_with_validation(results):
    """Format batch results to include validation information"""
    formatted_results = []
    
    for result in results:
        validation = result.get('validation', {})
        
        formatted_result = {
            'Thickness (mm)': result.get('Espesor (mm)', 0),
            'Length (m)': result.get('Cutting Length (m)', 0),
            'Predicted Time (min)': result.get('predicted_time_minutes', 0)
        }
        
        # Add validation details (always available)
        formatted_result.update({
            'ML Pred (min)': round(validation.get('ml_prediction', 0), 1),
            'Speed Est (min)': round(validation.get('speed_estimate', 0), 1),
            'Validated (min)': round(validation.get('validated_result', 0), 1),
            'Cutting Speed (mt/min)': round(validation.get('cutting_speed', 0), 1),
            'Status': validation.get('status', 'Validated')
        })
        
        formatted_results.append(formatted_result)
    
    return pd.DataFrame(formatted_results)


def display_validation_summary(results_df):
    """Display summary of validation results"""
    if 'Status' not in results_df.columns:
        return
    
    # Count validation statuses
    status_counts = results_df['Status'].value_counts()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        validated_count = status_counts.get('Validated', 0)
        st.metric("Validated Predictions", validated_count)
    
    with col2:
        error_count = status_counts.get('Error', 0)
        st.metric("Error Cases", error_count)
    
    with col3:
        total_count = len(results_df)
        validation_rate = (validated_count / total_count * 100) if total_count > 0 else 0
        st.metric("Validation Rate", f"{validation_rate:.1f}%")


def create_validation_explanation():
    """Create an explanation of the validation system"""
    with st.expander("ℹ️ About Validation System"):
        st.markdown("""
        **Cutting Speed Validation** combines AI predictions with physical cutting parameters:
        
        **🤖 ML Prediction**: Based on historical machining data
        **⚡ Speed Estimate**: Calculated from cutting speed tables (Length ÷ Speed)
        **✅ Validated Result**: Final prediction using validation logic
        
        **Validation Logic**:
        1. Calculate setup time adjustment: `(Length ÷ 100) × 60 minutes`
        2. Apply base time: `ML Prediction + Adjustment`
        3. Choose higher value: `max(Base Time - Adjustment, Speed Estimate)`
        4. Add setup time back: `Final Time + Adjustment`
        
        **Status Meanings**:
        - ✅ **Validated**: Both ML and cutting speed data available (normal operation)
        - ❌ **Error**: Validation failed, using ML fallback
        """)