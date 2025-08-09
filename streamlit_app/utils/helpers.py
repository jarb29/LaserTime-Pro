"""
Helper Functions
================

Utility functions for the Streamlit app.
"""

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

def export_results_to_csv(results_df):
    """Export results to CSV"""
    return results_df.to_csv(index=False)