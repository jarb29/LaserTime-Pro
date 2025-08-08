"""
Chart Components
================

Reusable chart components for visualizations.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_prediction_gauge(predicted_minutes, max_time=120):
    """Create a gauge chart for prediction results"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = predicted_minutes,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Time (minutes)"},
        delta = {'reference': max_time/2},
        gauge = {
            'axis': {'range': [None, max_time]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_time/3], 'color': "lightgray"},
                {'range': [max_time/3, 2*max_time/3], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_time * 0.8
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_batch_results_chart(results_df):
    """Create a bar chart for batch prediction results"""
    if results_df.empty:
        return None
    
    fig = px.bar(
        results_df,
        x=results_df.index,
        y='predicted_time_minutes',
        title="Batch Prediction Results",
        labels={'predicted_time_minutes': 'Time (minutes)', 'index': 'Job Number'},
        color='predicted_time_minutes',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="Job Number",
        yaxis_title="Predicted Time (minutes)",
        height=400
    )
    
    return fig

def create_espesor_analysis_chart(results_df):
    """Create analysis chart by material thickness"""
    if results_df.empty:
        return None
    
    fig = px.scatter(
        results_df,
        x='Espesor (mm)',
        y='predicted_time_minutes',
        size='Cutting Length (m)',
        title="Time vs Material Thickness",
        labels={'predicted_time_minutes': 'Time (minutes)'},
        color='Cutting Length (m)',
        color_continuous_scale='plasma'
    )
    
    fig.update_layout(height=400)
    return fig

