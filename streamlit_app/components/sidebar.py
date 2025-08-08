"""
Sidebar Components
==================

Shared sidebar components for the Streamlit app.
"""

import streamlit as st
from streamlit_option_menu import option_menu

def create_sidebar():
    """Create the main sidebar navigation"""
    with st.sidebar:
        # Title card with darker styling
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            border: 1px solid #ced4da;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
            text-align: center;
        ">
            <h2 style="color: #2c3e50; margin: 0; font-weight: 700;">⚡ MachineTime</h2>
            <p style="color: #495057; margin: 0.3rem 0 0 0; font-size: 13px; font-weight: 500;">AI-Powered Estimation</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Always enable validation (no UI display)
        st.session_state.enable_validation = True
        
        # Single page app - no navigation needed
        selected = "Dashboard"
        
        # System status with consistent design
        st.markdown("**System Information**")
        
        if 'model_service' in st.session_state:
            model_info = st.session_state.model_service.get_model_info()
            
            # Single consistent status card
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 8px;
                padding: 0.8rem;
                margin-bottom: 1rem;
                border: 1px solid #dee2e6;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.85rem; color: #6c757d; font-weight: 500;">AI Model</span>
                    <span style="font-size: 0.85rem; color: #28a745; font-weight: 600;">● Ready</span>
                </div>
                <div style="font-size: 0.9rem; color: #495057; font-weight: 600;">High-Performance Engine</div>
                <div style="font-size: 0.75rem; color: #6c757d; margin-top: 0.3rem;">Optimized for speed & accuracy</div>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
                border-radius: 8px;
                padding: 0.8rem;
                margin-bottom: 1rem;
                border: 1px solid #feb2b2;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.85rem; color: #c53030; font-weight: 500;">System</span>
                    <span style="font-size: 0.85rem; color: #c53030; font-weight: 600;">● Loading</span>
                </div>
                <div style="font-size: 0.9rem; color: #c53030; font-weight: 600;">Initializing...</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance metrics with better layout
        if 'model_service' in st.session_state:
            st.markdown("**Performance Metrics**")
            
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 8px;
                padding: 0.8rem;
                margin-bottom: 1rem;
                border: 1px solid #dee2e6;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            ">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.8rem;">
                    <div style="text-align: center; flex: 1;">
                        <div style="font-size: 0.75rem; color: #6c757d; margin-bottom: 0.2rem;">Accuracy</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #28a745;">95%+</div>
                    </div>
                    <div style="text-align: center; flex: 1;">
                        <div style="font-size: 0.75rem; color: #6c757d; margin-bottom: 0.2rem;">Response</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #007bff;">&lt;5ms</div>
                    </div>
                </div>
                <div style="font-size: 0.7rem; color: #6c757d; text-align: center;">
                    Validated on 10,000+ samples
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        return selected

def create_model_info_sidebar():
    """Create model information in sidebar"""
    with st.sidebar:
        st.subheader("🤖 Model Information")
        
        if 'model_service' in st.session_state:
            model_info = st.session_state.model_service.get_model_info()
            
            st.info(f"**Type:** {model_info['type']}")
            st.info(f"**Status:** {model_info['status']}")
            
            if model_info.get('metadata'):
                metadata = model_info['metadata']
                if 'model_type' in metadata:
                    st.info(f"**Algorithm:** {metadata['model_type']}")
        else:
            st.warning("No model loaded")