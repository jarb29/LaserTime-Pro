"""
Single Prediction Component
Handles individual job prediction functionality
"""

import streamlit as st


def show_single_prediction():
    """Show single prediction interface"""
    col1, col2 = st.columns([0.4, 0.6])
    
    with col1:
        espesor, cutting_length, predict_button = _show_input_form()
    
    with col2:
        _show_prediction_results(predict_button, espesor, cutting_length)


def _show_input_form():
    """Show input form section"""
    st.markdown("<h4 style='color: #2c3e50; margin-bottom: 1.5rem;'>Parámetros del Trabajo</h4>", unsafe_allow_html=True)
    
    espesor = st.number_input(
        "Espesor del Material (mm)",
        min_value=0.1,
        max_value=50.0,
        value=5.0,
        step=0.1,
        format="%.1f",
        help="Espesor del material en milímetros"
    )
    
    cutting_length = st.number_input(
        "Longitud de Corte (m)",
        min_value=0.1,
        max_value=1000.0,
        value=10.0,
        step=0.1,
        format="%.1f",
        help="Longitud total de corte en metros"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    predict_button = st.button(
        "Calcular Estimación", 
        type="primary", 
        use_container_width=True,
        help="Generar estimación de tiempo de mecanizado"
    )
    
    return espesor, cutting_length, predict_button


def _show_prediction_results(predict_button, espesor, cutting_length):
    """Show prediction results section"""
    st.markdown("<h3 style='text-align: center; color: #2c3e50; margin-bottom: 1.5rem;'>Resultados de Estimación</h3>", unsafe_allow_html=True)
    
    if predict_button:
        _process_single_prediction(espesor, cutting_length)
    else:
        _show_results_placeholder()


def _process_single_prediction(espesor, cutting_length):
    """Process single prediction"""
    from utils.helpers import validate_input_data
    
    # Validate inputs
    errors = validate_input_data(espesor, cutting_length)
    
    if errors:
        for error in errors:
            st.error(error)
        return
    
    # Make prediction
    if 'model_service' not in st.session_state:
        st.error("Sistema no listo. Por favor verifica el estado del modelo.")
        return
    
    try:
        with st.spinner("Calculando estimación..."):
            result = st.session_state.model_service.predict_single(
                espesor, cutting_length, True
            )
        
        _display_prediction_result(result, espesor, cutting_length)
        
    except Exception as e:
        st.error(f"Error en la estimación: {e}")


def _display_prediction_result(result, espesor, cutting_length):
    """Display formatted prediction result"""
    # Main result
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
            <div style='font-size: 1.2rem; font-weight: 600;'>Tiempo de Mecanizado Estimado</div>
            <div style='font-size: 2rem; font-weight: 700; margin: 0.5rem 0;'>{minutes} min {seconds} seg</div>
            <div style='font-size: 0.9rem; opacity: 0.8;'>Predicción IA de alto rendimiento</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics and visualization
    col2a, col2b = st.columns([1, 1])
    
    with col2a:
        _show_metrics(result, espesor, cutting_length)
    
    with col2b:
        _show_time_visualization(result)


def _show_metrics(result, espesor, cutting_length):
    """Show metric cards"""
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
            <div style="font-size: 0.875rem; color: #6c757d; font-weight: 500;">Total Segundos</div>
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
            <div style="font-size: 0.875rem; color: #6c757d; font-weight: 500;">Espesor del Material</div>
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
            <div style="font-size: 0.875rem; color: #6c757d; font-weight: 500;">Longitud de Corte</div>
            <div style="font-size: 1.6rem; font-weight: 600; color: #495057;">{cutting_length}m</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _show_time_visualization(result):
    """Show time visualization"""
    total_minutes = result['predicted_time_minutes']
    
    if total_minutes >= 60:
        hours = int(total_minutes // 60)
        remaining_minutes = int(total_minutes % 60)
        time_display = f"{hours}h {remaining_minutes}m"
        time_unit = "horas"
        time_value = f"{total_minutes/60:.1f}"
    else:
        time_display = f"{total_minutes:.1f} min"
        time_unit = "minutos"
        time_value = f"{total_minutes:.1f}"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #dee2e6;
    ">
        <div style="font-size: 0.875rem; color: #6c757d; margin-bottom: 1rem;">Desglose de Tiempo de Procesamiento</div>
        <div style="font-size: 2.5rem; font-weight: 700; color: #28a745; margin-bottom: 0.5rem;">{time_value}</div>
        <div style="font-size: 1rem; color: #6c757d; margin-bottom: 1rem;">{time_unit}</div>
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
                width: {min(100, (total_minutes/60)*100):.0f}%;
                border-radius: 4px;
            "></div>
        </div>
        <div style="font-size: 0.75rem; color: #6c757d;">Tiempo estimado de finalización: {time_display}</div>
    </div>
    """, unsafe_allow_html=True)


def _show_results_placeholder():
    """Show placeholder when no results"""
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    '>
        <div style='font-size: 1.2rem; color: #6c757d; margin-bottom: 1rem;'>Listo para Análisis</div>
        <div style='font-size: 0.9rem; color: #6c757d; line-height: 1.5;'>
            Ingresa los parámetros del material y haz clic en<br>
            <strong>Calcular Estimación</strong> para obtener resultados instantáneos
        </div>
        <div style='margin-top: 1.5rem; font-size: 3rem; opacity: 0.3;'>⚡</div>
    </div>
    """, unsafe_allow_html=True)