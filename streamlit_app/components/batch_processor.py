"""
Batch Processing Component with Intelligent Filter System
Handles file upload, data mapping, and batch prediction
"""

import streamlit as st
import pandas as pd
import io
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Check for openpyxl availability
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


# Helper functions should be defined before they are called
def _clean_data(data):
    cleaning_log = []
    
    cols_to_remove = [col for col in data.columns if data[col].isnull().sum() > 2]
    if cols_to_remove:
        data = data.drop(columns=cols_to_remove)
        cleaning_log.append(f"🗑️ Eliminadas {len(cols_to_remove)} columnas con >2 valores NaN")
    
    before_rows = len(data)
    data = data.dropna(axis=0, how='all')
    removed_rows = before_rows - len(data)
    if removed_rows > 0:
        cleaning_log.append(f"🗑️ Eliminadas {removed_rows} filas completamente vacías")
    
    return data.reset_index(drop=True), cleaning_log


def _auto_detect_header_row(data):
    target_keywords = ['FECHA', 'PV', 'VCLIENTE']
    
    for i in range(min(10, len(data))):
        row_values = [str(val).strip().upper() for val in data.iloc[i].values if pd.notna(val) and str(val).strip()]
        
        # Check if at least 2 of the 3 keywords are found
        matches = sum(1 for keyword in target_keywords if any(keyword in val for val in row_values))
        if matches >= 2:
            return i
            
    return None


def _apply_headers(data, header_row):
    """Apply headers and return clean data"""
    try:
        headers = data.iloc[header_row].fillna('').astype(str).tolist()
        headers = [h.strip() if h.strip() else f'Col_{i}' for i, h in enumerate(headers)]
        
        clean_data = data.iloc[header_row + 1:].copy()
        clean_data.columns = headers
        clean_data = clean_data.reset_index(drop=True)
        
        # Format FECHA column to show only date
        if 'FECHA' in clean_data.columns:
            clean_data['FECHA'] = pd.to_datetime(clean_data['FECHA'], errors='coerce').dt.date
        
        return clean_data
        
    except Exception as e:
        st.error(f"❌ Error al aplicar encabezados: {e}")
        return None


def _add_plan_column(result_df, start_date=None, work_start_hour=8, work_end_hour=20):
    """Add Start and End columns with precise datetime scheduling (TiempoP in minutes)"""
    if start_date is None:
        start_date = datetime.now().date()
    
    start_datetimes = []
    end_datetimes = []
    current_datetime = datetime.combine(start_date, datetime.min.time().replace(hour=work_start_hour))
    
    for _, row in result_df.iterrows():
        tiempo_p = row['TiempoP']  # Already in minutes
        
        # Skip weekends
        while current_datetime.weekday() >= 5:  # 5=Saturday, 6=Sunday
            current_datetime += timedelta(days=1)
            current_datetime = current_datetime.replace(hour=work_start_hour, minute=0, second=0)
        
        # Check if we need to move to next day (after 5 PM)
        work_end_time = current_datetime.replace(hour=work_end_hour, minute=0, second=0)
        if current_datetime >= work_end_time:
            current_datetime += timedelta(days=1)
            while current_datetime.weekday() >= 5:
                current_datetime += timedelta(days=1)
            current_datetime = current_datetime.replace(hour=work_start_hour, minute=0, second=0)
        
        # Job starts at current datetime
        job_start = current_datetime
        start_datetimes.append(job_start)
        
        # Calculate end datetime
        remaining_minutes = tiempo_p
        end_datetime = current_datetime
        
        while remaining_minutes > 0:
            # Calculate remaining time in current workday (until 5 PM)
            current_work_end = end_datetime.replace(hour=work_end_hour, minute=0, second=0)
            time_until_end_of_day = (current_work_end - end_datetime).total_seconds() / 60
            
            if remaining_minutes <= time_until_end_of_day:
                # Job finishes today
                end_datetime += timedelta(minutes=remaining_minutes)
                remaining_minutes = 0
            else:
                # Job continues to next day at 8 AM
                remaining_minutes -= time_until_end_of_day
                end_datetime += timedelta(days=1)
                while end_datetime.weekday() >= 5:
                    end_datetime += timedelta(days=1)
                end_datetime = end_datetime.replace(hour=work_start_hour, minute=0, second=0)
        
        end_datetimes.append(end_datetime)
        current_datetime = end_datetime
    
    result_df['Start'] = start_datetimes
    result_df['End'] = end_datetimes
    return result_df

def _process_batch_data(data_df, original_data):
    """Process batch predictions"""
    try:
        required_cols = ['Espesor', 'Longitud de corte (m)']
        if not all(col in data_df.columns for col in required_cols):
            st.error(f"❌ Columnas faltantes: {required_cols}")
            return None
        
        # Clean data for model
        model_data = data_df[required_cols].dropna()
        
        # Check if model service is available
        if 'model_service' not in st.session_state:
            st.error("Sistema no listo. Por favor verifique el estado del modelo.")
            return None
        
        with st.spinner(f"🔄 Procesando {len(model_data)} trabajos..."):
            predictions = []
            for index, row in model_data.iterrows():
                espesor = row['Espesor']
                longitud = row['Longitud de corte (m)']
                
                # Use model service for prediction
                result = st.session_state.model_service.predict_single(
                    espesor, longitud, False
                )
                predicted_time = result['predicted_time_minutes']
                
                predictions.append(predicted_time)
            
            # Create result dataframe with ALL original columns + predictions
            result_df = original_data.iloc[:len(model_data)].copy()  # Match row count
            result_df['TiempoP'] = predictions
            
            # Add Plan column
            result_df = _add_plan_column(result_df)
            
            st.success(f"🎉 ¡Procesamiento completado de {len(result_df)} filas!")
            return result_df
            
    except Exception as e:
        st.error(f"❌ Error en el procesamiento: {e}")
        return None


def _show_placeholder():
    """Show placeholder"""
    st.markdown("""
    <div style='
        background: #f8f9fa;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        border: 1px solid #dee2e6;
    '>
        <div style='font-size: 1.2rem; color: #6c757d; margin-bottom: 1rem;'>🎯 Listo para Procesar</div>
        <div style='font-size: 0.9rem; color: #6c757d;'>
            Sube un archivo o ingresa datos manualmente,<br>
            luego haz clic en <strong>Procesar Lote</strong>
        </div>
        <div style='margin-top: 1rem; font-size: 2rem; opacity: 0.3;'>🚀</div>
    </div>
    """, unsafe_allow_html=True)


def _show_file_processing():
    uploaded_file = st.session_state.uploaded_file
    
    if 'excel_file' not in st.session_state:
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xlsm'):
            if not OPENPYXL_AVAILABLE:
                st.error("❌ Error: openpyxl no está instalado. Por favor instala openpyxl para procesar archivos Excel.")
                st.info("💡 Sugerencia: Usa archivos CSV como alternativa.")
                st.code("pip install openpyxl", language="bash")
                return
            try:
                st.session_state.excel_file = pd.ExcelFile(uploaded_file)
                st.session_state.sheet_names = st.session_state.excel_file.sheet_names
                st.session_state.selected_sheet = st.session_state.sheet_names[0]
            except Exception as e:
                st.error(f"❌ Error al procesar archivo Excel: {e}")
                return
        else:
            st.session_state.excel_file = None
            st.session_state.initial_data = pd.read_csv(uploaded_file)

    if st.session_state.excel_file:
        selected_sheet = st.selectbox(
            "📋 Seleccionar Hoja:", 
            st.session_state.sheet_names,
            index=st.session_state.sheet_names.index(st.session_state.selected_sheet),
            key="sheet_selector"
        )
        
        if selected_sheet != st.session_state.selected_sheet:
            st.session_state.selected_sheet = selected_sheet
            st.rerun()
            
        st.session_state.initial_data = pd.read_excel(st.session_state.uploaded_file, sheet_name=selected_sheet)
    
    # Process data silently after sheet selection
    if 'initial_data' in st.session_state:
        cleaned_data, _ = _clean_data(st.session_state.initial_data)
        detected_row = _auto_detect_header_row(cleaned_data)
        
        if detected_row is not None:
            st.session_state.data_with_headers = _apply_headers(cleaned_data, detected_row)
            st.session_state.processing_step = 'column_mapping'
            st.rerun()
        else:
            st.error("❌ No se pudo procesar el archivo. Por favor verifica el formato de datos.")


def _show_column_mapping():
    data = st.session_state.data_with_headers.copy()
    
    if 'Longitud de corte (m)' not in data.columns:
        data['Longitud de corte (m)'] = 10.0
    
    espesor_col = st.selectbox(
        "📊 Selecciona la columna 'Espesor':",
        ["-- Seleccionar --"] + list(data.columns),
        key="espesor_col"
    )
    
    # Show results dataframe if available, otherwise show original data
    if 'results_df' in st.session_state and st.session_state.results_df is not None:
        # Show summary statistics
        result_df = st.session_state.results_df
        total_jobs = len(result_df)
        total_time_minutes = result_df['TiempoP'].sum()
        total_time_hours = total_time_minutes / 60
        completion_date = result_df['End'].max()
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📊 Total Trabajos",
                value=f"{total_jobs:,}"
            )
        
        with col2:
            st.metric(
                label="⏱️ Tiempo Total",
                value=f"{total_time_hours:.1f} horas",
                delta=f"{total_time_minutes:.0f} minutos"
            )
        
        with col3:
            st.metric(
                label="📅 Fecha de Finalización",
                value=completion_date.strftime("%Y-%m-%d %H:%M")
            )
        
        # Add visualization tabs
        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Análisis CNC", "🎨 Cronograma Moderno", "☀️ Análisis Sunburst", "🌊 Flujo Sankey"])
        
        with tab1:
            st.markdown("**🔧 Análisis de Tiempo de Procesamiento CNC**")
            
            # Create professional seaborn plot
            plt.style.use('default')
            sns.set_palette("husl")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Group data by CNC and sum TiempoP
            plot_data = result_df.groupby('CNC:')['TiempoP'].sum().reset_index()
            plot_data = plot_data.sort_values('TiempoP', ascending=False)
            
            # Create bar plot
            bars = sns.barplot(
                data=plot_data, 
                x='CNC:', 
                y='TiempoP',
                hue='CNC:',
                ax=ax,
                palette='viridis',
                legend=False
            )
            
            # Customize the plot
            ax.set_title('Distribución de Tiempo de Procesamiento CNC', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Código CNC', fontsize=12, fontweight='bold')
            ax.set_ylabel('Tiempo de Procesamiento (minutos)', fontsize=12, fontweight='bold')
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars.patches:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=9)
            
            # Style improvements
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with tab2:
            st.markdown("**🎨 Cronograma de Producción Moderno**")
            
            # Create modern timeline with enhanced visual design
            timeline_df = result_df.copy().sort_values('Start').reset_index(drop=True)
            timeline_df['Job_ID'] = 'Job ' + (timeline_df.index + 1).astype(str)
            timeline_df['Duration_Hours'] = (timeline_df['End'] - timeline_df['Start']).dt.total_seconds() / 3600
            timeline_df['Start_Display'] = timeline_df['Start'].dt.strftime('%a %Y-%m-%d %H:%M')
            timeline_df['End_Display'] = timeline_df['End'].dt.strftime('%a %Y-%m-%d %H:%M')
            
            # Create modern timeline using plotly with glass morphism style
            fig_modern = px.timeline(
                timeline_df,
                x_start='Start',
                x_end='End',
                y='Job_ID',
                color='TiempoP',
                title='Cronograma de Producción Moderno - Estilo Glass Morphism',
                labels={'Job_ID': 'Trabajos de Producción', 'TiempoP': 'Tiempo de Procesamiento (min)'},
                hover_data={'CNC:': True, 'Duration_Hours': ':.1f', 'Start_Display': True, 'End_Display': True},
                color_continuous_scale='Viridis'
            )
            
            # Update hover template to show date and day
            fig_modern.update_traces(
                hovertemplate='<b>%{y}</b><br>' +
                             'Inicio: %{customdata[2]}<br>' +
                             'Fin: %{customdata[3]}<br>' +
                             'CNC: %{customdata[0]}<br>' +
                             'Duración: %{customdata[1]:.1f} horas<br>' +
                             'Tiempo de Procesamiento: %{marker.color:.0f} min<br>' +
                             '<extra></extra>'
            )
            
            # Modern glass morphism styling
            fig_modern.update_layout(
                height=max(600, len(timeline_df) * 50),
                font=dict(family='Inter, system-ui, sans-serif', size=12),
                title=dict(
                    text='Cronograma de Producción Moderno',
                    x=0.5,
                    font=dict(size=20, family='Inter, system-ui, sans-serif', color='#1f2937')
                ),
                plot_bgcolor='rgba(248, 250, 252, 0.8)',
                paper_bgcolor='rgba(255, 255, 255, 0.95)',
                xaxis=dict(
                    title='Cronograma',
                    tickformat='%Y-%m-%d',
                    tickangle=0,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(148, 163, 184, 0.3)',
                    tickfont=dict(color='#475569', size=11),
                    title_font=dict(color='#334155', size=13)
                ),
                yaxis=dict(
                    title='Trabajos',
                    autorange='reversed',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(148, 163, 184, 0.3)',
                    tickfont=dict(color='#475569', size=11),
                    title_font=dict(color='#334155', size=13)
                ),
                coloraxis_colorbar=dict(
                    title='Tiempo de Procesamiento (min)',
                    title_font=dict(color='#334155'),
                    tickfont=dict(color='#475569')
                ),
                margin=dict(l=100, r=100, t=80, b=60)
            )
            
            # Modern bar styling with rounded corners effect
            fig_modern.update_traces(
                marker=dict(
                    line=dict(width=0),
                    opacity=0.85
                ),
                textfont=dict(size=10, color='white', family='Inter'),
                textposition='inside'
            )
            
            st.plotly_chart(fig_modern, use_container_width=True)
        
        with tab3:
            st.markdown("**☀️ Análisis Jerárquico de Producción**")
            
            # Check available columns and create sunburst
            sunburst_cols = ['Cliente', 'PV', 'Esp.', 'CNC:']
            available_sunburst_cols = [col for col in sunburst_cols if col in result_df.columns]
            
            if len(available_sunburst_cols) >= 3:
                # Create sunburst data
                sunburst_data = result_df.groupby(available_sunburst_cols)['TiempoP'].sum().reset_index()
                
                # Custom warm professional palette
                custom_colors = ['#D2691E', '#8B4513', '#CD853F', '#DEB887', '#F4A460', '#DAA520', '#B8860B', '#9ACD32']
                
                fig_sunburst = px.sunburst(
                    sunburst_data,
                    path=available_sunburst_cols,
                    values='TiempoP',
                    title='Jerarquía de Producción: Cliente → Proyecto → Espesor → CNC',
                    color='TiempoP',
                    color_continuous_scale=[[0, '#8B4513'], [0.3, '#CD853F'], [0.6, '#DEB887'], [1, '#D2691E']]
                )
                
                fig_sunburst.update_layout(
                    font=dict(size=12),
                    height=600
                )
                
                st.plotly_chart(fig_sunburst, use_container_width=True)
            else:
                st.warning(f"Sunburst requiere columnas: {sunburst_cols}. Disponibles: {available_sunburst_cols}")
        
        with tab4:
            st.markdown("**🌊 Análisis de Flujo de Producción**")
            
            # Check available columns for Sankey
            sankey_required_cols = ['Cliente', 'OF', 'PV', 'Esp.', 'Longitud de corte (m)', 'CNC:', 'TiempoP']
            sankey_available_cols = [col for col in sankey_required_cols if col in result_df.columns]
            
            if len(sankey_available_cols) >= 3:
                # Create Sankey diagram data
                sankey_data = result_df.copy()
                

                
                # Create unique node lists with OF included
                clients = sorted(sankey_data['Cliente'].unique().tolist())
                ofs = sorted(sankey_data['OF'].unique().tolist())
                projects = sorted(sankey_data['PV'].unique().tolist())
                espesors = sorted(sankey_data['Esp.'].unique().tolist())
                longitudes = sorted(sankey_data['Longitud de corte (m)'].unique().tolist())
                cncs = sorted(sankey_data['CNC:'].unique().tolist())
                
                # Create clean node labels with prefixes for clarity
                client_labels = [f"Cliente: {c}" for c in clients]
                of_labels = [f"OF: {o}" for o in ofs]
                project_labels = [f"Proyecto: {p}" for p in projects]
                espesor_labels = [f"Esp: {e}mm" for e in espesors]
                longitud_labels = [f"Longitud: {l}m" for l in longitudes]
                cnc_labels = [f"CNC: {c}" for c in cncs]
                
                # Create node labels and indices
                all_nodes = client_labels + of_labels + project_labels + espesor_labels + longitud_labels + cnc_labels
                
                # Create mapping for original values to indices
                node_indices = {}
                for i, client in enumerate(clients):
                    node_indices[client] = i
                for i, of in enumerate(ofs):
                    node_indices[of] = len(clients) + i
                for i, project in enumerate(projects):
                    node_indices[project] = len(clients) + len(ofs) + i
                for i, espesor in enumerate(espesors):
                    node_indices[espesor] = len(clients) + len(ofs) + len(projects) + i
                for i, longitud in enumerate(longitudes):
                    node_indices[longitud] = len(clients) + len(ofs) + len(projects) + len(espesors) + i
                for i, cnc in enumerate(cncs):
                    node_indices[cnc] = len(clients) + len(ofs) + len(projects) + len(espesors) + len(longitudes) + i
                
                # Create flows
                sources = []
                targets = []
                values = []
                
                # Client to OF flows
                client_of = sankey_data.groupby(['Cliente', 'OF'])['TiempoP'].sum().reset_index()
                for _, row in client_of.iterrows():
                    sources.append(node_indices[row['Cliente']])
                    targets.append(node_indices[row['OF']])
                    values.append(row['TiempoP'])
                
                # OF to Project flows
                of_project = sankey_data.groupby(['OF', 'PV'])['TiempoP'].sum().reset_index()
                for _, row in of_project.iterrows():
                    sources.append(node_indices[row['OF']])
                    targets.append(node_indices[row['PV']])
                    values.append(row['TiempoP'])
                
                # Project to Espesor flows
                project_espesor = sankey_data.groupby(['PV', 'Esp.'])['TiempoP'].sum().reset_index()
                for _, row in project_espesor.iterrows():
                    sources.append(node_indices[row['PV']])
                    targets.append(node_indices[row['Esp.']])
                    values.append(row['TiempoP'])
                
                # Espesor to Longitud flows
                espesor_longitud = sankey_data.groupby(['Esp.', 'Longitud de corte (m)'])['TiempoP'].sum().reset_index()
                for _, row in espesor_longitud.iterrows():
                    sources.append(node_indices[row['Esp.']])
                    targets.append(node_indices[row['Longitud de corte (m)']])
                    values.append(row['TiempoP'])
                
                # Longitud to CNC flows
                longitud_cnc = sankey_data.groupby(['Longitud de corte (m)', 'CNC:'])['TiempoP'].sum().reset_index()
                for _, row in longitud_cnc.iterrows():
                    sources.append(node_indices[row['Longitud de corte (m)']])
                    targets.append(node_indices[row['CNC:']])
                    values.append(row['TiempoP'])
                

                
                # Professional corporate color scheme
                node_colors = (['#2E86AB'] * len(clients) +     # Professional blue for clients
                              ['#6A994E'] * len(ofs) +         # Green for work orders
                              ['#A23B72'] * len(projects) +    # Deep magenta for projects
                              ['#E76F51'] * len(espesors) +    # Coral for thickness
                              ['#264653'] * len(longitudes) +  # Dark teal for length
                              ['#F18F01'] * len(cncs))         # Professional orange for CNCs
                
                link_colors = ['rgba(46, 134, 171, 0.3)'] * len(sources)  # Semi-transparent blue links
                
                # Create Sankey diagram with enhanced readability
                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=20,
                        thickness=25,
                        line=dict(color='white', width=3),
                        label=all_nodes,
                        color=node_colors,
                        hovertemplate='%{label}<br>Tiempo Total: %{value:.1f} min<extra></extra>'
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors,
                        hovertemplate='Flujo: %{value:.1f} min<extra></extra>'
                    )
                )])
                
                fig_sankey.update_layout(
                    title=dict(
                        text='Flujo de Producción: Cliente → Orden de Trabajo → Proyecto → Espesor → Longitud → Máquina CNC',
                        font=dict(size=16, color='#2c3e50')
                    ),
                    font=dict(size=13, family='Inter, system-ui, sans-serif'),
                    height=700,
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                
                st.plotly_chart(fig_sankey, use_container_width=True)
            else:
                st.warning(f"Sankey requiere columnas: {sankey_required_cols}. Disponibles: {sankey_available_cols}")
        
        st.markdown("---")
        st.markdown("**📊 Resultados Detallados:**")
        
        # Add zebra pattern CSS
        st.markdown("""
        <style>
        .stDataFrame tbody tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .stDataFrame tbody tr:nth-child(odd) {
            background-color: #ffffff;
        }
        .stDataFrame tbody tr:hover {
            background-color: #e3f2fd !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        edited_data = st.data_editor(st.session_state.results_df, use_container_width=True, hide_index=True, num_rows="dynamic", key="results_editor")
        
        # Show download button when results are available
        if OPENPYXL_AVAILABLE:
            try:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    # Create a copy for export with properly formatted datetime columns
                    export_df = st.session_state.results_df.copy()
                    
                    # Format datetime columns for Excel export
                    if 'Start' in export_df.columns:
                        export_df['Start'] = export_df['Start'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    if 'End' in export_df.columns:
                        export_df['End'] = export_df['End'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    export_df.to_excel(writer, sheet_name='Predictions', index=False)
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="📈 Descargar Resultados como Excel",
                    data=excel_data,
                    file_name="batch_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error al generar Excel: {e}")
                # Fallback to CSV
                csv_data = st.session_state.results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Resultados como CSV",
                    data=csv_data,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            # Fallback to CSV download if openpyxl is not available
            csv_data = st.session_state.results_df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Resultados como CSV",
                data=csv_data,
                file_name="batch_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.info("💡 Descarga disponible en formato CSV (openpyxl no disponible para Excel).")
    else:
        edited_data = st.data_editor(data, use_container_width=True, hide_index=True, num_rows="dynamic", key="data_editor")
    
    if espesor_col != "-- Seleccionar --":
        try:
            if 'results_df' in st.session_state and st.session_state.results_df is not None:
                # Use results data if available - check if Espesor column exists
                if 'Espesor' in edited_data.columns:
                    final_df = edited_data[['Espesor', 'Longitud de corte (m)']].copy()
                else:
                    # Fallback to selected column
                    final_df = edited_data[[espesor_col, 'Longitud de corte (m)']].copy()
                    final_df = final_df.rename(columns={espesor_col: 'Espesor'})
            else:
                # Use original data
                final_df = edited_data[[espesor_col, 'Longitud de corte (m)']].copy()
                final_df = final_df.rename(columns={espesor_col: 'Espesor'})
            
            final_df['Espesor'] = pd.to_numeric(final_df['Espesor'], errors='coerce')
            final_df['Longitud de corte (m)'] = pd.to_numeric(final_df['Longitud de corte (m)'], errors='coerce')
            final_df = final_df.dropna()
            
            st.session_state.final_data = final_df
            st.session_state.current_original_data = edited_data  # Store current data state
            
        except Exception as e:
            st.error(f"❌ Error de selección: {e}")


def show_batch_prediction():
    _show_data_input()
    
    if 'final_data' in st.session_state and st.session_state.final_data is not None:
        st.markdown("---")
        predict_button = st.button("🚀 Procesar Lote", type="primary", use_container_width=True)
        _show_batch_results(st.session_state.final_data, predict_button)


def _show_data_input():
    if 'processing_step' not in st.session_state:
        st.session_state.processing_step = 'initial_state'
        
    if st.session_state.processing_step == 'initial_state':
        uploaded_file = st.file_uploader(
            "📁 Subir Archivo",
            type=['csv', 'xlsx', 'xlsm']
        )
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.processing_step = 'file_processing'
            st.rerun()
    
    elif st.session_state.processing_step == 'file_processing':
        _show_file_processing()
    
    elif st.session_state.processing_step == 'column_mapping':
        _show_column_mapping()


def _show_batch_results(data, predict_button):
    if predict_button:
        original_data = st.session_state.get('current_original_data', st.session_state.data_with_headers)
        result_df = _process_batch_data(data, original_data)
        
        if result_df is not None:
            # Store results in session state to show in data editor
            st.session_state.results_df = result_df
            st.rerun()  # Refresh to show results in data editor