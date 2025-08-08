import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime, timedelta


def plot_tiempo_dashboard(df, year, month):
    """
    Crear un dashboard completo de visualizaciones de tiempo de mecanizado.
    """
    # Set the style properly
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 9})

    # Filter and prepare data
    filtered_df = df[(df['Year'] == year) & (df['Month'] == month)].copy()

    if filtered_df.empty:
        print(f"No se encontraron datos para {year}-{month}")
        return None

    # Process data
    filtered_df['Date'] = pd.to_datetime(filtered_df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1))
    filtered_df['Timestamp'] = pd.to_datetime(filtered_df['Timestamp'])
    filtered_df['Hour'] = filtered_df['Timestamp'].dt.hour

    grouped = filtered_df.groupby(['Date', 'Turno'])['Tiempo'].sum().reset_index()
    pivot_df = grouped.pivot(index='Date', columns='Turno', values='Tiempo').reset_index()
    pivot_df = pivot_df.sort_values('Date')

    # Fill NaN and convert to hours
    for shift in ['D', 'N']:
        if shift not in pivot_df.columns:
            pivot_df[shift] = 0
        pivot_df[shift] = pivot_df[shift] / 3600

    # Replace any inf or NaN values with 0
    pivot_df = pivot_df.replace([np.inf, -np.inf], 0).fillna(0)
    filtered_df = filtered_df.replace([np.inf, -np.inf], 0).fillna(0)

    # Calculate additional metrics
    pivot_df['Total'] = pivot_df['D'] + pivot_df['N']
    pivot_df['Cumulative'] = pivot_df['Total'].cumsum()

    # Add week number for weekly averages
    pivot_df['Week'] = pd.to_datetime(pivot_df['Date']).dt.isocalendar().week

    # Calculate weekly averages
    weekly_avgs = pivot_df.groupby('Week')[['D', 'N', 'Total']].mean().reset_index()

    # Create figure with subplots - make Day vs Night plot taller
    fig = plt.figure(figsize=(20, 28), facecolor='white')
    gs = fig.add_gridspec(7, 2, height_ratios=[2, 1, 1, 1.5, 1, 1, 1])

    # Colors
    day_color = '#1f77b4'
    night_color = '#ff7f0e'
    avg_day_color = '#2ca02c'
    avg_night_color = '#d62728'
    avg_full_day_color = '#9467bd'

    # Common x-axis data
    x = np.arange(len(pivot_df['Date']))
    date_labels = [d.strftime('%d') for d in pivot_df['Date']]

    # Spanish month names
    month_names = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }
    month_name = month_names[month]

    # 1. Main plot - Grouped bars with weekly averages
    ax1 = fig.add_subplot(gs[0, :])
    bar_width = 0.35

    # Plot bars
    day_bars = ax1.bar(x - bar_width/2, pivot_df['D'], bar_width,
                      label='Turno Día', color=day_color, alpha=0.8,
                      edgecolor='white', linewidth=1)
    night_bars = ax1.bar(x + bar_width/2, pivot_df['N'], bar_width,
                       label='Turno Noche', color=night_color, alpha=0.8,
                       edgecolor='white', linewidth=1)

    # Add weekly average lines
    weekly_legend_handles = []
    weekly_legend_labels = []

    for week in weekly_avgs['Week'].unique():
        week_data = pivot_df[pivot_df['Week'] == week]
        if len(week_data) > 0:
            week_avg_day = weekly_avgs[weekly_avgs['Week'] == week]['D'].values[0]
            week_avg_night = weekly_avgs[weekly_avgs['Week'] == week]['N'].values[0]
            week_avg_total = weekly_avgs[weekly_avgs['Week'] == week]['Total'].values[0]

            week_indices = [i for i, date in enumerate(pivot_df['Date']) if pivot_df.loc[pivot_df['Date'] == date, 'Week'].values[0] == week]
            if week_indices:
                min_idx, max_idx = min(week_indices), max(week_indices)
                mid_idx = min_idx + (max_idx - min_idx) / 2

                # Add weekly average lines without labels
                day_line = ax1.plot([min_idx-0.4, max_idx+0.4], [week_avg_day, week_avg_day], '--',
                        color=avg_day_color, alpha=0.7)[0]

                night_line = ax1.plot([min_idx-0.4, max_idx+0.4], [week_avg_night, week_avg_night], '--',
                        color=avg_night_color, alpha=0.7)[0]

                # Add week label
                ax1.text(mid_idx, 0.5, f"S{week}",
                        ha='center', va='bottom', fontsize=8, color='gray')

                # Add to legend instead of placing text on the plot
                weekly_legend_handles.append(day_line)
                weekly_legend_labels.append(f"S{week} Día: {week_avg_day:.1f}h")

                weekly_legend_handles.append(night_line)
                weekly_legend_labels.append(f"S{week} Noche: {week_avg_night:.1f}h")

                # Add weekly average line for total
                total_line = ax1.plot([min_idx-0.4, max_idx+0.4], [week_avg_total, week_avg_total], '--',
                             color=avg_full_day_color, alpha=0.7)[0]

                # Add to legend
                weekly_legend_handles.append(total_line)
                weekly_legend_labels.append(f"S{week} Total: {week_avg_total:.1f}h")

    # Add trend line
    z = np.polyfit(range(len(pivot_df)), pivot_df['Total'], 1)
    p = np.poly1d(z)
    trend_line = ax1.plot(x, p(range(len(pivot_df))), "r--", linewidth=1.5)[0]
    trend_label = f'Tendencia: {"↑" if z[0]>0 else "↓"} {abs(z[0]):.2f}h/día'

    # Customize plot
    ax1.set_title(f'Análisis de Tiempo de Mecanizado - {month_name} {year}',
                 pad=20, fontsize=16, fontweight='bold')
    ax1.set_xlabel('Día del Mes', fontsize=12)
    ax1.set_ylabel('Horas', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(date_labels)

    # Add value labels
    for bar in day_bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.1f}h', ha='center', va='bottom',
                   fontsize=8, fontweight='bold')

    for bar in night_bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.1f}h', ha='center', va='bottom',
                   fontsize=8, fontweight='bold')

    # Add secondary y-axis for total per day
    ax1_twin = ax1.twinx()
    total_line = ax1_twin.plot(x, pivot_df['Total'], 'k-', linewidth=1.5, label='Total')[0]
    ax1_twin.set_ylabel('Horas Totales por Día', color='k')

    # Add value labels to the total line with increased size and different color
    for i, total in enumerate(pivot_df['Total']):
        if total > 0:
            ax1_twin.annotate(f'{total:.1f}h',
                             xy=(i, total),
                             xytext=(0, 4),
                             textcoords='offset points',
                             ha='center',
                             va='bottom',
                             fontsize=10,
                             fontweight='bold',
                            color='darkblue',
                             bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9))

    # Calculate average daily total hours for the month
    avg_month_hours = pivot_df['Total'].mean()

    # Combine all legends with updated Total label that includes the average value
    all_handles = [day_bars[0], night_bars[0], total_line, trend_line] + weekly_legend_handles
    all_labels = [
        'Turno Día',
        'Turno Noche',
        f'Total (Prom: {avg_month_hours:.1f}h/día)',  # Add average value to the label
        trend_label
    ] + weekly_legend_labels

    # Create a legend with multiple columns to save vertical space, centered and 25% larger
    ax1.legend(all_handles, all_labels,
            loc='upper center',  # Center the legend
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=3,  # Use 3 columns for the legend
            fontsize=10,  # Increase font size by 25% (from 8 to 10)
            bbox_to_anchor=(0.5, 1.0))  # Position at the top center



    # 2. Stacked bar chart
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(x, pivot_df['D'], color=day_color, alpha=0.8, edgecolor='white', linewidth=1, label='Día')
    ax2.bar(x, pivot_df['N'], bottom=pivot_df['D'], color=night_color, alpha=0.8,
           edgecolor='white', linewidth=1, label='Noche')

    # Add total labels
    for i, (day, night) in enumerate(zip(pivot_df['D'], pivot_df['N'])):
        total = day + night
        if total > 0:
            ax2.text(i, total, f'{total:.1f}h', ha='center', va='bottom', fontsize=8)

    ax2.set_title('Horas Totales Diarias (Apiladas)', fontsize=14)
    ax2.set_xlabel('Día del Mes')
    ax2.set_ylabel('Horas')
    ax2.set_xticks(x)
    ax2.set_xticklabels(date_labels)
    ax2.legend()

    # 3. Weekly averages
    ax3 = fig.add_subplot(gs[1, 1])
    weekly_avg = pivot_df.groupby('Week')[['D', 'N', 'Total']].mean().reset_index()

    ax3.bar(weekly_avg['Week'], weekly_avg['D'], color=day_color, alpha=0.8,
           edgecolor='white', linewidth=1, label='Día')
    ax3.bar(weekly_avg['Week'], weekly_avg['N'], bottom=weekly_avg['D'],
           color=night_color, alpha=0.8, edgecolor='white', linewidth=1, label='Noche')

    # Add total labels
    for i, row in weekly_avg.iterrows():
        ax3.text(row['Week'], row['Total'], f'{row["Total"]:.1f}h',
                ha='center', va='bottom', fontsize=8)

    ax3.set_title('Promedio Semanal de Horas', fontsize=14)
    ax3.set_xlabel('Número de Semana')
    ax3.set_ylabel('Horas Promedio')
    ax3.legend()

    # 4. Heatmap of daily hours
    ax4 = fig.add_subplot(gs[2, :])
    pivot_df['DayOfWeek'] = pivot_df['Date'].dt.dayofweek
    pivot_df['WeekNum'] = (pivot_df['Date'].dt.day - 1) // 7

    # Create a pivot table for the heatmap
    heatmap_data = pivot_df.pivot_table(
        index='WeekNum',
        columns='DayOfWeek',
        values='Total',
        aggfunc='sum'
    ).fillna(0)

    # Plot heatmap
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.1f', ax=ax4)
    ax4.set_title('Mapa de Calor de Horas Diarias', fontsize=14)
    ax4.set_xlabel('Día de la Semana')
    ax4.set_ylabel('Semana del Mes')

    # Get the actual days present in the data and set labels accordingly
    days_present = sorted(pivot_df['DayOfWeek'].unique())
    day_names = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    ax4.set_xticklabels([day_names[day] for day in days_present])

    # 5. Day vs Night comparison (taller plot)
    ax5 = fig.add_subplot(gs[3, :])

    # Filter out any rows with NaN or inf values for the scatter plot
    valid_data = pivot_df[(np.isfinite(pivot_df['D'])) & (np.isfinite(pivot_df['N']))]

    # Create scatter plot with smaller points and use day numbers as colors
    scatter = ax5.scatter(valid_data['D'], valid_data['N'],
                         alpha=0.7, s=35,
                         c=valid_data['Date'].dt.day,
                         cmap='viridis')

    # Add day numbers as annotations with small offset
    for i, row in valid_data.iterrows():
        day_num = row['Date'].day
        ax5.annotate(str(day_num),
                    xy=(row['D'], row['N']),
                    xytext=(2, 2),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

    # Add a colorbar to show which color corresponds to which day
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Día del Mes')

    # Add diagonal line for equal hours
    max_val = max(pivot_df['D'].max(), pivot_df['N'].max()) + 1
    if np.isfinite(max_val) and max_val > 0:
        ax5.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)

        # Add quadrant labels with better positioning and style
        ax5.text(max_val*0.75, max_val*0.25, 'Dominante Día',
                ha='center', va='center', fontsize=11, alpha=0.7,
                bbox=dict(facecolor='white', alpha=0.7))
        ax5.text(max_val*0.25, max_val*0.75, 'Dominante Noche',
                ha='center', va='center', fontsize=11, alpha=0.7,
                bbox=dict(facecolor='white', alpha=0.7))

    ax5.set_title('Comparación Horas Día vs Noche', fontsize=14)
    ax5.set_xlabel('Horas Turno Día', fontsize=12)
    ax5.set_ylabel('Horas Turno Noche', fontsize=12)
    ax5.grid(True, alpha=0.3)

    # 6. Programa vs Tiempo plot
    ax6 = fig.add_subplot(gs[4, :])

    if 'Programa' in filtered_df.columns:
        # Group by program and calculate total time
        program_stats = filtered_df.groupby('Programa')['Tiempo'].agg(['sum', 'count']).reset_index()
        program_stats.columns = ['Programa', 'Tiempo_Total', 'Cantidad_Ejecuciones']

        # Sort by total time and get top 15 programs
        top_programs = program_stats.sort_values('Tiempo_Total', ascending=False).head(15)

        # Convert seconds to hours for better readability
        top_programs['Horas_Totales'] = top_programs['Tiempo_Total'] / 3600

        # Create bar chart with two y-axes
        bars = ax6.bar(top_programs['Programa'], top_programs['Horas_Totales'],
                      color='skyblue', alpha=0.7, edgecolor='white')

        # Add run count on secondary y-axis
        ax6_twin = ax6.twinx()
        run_count_line = ax6_twin.plot(top_programs['Programa'], top_programs['Cantidad_Ejecuciones'], 'ro-',
                     linewidth=2, markersize=6, label='Cantidad de Ejecuciones')

        # Add labels for hours
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.1f}h', ha='center', va='bottom',
                   fontsize=8, fontweight='bold')

        # Add labels for run count with enhanced styling
        for i, count in enumerate(top_programs['Cantidad_Ejecuciones']):
            ax6_twin.annotate(f'{count}',
                             xy=(i, count),
                             xytext=(0, 4),
                             textcoords='offset points',
                             ha='center',
                             va='bottom',
                             fontsize=10,
                             fontweight='bold',
                             color='red',
                             bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9))

        ax6.set_title('Análisis de Uso de Programas', fontsize=14)
        ax6.set_xlabel('Programa')
        ax6.set_ylabel('Horas Totales')
        ax6_twin.set_ylabel('Número de Ejecuciones')

        # Rotate x-axis labels for better readability
        plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')

        # Add legend
        ax6.legend(['Horas Totales'], loc='upper left')
        ax6_twin.legend(loc='upper right')
    else:
        ax6.text(0.5, 0.5, 'Datos de programa no disponibles', ha='center', va='center')

    # 7. Hourly Production Distribution
    ax7 = fig.add_subplot(gs[5, :])

    # Hourly distribution of machining time
    hourly_data = filtered_df.groupby(['Hour', 'Turno'])['Tiempo'].sum().reset_index()

    if not hourly_data.empty:
        hourly_pivot = hourly_data.pivot(index='Hour', columns='Turno', values='Tiempo').fillna(0)

        # Ensure all hours are represented
        all_hours = pd.DataFrame({'Hour': range(24)})
        hourly_pivot = all_hours.set_index('Hour').join(hourly_pivot).fillna(0)

        # Plot stacked bars
        hourly_pivot.plot(kind='bar', stacked=True, ax=ax7, color=['#1f77b4', '#ff7f0e'])
        ax7.set_title('Distribución Horaria del Tiempo de Mecanizado', fontsize=14)
        ax7.set_xlabel('Hora del Día')
        ax7.set_ylabel('Tiempo Total de Mecanizado (s)')
        ax7.set_xticks(range(24))
        ax7.set_xticklabels([f'{h:02d}:00' for h in range(24)])
        ax7.legend(title='Turno')
    else:
        ax7.text(0.5, 0.5, 'Datos horarios no disponibles', ha='center', va='center')

    # 8. Efficiency metrics by hour
    ax8 = fig.add_subplot(gs[6, :])

    if 'Longitude Corte (m)' in filtered_df.columns:
        hourly_efficiency = filtered_df.groupby('Hour').agg({
            'Tiempo': 'sum',
            'Longitude Corte (m)': 'sum'
        }).reset_index()

        hourly_efficiency['Eficiencia'] = hourly_efficiency['Longitude Corte (m)'] / (hourly_efficiency['Tiempo'] + 1)

        # Create twin axes
        ax8.bar(hourly_efficiency['Hour'], hourly_efficiency['Tiempo'], alpha=0.5, label='Tiempo')
        ax8_twin = ax8.twinx()
        efficiency_line = ax8_twin.plot(hourly_efficiency['Hour'], hourly_efficiency['Eficiencia'], 'r-o', linewidth=2, label='Eficiencia')

        # Add enhanced labels for efficiency values
        for i, eff in enumerate(hourly_efficiency['Eficiencia']):
            if eff > 0:
                ax8_twin.annotate(f'{eff:.3f}',
                                 xy=(hourly_efficiency['Hour'][i], eff),
                                 xytext=(0, 4),
                                 textcoords='offset points',
                                 ha='center',
                                 va='bottom',
                                 fontsize=10,
                                 fontweight='bold',
                                 color='red',
                                 bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9))

        ax8.set_title('Eficiencia Horaria (Longitud de Corte por Segundo)', fontsize=14)
        ax8.set_xlabel('Hora del Día')
        ax8.set_ylabel('Tiempo Total (s)')
        ax8_twin.set_ylabel('Eficiencia (m/s)')
        ax8.set_xticks(range(24))
        ax8.set_xticklabels([f'{h:02d}:00' for h in range(24)])

        # Combine legends
        lines1, labels1 = ax8.get_legend_handles_labels()
        lines2, labels2 = ax8_twin.get_legend_handles_labels()
        ax8.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax8.text(0.5, 0.5, 'Datos de eficiencia no disponibles', ha='center', va='center')

    # 9. Add total hours information
    total_day = pivot_df['D'].sum()
    total_night = pivot_df['N'].sum()
    total_text = f'Horas Totales - Día: {total_day:.1f}h  Noche: {total_night:.1f}h  Combinado: {total_day + total_night:.1f}h'
    plt.figtext(0.5, 0.01, total_text, ha='center', fontsize=12,
                style='italic', bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})

    # Adjust layout with more space between subplots
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.6, wspace=0.3)

    return fig


def plot_tiempo_dashboard_enhanced(df, year, month, save_dir="data_planta", display=True):
    """
    Crear visualizaciones de tiempo de mecanizado y guardarlas como archivos separados.
    Incluye 'Analisis de tiempo de mecanizado' y 'Promedio semanal de horas'.
    Filtra datos desde el mes y año especificados hasta la fecha actual.
    Las semanas comienzan los domingos a las 20:00.

    Args:
        df: DataFrame con los datos
        year: Año de inicio del filtro
        month: Mes de inicio del filtro
        save_dir: Directorio donde guardar las imágenes
        display: Si es True, muestra los gráficos en el notebook además de guardarlos

    Returns:
        dict: Rutas a los archivos guardados y figuras si display=True
    """
    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set the style properly
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'sans-serif',
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold'
    })

    # Get current date info
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month

    # Create a date filter that works across years
    # Convert dates to datetime for proper comparison
    df['DateObj'] = pd.to_datetime(df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1))
    df['DateTime'] = pd.to_datetime(df['Timestamp'])  # Ensure we have full datetime for week calculation
    start_date = pd.to_datetime(f"{year}-{month}-01")

    # Filter data from specified month/year until current date
    filtered_df = df[df['DateObj'] >= start_date].copy()

    if filtered_df.empty:
        print(f"No se encontraron datos desde {year}-{month} hasta la fecha actual")
        return None

    # Process data
    filtered_df['Date'] = filtered_df['DateObj']  # Already calculated above
    filtered_df['Hour'] = filtered_df['DateTime'].dt.hour
    filtered_df['DayOfWeek'] = filtered_df['DateTime'].dt.dayofweek  # 0=Monday, 6=Sunday

    # Define custom week based on Sunday 20:00
    # A date belongs to the next week if it's Sunday and hour >= 20
    def get_custom_week(row):
        date = row['DateTime']
        # Get ISO calendar week
        iso_year, iso_week, _ = date.isocalendar()

        # If it's Sunday (6) and hour >= 20, it belongs to the next week
        if row['DayOfWeek'] == 6 and row['Hour'] >= 20:
            # Add one day to get to next week
            next_day = date + timedelta(days=1)
            iso_year, iso_week, _ = next_day.isocalendar()

        return iso_year, iso_week

    # Apply custom week function
    filtered_df[['CustomYear', 'CustomWeek']] = filtered_df.apply(get_custom_week, axis=1, result_type='expand')

    # Group by date and shift
    grouped = filtered_df.groupby(['Date', 'Turno'])['Tiempo'].sum().reset_index()
    pivot_df = grouped.pivot(index='Date', columns='Turno', values='Tiempo').reset_index()
    pivot_df = pivot_df.sort_values('Date')

    # Fill NaN and convert to hours
    for shift in ['D', 'N']:
        if shift not in pivot_df.columns:
            pivot_df[shift] = 0
        pivot_df[shift] = pivot_df[shift] / 3600

    # Replace any inf or NaN values with 0
    pivot_df = pivot_df.replace([np.inf, -np.inf], 0).fillna(0)
    filtered_df = filtered_df.replace([np.inf, -np.inf], 0).fillna(0)

    # Calculate additional metrics
    pivot_df['Total'] = pivot_df['D'] + pivot_df['N']

    # Merge custom week information to pivot_df
    # First, create a mapping from Date to CustomWeek/CustomYear
    date_to_week = filtered_df.groupby('Date')[['CustomWeek', 'CustomYear']].first()
    pivot_df = pivot_df.merge(date_to_week, on='Date', how='left')

    # Calculate weekly averages based on custom weeks
    weekly_avg = pivot_df.groupby(['CustomYear', 'CustomWeek'])[['D', 'N', 'Total']].mean().reset_index()
    weekly_avg.rename(columns={'CustomYear': 'Year', 'CustomWeek': 'Week'}, inplace=True)

    # Enhanced colors
    day_color = '#2c7fb8'    # Deeper blue
    night_color = '#f46d43'  # Warmer orange
    week_color = '#7a0177'   # Purple for week boundaries

    # Common x-axis data
    x = np.arange(len(pivot_df['Date']))
    date_labels = [d.strftime('%d-%m-%y') for d in pivot_df['Date']]  # Include month and year in labels

    # Spanish month names
    month_names = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }
    month_name = month_names[month]

    # Get the range of dates for the title
    if year == current_year:
        title_text = f'Análisis de Tiempo de Mecanizado - {month_name} a {month_names[current_month]} {year}'
    else:
        title_text = f'Análisis de Tiempo de Mecanizado - {month_name} {year} a {month_names[current_month]} {current_year}'

    # Dictionary to store file paths and figures
    result = {'paths': {}, 'figures': {}}

    # Find week boundaries based on custom week definition
    week_boundaries = []
    week_numbers = []
    year_week_pairs = []
    current_week = None
    current_year = None

    # First, collect all week boundaries and numbers
    for i, (_, row) in enumerate(pivot_df.iterrows()):
        year = row['CustomYear']
        week = row['CustomWeek']
        year_week = f"{year}-{week}"

        # Store the first occurrence of each week
        if current_week != week or current_year != year:
            if current_week is not None:
                # Week changed, add a boundary at the previous position
                week_boundaries.append(i - 0.5)  # Position between bars
                week_numbers.append(current_week)  # Store the week that just ended
                year_week_pairs.append((current_year, current_week))
            current_week = week
            current_year = year

    # Add the last week number
    if pivot_df.shape[0] > 0:
        week_numbers.append(current_week)
        year_week_pairs.append((current_year, current_week))

    # Calculate midpoints between boundaries for week labels
    # Start with position 0 as the first boundary
    all_boundaries = [0] + week_boundaries + [len(pivot_df)]  # Add start and end positions

    # Calculate total hours by Day and Night for summary box
    total_day = pivot_df['D'].sum()
    total_night = pivot_df['N'].sum()
    total_combined = total_day + total_night

    # Create a more visually appealing total summary
    total_text = f'Horas Totales:  Día: {total_day:.1f}h  |  Noche: {total_night:.1f}h  |  Total: {total_combined:.1f}h'

    # 1. Main plot - Análisis de Tiempo de Mecanizado
    fig1, ax1 = plt.subplots(figsize=(16, 10), facecolor='#f8f9fa')
    ax1.set_facecolor('#f8f9fa')  # Light background
    bar_width = 0.35

    # Plot bars with gradient colors
    day_bars = ax1.bar(x - bar_width/2, pivot_df['D'], bar_width,
                      label='Turno Día', color=day_color, alpha=0.85,
                      edgecolor='white', linewidth=1)
    night_bars = ax1.bar(x + bar_width/2, pivot_df['N'], bar_width,
                       label='Turno Noche', color=night_color, alpha=0.85,
                       edgecolor='white', linewidth=1)

    # Add value labels on top of bars
    for bar in day_bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.1f}h', ha='center', va='bottom',
                   fontsize=8, fontweight='bold', color='black',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='none'))

    for bar in night_bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.1f}h', ha='center', va='bottom',
                   fontsize=8, fontweight='bold', color='black',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='none'))

    # Customize plot
    ax1.set_title(title_text, pad=20, fontsize=16, fontweight='bold')
    ax1.set_xlabel('Fecha', fontsize=12)
    ax1.set_ylabel('Horas', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(date_labels, rotation=45, ha='right')

    # Add a subtle grid
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Draw vertical lines at week boundaries and add week labels
    for i, boundary in enumerate(week_boundaries):
        # Draw the vertical line with enhanced style
        ax1.axvline(x=boundary, color=week_color, linestyle='-', alpha=0.7, linewidth=1.5)

        # Add a note that this is Sunday 20:00 with enhanced style
        ax1.text(boundary + 0.1, ax1.get_ylim()[1] * 0.75, 'Domingo 20:00',
                ha='left', va='center', fontsize=8, color=week_color,
                rotation=90, alpha=0.9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', edgecolor=week_color))

    # Add alternating background colors for weeks
    for i in range(len(all_boundaries)-1):
        start = all_boundaries[i]
        end = all_boundaries[i+1]
        if i % 2 == 0:  # Alternate colors
            ax1.axvspan(start, end, alpha=0.05, color='gray', zorder=0)

    # Add week labels at the center of each week section
    for i in range(len(all_boundaries)-1):
        # Calculate midpoint between boundaries
        midpoint = (all_boundaries[i] + all_boundaries[i+1]) / 2

        # Get the week number
        if i < len(week_numbers):
            week_num = week_numbers[i]
            year_val = year_week_pairs[i][0]

            # Add week label at the top of the plot with enhanced style
            y_pos = ax1.get_ylim()[1] * 0.95  # Position near the top
            ax1.text(midpoint, y_pos, f'Semana {week_num}/{str(year_val)[-2:]}',
                    ha='center', va='top', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3',
                             edgecolor=week_color, linewidth=1.5))

            # Calculate totals for this week
            week_data = pivot_df[(pivot_df['CustomWeek'] == week_num) & (pivot_df['CustomYear'] == year_val)]
            week_day_total = week_data['D'].sum()
            week_night_total = week_data['N'].sum()
            week_total = week_day_total + week_night_total

            # Add totals below the week label with enhanced style
            y_pos = ax1.get_ylim()[1] * 0.85  # Position below the week label

            # Determine which shift had more hours
            diff = abs(week_day_total - week_night_total)
            if week_day_total > week_night_total:
                comparison = f">: Día por {diff:.1f}h"
                comp_color = day_color
            elif week_night_total > week_day_total:
                comparison = f">: Noche por {diff:.1f}h"
                comp_color = night_color
            else:
                comparison = "Día = Noche"
                comp_color = 'black'

            # Create a more visually appealing summary box
            summary_text = f'Día: {week_day_total:.1f}h\nNoche: {week_night_total:.1f}h\nTotal: {week_total:.1f}h\n{comparison}'
            ax1.text(midpoint, y_pos, summary_text,
                    ha='center', va='top', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.3',
                             edgecolor=comp_color, linewidth=1))

    # Enhanced legend with custom styling
    legend = ax1.legend([day_bars[0], night_bars[0]], ['Turno Día', 'Turno Noche'],
              loc='upper right', fontsize=10, framealpha=0.9,
              edgecolor='gray', fancybox=True)
    legend.get_frame().set_linewidth(0.5)

    # Create a comprehensive statistics text box
    stats_text = (
        f'Resumen de Horas\n'
        f'───────────────────\n'
        f'Día: {total_day:.1f}h ({total_day/total_combined*100:.1f}%)\n'
        f'Noche: {total_night:.1f}h ({total_night/total_combined*100:.1f}%)\n'
        f'Total: {total_combined:.1f}h\n'
        f'Días Analizados: {len(pivot_df)}'
    )

    # Add the statistics text box with styling
    stats_box = ax1.text(0.02, 0.97, stats_text,
                       transform=ax1.transAxes,
                       verticalalignment='top',
                       horizontalalignment='left',
                       bbox=dict(
                           boxstyle="round,pad=0.6",
                           facecolor='white',
                           edgecolor='gray',
                           linewidth=1,
                           alpha=0.95
                       ),
                       fontsize=11,
                       color='black',
                       fontfamily='monospace')

    plt.tight_layout()

    # Save the first plot
    plot1_path = os.path.join(save_dir, f'analisis_tiempo_{year}_{month}.png')
    fig1.savefig(plot1_path, dpi=300, bbox_inches='tight')
    result['paths']['analisis_tiempo'] = plot1_path
    result['figures']['analisis_tiempo'] = fig1

    # Display the plot if requested
    if display:
        plt.show()
    else:
        plt.close(fig1)

    # 2. Weekly averages plot
    fig2, ax2 = plt.subplots(figsize=(14, 8), facecolor='#f8f9fa')
    ax2.set_facecolor('#f8f9fa')  # Light background

    # Create labels for weeks that include year
    weekly_avg['WeekLabel'] = weekly_avg.apply(lambda x: f"S{int(x['Week'])}/{str(int(x['Year']))[-2:]}", axis=1)

    # Use integer positions for x-axis
    x_pos = np.arange(len(weekly_avg))

    # Enhanced stacked bars with gradient effect
    day_bars_weekly = ax2.bar(x_pos, weekly_avg['D'], color=day_color, alpha=0.85,
           edgecolor='white', linewidth=1, label='Día')
    night_bars_weekly = ax2.bar(x_pos, weekly_avg['N'], bottom=weekly_avg['D'],
           color=night_color, alpha=0.85, edgecolor='white', linewidth=1, label='Noche')

    # Add labels for day, night and total hours with enhanced styling
    for i, row in weekly_avg.iterrows():
        # Label for day shift (in the middle of day bar)
        if row['D'] > 0:
            ax2.text(i, row['D']/2,
                    f'D: {row["D"]:.1f}h',
                    ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')

        # Label for night shift (in the middle of night bar)
        if row['N'] > 0:
            ax2.text(i, row['D'] + row['N']/2,
                    f'N: {row["N"]:.1f}h',
                    ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')

        # Label for total (on top of stacked bar) with enhanced styling
        total = row['Total']
        if total > 0:
            ax2.text(i, row['D'] + row['N'] + 0.1,
                    f'Total: {total:.1f}h',
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2', edgecolor='none'))

    # Add alternating background for better readability
    for i in range(len(weekly_avg)):
        if i % 2 == 0:  # Alternate colors
            ax2.axvspan(i-0.4, i+0.4, alpha=0.05, color='gray', zorder=0)

    ax2.set_title('Promedio Semanal de Horas (Semanas: Domingo 20:00 a Domingo 19:59)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Semana del Año', fontsize=12)
    ax2.set_ylabel('Horas Promedio', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(weekly_avg['WeekLabel'])
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Enhanced legend
    legend2 = ax2.legend(title='Turnos', title_fontsize=10, framealpha=0.9,
                        edgecolor='gray', fancybox=True)
    legend2.get_frame().set_linewidth(0.5)

    # Create a comprehensive statistics text box for weekly data
    weekly_stats_text = (
        f'Resumen Semanal\n'
        f'───────────────────\n'
        f'Promedio Día: {weekly_avg["D"].mean():.1f}h\n'
        f'Promedio Noche: {weekly_avg["N"].mean():.1f}h\n'
        f'Promedio Total: {weekly_avg["Total"].mean():.1f}h\n'
        f'Semanas Analizadas: {len(weekly_avg)}'
    )

    # Add the statistics text box with styling
    weekly_stats_box = ax2.text(0.02, 0.97, weekly_stats_text,
                       transform=ax2.transAxes,
                       verticalalignment='top',
                       horizontalalignment='left',
                       bbox=dict(
                           boxstyle="round,pad=0.6",
                           facecolor='white',
                           edgecolor='gray',
                           linewidth=1,
                           alpha=0.95
                       ),
                       fontsize=11,
                       color='black',
                       fontfamily='monospace')

    plt.tight_layout()

    # Save the second plot
    plot2_path = os.path.join(save_dir, f'promedio_semanal_{year}_{month}.png')
    fig2.savefig(plot2_path, dpi=300, bbox_inches='tight')
    result['paths']['promedio_semanal'] = plot2_path
    result['figures']['promedio_semanal'] = fig2

    # Display the plot if requested
    if display:
        plt.show()
    else:
        plt.close(fig2)

    # No separate summary plot needed

    print(f"Gráficos guardados en la carpeta: {save_dir}")
    return result