from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os

def plot_tiempo_range(df, start_year, start_month, end_year, end_month, save_dir="data_planta", display=True):
    """
    Crear visualizaciones de tiempo de mecanizado y guardarlas como archivos separados.
    Incluye 'Analisis de tiempo de mecanizado' y 'Promedio semanal de horas'.
    Filtra datos entre las fechas especificadas (ambas inclusivas).
    Las semanas comienzan los domingos a las 20:00.

    Args:
        df: DataFrame con los datos
        start_year: Año de inicio del filtro
        start_month: Mes de inicio del filtro
        end_year: Año de fin del filtro
        end_month: Mes de fin del filtro
        save_dir: Directorio donde guardar las imágenes
        display: Si es True, muestra los gráficos en el notebook además de guardarlos

    Returns:
        dict: Rutas a los archivos guardados y figuras si display=True
    """
    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define Harvard color palette
    harvard_palette = {
        'crimson': '#A51C30',       # Harvard Crimson
        'slate': '#8996A0',         # Harvard Slate
        'blue': '#4E84C4',          # Harvard Blue
        'ivy': '#52854C',           # Harvard Ivy
        'gold': '#C4961A',          # Harvard Gold
        'black': '#1E1E1E',         # Rich Black
        'gray_dark': '#4A4A4A',     # Dark Gray
        'gray_medium': '#767676',   # Medium Gray
        'gray_light': '#D5D5D5',    # Light Gray
        'white': '#FFFFFF',         # White
        'background': '#F8F8F8',    # Light Background
    }

    # Reset matplotlib to default settings
    plt.rcdefaults()

    # Set up Harvard-style visualization parameters
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Palatino', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.figsize': (12, 10),
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.color': '#E5E5E5',
        'grid.linestyle': ':',
        'grid.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.major.size': 5.0,
        'ytick.major.size': 5.0,
        'xtick.minor.size': 3.0,
        'ytick.minor.size': 3.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.axisbelow': True,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
    })


    # Convert dates to datetime for proper comparison
    df['DateObj'] = pd.to_datetime(df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1))
    df['DateTime'] = pd.to_datetime(df['Timestamp'])

    # Create start and end date filters
    start_date = pd.to_datetime(f"{start_year}-{start_month}-01")

    # For end date, get the last day of the month and include the full day
    if end_month == 12:
        next_month_year = end_year + 1
        next_month = 1
    else:
        next_month_year = end_year
        next_month = end_month + 1

    # Get the last day of end_month
    end_date = pd.to_datetime(f"{next_month_year}-{next_month}-01") - timedelta(days=1)
    end_date = end_date.replace(hour=23, minute=59, second=59)

    # Filter data between specified dates (inclusive)
    filtered_df = df[(df['DateObj'] >= start_date) & (df['DateObj'] <= end_date)].copy()

    if filtered_df.empty:
        print(f"No data found between {start_year}-{start_month} and {end_year}-{end_month}")
        return None

    # Process data
    filtered_df['Date'] = filtered_df['DateObj']
    filtered_df['Hour'] = filtered_df['DateTime'].dt.hour
    filtered_df['DayOfWeek'] = filtered_df['DateTime'].dt.dayofweek

    # Define custom week based on Sunday 20:00
    def get_custom_week(row):
        date = row['DateTime']
        iso_year, iso_week, _ = date.isocalendar()

        # If it's Sunday (6) and hour >= 20, it belongs to the next week
        if row['DayOfWeek'] == 6 and row['Hour'] >= 20:
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

    # Merge custom week information
    date_to_week = filtered_df.groupby('Date')[['CustomWeek', 'CustomYear']].first()
    pivot_df = pivot_df.merge(date_to_week, on='Date', how='left')

    # Calculate weekly averages
    weekly_avg = pivot_df.groupby(['CustomYear', 'CustomWeek'])[['D', 'N', 'Total']].mean().reset_index()
    weekly_avg.rename(columns={'CustomYear': 'Year', 'CustomWeek': 'Week'}, inplace=True)

    # Use Harvard colors for better aesthetics
    day_color = harvard_palette['blue']      # Harvard blue
    night_color = harvard_palette['crimson'] # Harvard crimson
    week_color = harvard_palette['ivy']      # Harvard ivy

    # Common x-axis data
    x = np.arange(len(pivot_df['Date']))
    date_labels = [d.strftime('%d-%m-%y') for d in pivot_df['Date']]  # Include month and year in labels

    # Spanish month names
    month_names = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }

    # Get the range of dates for the title
    start_month_name = month_names[start_month]
    end_month_name = month_names[end_month]

    if start_year == end_year:
        title_text = f'Análisis de Tiempo de Mecanizado - {start_month_name} a {end_month_name} {start_year}'
    else:
        title_text = f'Análisis de Tiempo de Mecanizado - {start_month_name} {start_year} a {end_month_name} {end_year}'

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

    # Enhanced legend with Harvard styling
    legend = ax1.legend([day_bars[0], night_bars[0]], ['Turno Día', 'Turno Noche'],
              loc='upper right', fontsize=11, framealpha=0.95,
              edgecolor=harvard_palette['slate'], facecolor=harvard_palette['white'],
              fancybox=True)
    legend.get_frame().set_linewidth(1.0)

    # Add shadow effect to the legend
    legend.get_frame().set_path_effects([
        path_effects.SimplePatchShadow(offset=(2, -2), alpha=0.2),
        path_effects.Normal()
    ])

    # Calculate total hours by Day and Night
    total_day = pivot_df['D'].sum()
    total_night = pivot_df['N'].sum()
    total_combined = total_day + total_night

    # Create a comprehensive statistics text box
    stats_text = (
        f'Resumen de Horas\n'
        f'------------------\n'
        f'Día: {total_day:.1f}h ({total_day/total_combined*100:.1f}%)\n'
        f'Noche: {total_night:.1f}h ({total_night/total_combined*100:.1f}%)\n'
        f'Total: {total_combined:.1f}h\n'
        f'Días Analizados: {len(pivot_df)}'
    )

    # Add the statistics text box with Harvard styling
    stats_box = ax1.text(0.02, 0.97, stats_text,
                       transform=ax1.transAxes,
                       verticalalignment='top',
                       horizontalalignment='left',
                       bbox=dict(
                           boxstyle="round,pad=0.6",
                           facecolor=harvard_palette['white'],
                           edgecolor=harvard_palette['slate'],
                           linewidth=1,
                           alpha=0.95
                       ),
                       fontsize=11,
                       color=harvard_palette['black'],
                       fontfamily='monospace')

    # Add shadow effect to the stats box
    stats_box.set_path_effects([
        path_effects.SimplePatchShadow(offset=(2, -2), alpha=0.2),
        path_effects.Normal()
    ])

    plt.tight_layout()

    # Save the first plot
    plot1_path = os.path.join(save_dir, f'analisis_tiempo_{start_year}_{start_month}_a_{end_year}_{end_month}.png')
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
        if i % 2 == 0:
            ax2.axvspan(i-0.4, i+0.4, alpha=0.05, color='gray', zorder=0)

    ax2.set_title('Promedio Semanal de Horas (Semanas: Domingo 20:00 a Domingo 19:59)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Semana del Año', fontsize=12)
    ax2.set_ylabel('Horas Promedio', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(weekly_avg['WeekLabel'])
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Enhanced legend with Harvard styling
    legend2 = ax2.legend(
        loc='upper right',
        fontsize=11,
        framealpha=0.95,
        edgecolor=harvard_palette['slate'],
        facecolor=harvard_palette['white'],
        fancybox=True,
        title='Turnos',
        title_fontsize=12
    )
    legend2.get_frame().set_linewidth(1.0)

    # Add shadow effect to the legend
    legend2.get_frame().set_path_effects([
        # path_effects.SimplePatchShadow(offset=(2, -2), alpha=0.2),
        path_effects.Normal()
    ])

    # Create a comprehensive statistics text box for weekly data
    weekly_stats_text = (
        f'Resumen Semanal\n'
        f'------------------\n'
        f'Promedio Día: {weekly_avg["D"].mean():.1f}h\n'
        f'Promedio Noche: {weekly_avg["N"].mean():.1f}h\n'
        f'Promedio Total: {weekly_avg["Total"].mean():.1f}h\n'
        f'Semanas Analizadas: {len(weekly_avg)}'
    )

    # Add the statistics text box with Harvard styling
    weekly_stats_box = ax2.text(0.02, 0.97, weekly_stats_text,
                       transform=ax2.transAxes,
                       verticalalignment='top',
                       horizontalalignment='left',
                       bbox=dict(
                           boxstyle="round,pad=0.6",
                           facecolor=harvard_palette['white'],
                           edgecolor=harvard_palette['slate'],
                           linewidth=1,
                           alpha=0.95
                       ),
                       fontsize=11,
                       color=harvard_palette['black'],
                       fontfamily='monospace')

    # Add shadow effect to the stats box
    weekly_stats_box.set_path_effects([
        # path_effects.SimplePatchShadow(offset=(2, -2), alpha=0.2),
        path_effects.Normal()
    ])

    plt.tight_layout()

    # Save the second plot
    plot2_path = os.path.join(save_dir, f'promedio_semanal_{start_year}_{start_month}_a_{end_year}_{end_month}.png')
    fig2.savefig(plot2_path, dpi=300, bbox_inches='tight')
    result['paths']['promedio_semanal'] = plot2_path
    result['figures']['promedio_semanal'] = fig2

    # Display the plot if requested
    if display:
        plt.show()
    else:
        plt.close(fig2)

    # 3. Total hours summary plot with Harvard styling
    fig3 = plt.figure(figsize=(10, 2), dpi=150, facecolor=harvard_palette['white'])
    ax3 = fig3.add_subplot(111)
    ax3.set_facecolor(harvard_palette['background'])

    # Calculate total hours by Day and Night (already calculated above)
    # Create a more detailed and visually appealing total summary
    total_text = (
        f'Resumen Total de Horas\n'
        f'----------------------------------\n'
        f'Día: {total_day:.1f}h ({total_day/total_combined*100:.1f}%)  |  '
        f'Noche: {total_night:.1f}h ({total_night/total_combined*100:.1f}%)  |  '
        f'Total: {total_combined:.1f}h\n'
        f'Período: {start_month_name} {start_year} a {end_month_name} {end_year}  |  '
        f'Días Analizados: {len(pivot_df)}'
    )

    # Remove axes and add text with enhanced Harvard styling
    ax3.axis('off')
    summary_box = ax3.text(0.5, 0.5, total_text,
             ha='center', va='center',
             fontsize=14,
             fontweight='bold',
             transform=ax3.transAxes,
             bbox=dict(
                 facecolor=harvard_palette['white'],
                 alpha=0.95,
                 boxstyle='round,pad=0.8',
                 edgecolor=harvard_palette['slate'],
                 linewidth=2
             ),
             color=harvard_palette['black'],
             fontfamily='serif')

    # Add shadow effect to the summary box
    summary_box.set_path_effects([
        # path_effects.SimplePatchShadow(offset=(1, -1), alpha=0.2),
        path_effects.Normal()
    ])

    # Save the third plot
    plot3_path = os.path.join(save_dir, f'horas_totales_{start_year}_{start_month}_a_{end_year}_{end_month}.png')
    fig3.savefig(plot3_path, dpi=300, bbox_inches='tight')
    result['paths']['horas_totales'] = plot3_path
    result['figures']['horas_totales'] = fig3

    # Display the plot if requested
    if display:
        plt.show()
    else:
        plt.close(fig3)

    print(f"Gráficos guardados en la carpeta: {save_dir}")
    return result