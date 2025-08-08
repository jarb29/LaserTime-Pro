import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns


def plot_program_comparison(program_summary_df, top_n=15):
    """
    Create a plot comparing program machining time vs real time.

    This function generates a stacked bar chart that compares the theoretical machining time
    with the actual real time taken for each program. It highlights the time difference and
    provides statistical information about the average time difference.

    Parameters:
    ----------
    program_summary_df : pandas.DataFrame
        DataFrame with program summary data. Must contain the following columns:
        - 'Programa': Program identifier
        - 'Tiempo': Theoretical machining time in seconds
        - 'Tiempo_Real': Actual real time in hours

    top_n : int, optional
        Number of top programs to display, sorted by machining time (default: 15)

    Returns:
    -------
    matplotlib.figure.Figure
        Matplotlib figure object containing the visualization

    Notes:
    -----
    The function filters out programs with time differences greater than 20 hours
    to prevent extreme outliers from distorting the visualization.
    """
    import matplotlib.patheffects as path_effects
    from matplotlib.ticker import AutoMinorLocator
    from matplotlib.colors import LinearSegmentedColormap

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

    # Create a copy to avoid modifying the original
    df = program_summary_df.copy()

    # Calculate program totals
    program_totals = df.groupby('Programa').agg({
        'Tiempo': 'sum',
        'Tiempo_Real': 'sum'
    }).reset_index()

    # Convert to hours for better readability
    program_totals['Tiempo_Hours'] = program_totals['Tiempo'] / 3600
    program_totals['Tiempo_Real_Hours'] = program_totals['Tiempo_Real']
    program_totals['Time_Difference'] = program_totals['Tiempo_Real_Hours'] - program_totals['Tiempo_Hours']
    program_totals['Time_Difference_Pct'] = (program_totals['Time_Difference'] / program_totals['Tiempo_Real_Hours']) * 100

    # Filter out rows where Time_Difference > 20 hours
    program_totals = program_totals[program_totals['Time_Difference'] <= 20]

    # Sort by machining time and get top programs
    top_programs = program_totals.sort_values('Tiempo_Hours', ascending=False).head(top_n)

    # Create a new figure with a single subplot
    fig = plt.figure(figsize=(14, 10), dpi=150, facecolor=harvard_palette['white'])
    ax = fig.add_subplot(111)

    # Apply Harvard styling to the axis
    ax.tick_params(colors=harvard_palette['gray_dark'], which='both')
    ax.spines['left'].set_color(harvard_palette['gray_dark'])
    ax.spines['bottom'].set_color(harvard_palette['gray_dark'])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_facecolor(harvard_palette['background'])
    ax.grid(True, linestyle=':', linewidth=0.8, color=harvard_palette['gray_light'], alpha=0.7)

    # Set up x-axis
    x = np.arange(len(top_programs))

    # Create custom colormap based on Harvard colors
    cmap_blue = LinearSegmentedColormap.from_list('harvard_blues',
                                           [harvard_palette['slate'], harvard_palette['blue']],
                                           N=256)
    cmap_red = LinearSegmentedColormap.from_list('harvard_reds',
                                          [harvard_palette['slate'], harvard_palette['crimson']],
                                          N=256)

    # Create stacked bar chart with enhanced styling
    bars1 = ax.bar(x, top_programs['Tiempo_Hours'],
                  label='Tiempo de Mecanizado',
                  color=harvard_palette['blue'],
                  edgecolor=harvard_palette['white'],
                  linewidth=1.2,
                  alpha=0.9)

    bars2 = ax.bar(x, top_programs['Time_Difference'],
                  bottom=top_programs['Tiempo_Hours'],
                  label='Diferencia de Tiempo',
                  color=harvard_palette['crimson'],
                  edgecolor=harvard_palette['white'],
                  linewidth=1.2,
                  alpha=0.8)

    # Add shadow effect to bars
    for bar in bars1:
        bar.set_path_effects([
            path_effects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
            path_effects.Normal()
        ])

    for bar in bars2:
        bar.set_path_effects([
            path_effects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
            path_effects.Normal()
        ])

    # Calculate average time difference
    avg_time_diff = top_programs['Time_Difference'].mean()
    median_time_diff = top_programs['Time_Difference'].median()
    std_dev = top_programs['Time_Difference'].std()

    # Create a line for the legend that represents the average difference
    avg_line = ax.plot([], [],
                color=harvard_palette['crimson'],
                linestyle='--',
                linewidth=2,
                alpha=0.9,
                label=f'Dif. Promedio: +{avg_time_diff:.1f}h')[0]

    # Add shadow effect to the legend line
    avg_line.set_path_effects([
        path_effects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
        path_effects.Normal()
    ])

    # Add average line for time difference (keep the dot line feature)
    for i in range(len(top_programs)):
        line = ax.plot([i-0.4, i+0.4], [top_programs['Tiempo_Hours'].iloc[i] + avg_time_diff] * 2,
                color=harvard_palette['crimson'], linestyle='--', linewidth=2, alpha=0.9)

        # Add shadow effect to the line
        for l in line:
            l.set_path_effects([
                path_effects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
                path_effects.Normal()
            ])

    # Add value labels with better positioning and enhanced styling
    for i, (tiempo, tiempo_real) in enumerate(zip(top_programs['Tiempo_Hours'], top_programs['Tiempo_Real_Hours'])):
        # Label for machining time
        tiempo_label = ax.text(i, tiempo/2, f'{tiempo:.1f}h',
                ha='center', va='center',
                color=harvard_palette['white'],
                fontweight='bold',
                fontsize=11)

        tiempo_label.set_path_effects([
            path_effects.withStroke(linewidth=3, foreground=harvard_palette['blue']),
            path_effects.Normal()
        ])

        # Label for total real time at the top with enhanced styling
        total_label = ax.text(i, tiempo_real + 0.2, f'{tiempo_real:.1f}h',
                ha='center', va='bottom',
                color=harvard_palette['black'],
                fontweight='bold',
                fontsize=11,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=harvard_palette['white'],
                    edgecolor=harvard_palette['slate'],
                    linewidth=1,
                    alpha=0.9
                ))

        total_label.set_path_effects([
            path_effects.SimplePatchShadow(offset=(1, -1), alpha=0.3),
            path_effects.Normal()
        ])

        # Label for difference with enhanced styling
        diff = tiempo_real - tiempo
        if diff > 0.5:  # Only show if significant
            diff_label = ax.text(i, tiempo + diff/2, f'+{diff:.1f}h',
                    ha='center', va='center',
                    color=harvard_palette['white'],
                    fontweight='bold',
                    fontsize=11)

            diff_label.set_path_effects([
                path_effects.withStroke(linewidth=3, foreground=harvard_palette['crimson']),
                path_effects.Normal()
            ])

    # Create a comprehensive statistics text box
    stats_text = (
        f'Estadísticas de Diferencia de Tiempo\n'
        f'───────────────────────────────\n'
        f'Media: +{avg_time_diff:.2f}h\n'
        f'Mediana: +{median_time_diff:.2f}h\n'
        f'Desv. Est.: {std_dev:.2f}h\n'
        f'Mín: +{top_programs["Time_Difference"].min():.2f}h\n'
        f'Máx: +{top_programs["Time_Difference"].max():.2f}h\n'
        f'Programas Analizados: {len(top_programs)}'
    )

    # Add the statistics text box with Harvard styling
    stats_box = ax.text(0.02, 0.97, stats_text,
                       transform=ax.transAxes,
                       verticalalignment='top',
                       horizontalalignment='left',
                       bbox=dict(
                           boxstyle="round,pad=0.6",
                           facecolor=harvard_palette['white'],
                           edgecolor=harvard_palette['slate'],
                           linewidth=1,
                           alpha=0.95
                       ),
                       fontsize=12,
                       color=harvard_palette['black'],
                       fontfamily='monospace')

    # Add shadow effect to the stats box
    stats_box.set_path_effects([
        path_effects.SimplePatchShadow(offset=(2, -2), alpha=0.2),
        path_effects.Normal()
    ])

    # Customize plot with Harvard-style labels
    title = ax.set_title('Tiempo de Mecanizado vs Tiempo Real por Programa',
                      fontsize=16, fontweight='bold',
                      color=harvard_palette['black'],
                      pad=20)

    # Add subtitle
    ax.text(0.5, 1.05, f'Análisis de los {top_n} Programas Principales',
           transform=ax.transAxes,
           ha='center', va='bottom',
           fontsize=14, fontstyle='italic',
           color=harvard_palette['gray_dark'])

    # Add shadow effect to title
    title.set_path_effects([
        path_effects.SimplePatchShadow(offset=(2, -2), alpha=0.2),
        path_effects.Normal()
    ])

    ax.set_xlabel('Programa', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Horas', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(top_programs['Programa'], rotation=45, ha='right')

    # Create a more stylish legend with Harvard styling
    legend = ax.legend(
        loc='upper right',
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor=harvard_palette['slate'],
        facecolor=harvard_palette['white'],
        fontsize=11
    )

    # Add shadow effect to the legend
    legend.get_frame().set_path_effects([
        path_effects.SimplePatchShadow(offset=(2, -2), alpha=0.2),
        path_effects.Normal()
    ])


    # Adjust layout to make sure everything fits
    plt.tight_layout()

    return fig


def plot_downtime_analysis(df, max_downtime_hours=24, outlier_threshold=None):
    """
    Create a plot analyzing downtime by shift and day.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame with downtime data containing 'Date', 'Tiempo_detenido',
        'Tiempo_detenido_D', and 'Tiempo_detenido_N' columns.
    max_downtime_hours : float, optional
        Maximum downtime hours to display. Default is 24.
    outlier_threshold : float, optional
        Threshold for filtering outliers. If None, no filtering is applied.

    Returns:
    -------
    matplotlib.figure.Figure
        Matplotlib figure object.
    """
    # Reset matplotlib to default settings
    plt.rcdefaults()

    # Set style with minimal customization
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10})

    # Create a copy to avoid modifying the original
    df_copy = df.copy()

    # Group by date and calculate daily totals
    daily_totals = df_copy.groupby('Date').agg({
        'Tiempo_detenido': 'sum',
        'Tiempo_detenido_D': 'sum',
        'Tiempo_detenido_N': 'sum'
    }).reset_index()

    # Filter out extreme outliers if threshold is provided
    if outlier_threshold is not None:
        daily_totals_filtered = daily_totals[daily_totals['Tiempo_detenido'] <= outlier_threshold]
    else:
        daily_totals_filtered = daily_totals.copy()

    # Create a new figure with a single subplot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create a twin axis for the total line
    ax_twin = ax.twinx()

    # Set up x-axis with dates
    x = np.arange(len(daily_totals))

    # Define colors
    day_color = '#3498db'    # Blue
    night_color = '#2c3e50'  # Dark blue
    total_color = '#e74c3c'  # Red

    # Create stacked bar chart for day and night shifts
    ax.bar(x, daily_totals['Tiempo_detenido_D'], label='Turno Día', color=day_color, alpha=0.7)
    ax.bar(x, daily_totals['Tiempo_detenido_N'], bottom=daily_totals['Tiempo_detenido_D'],
           label='Turno Noche', color=night_color, alpha=0.7)

    # Add total line
    ax_twin.plot(x, daily_totals['Tiempo_detenido'], label='Total', color=total_color,
                marker='o', linewidth=2, markersize=5)

    # Set labels and title with enhanced styling and better visibility
    ax.set_title('Análisis de Tiempo Detenido por Turno y Día', fontweight='bold', pad=25,
               color='#263238', fontsize=20)
    ax.set_ylabel('Horas Detenido por Turno', fontweight='bold', color='#263238', labelpad=15)
    ax_twin.set_ylabel('Horas Detenido Total', color=total_color, fontweight='bold', labelpad=15)

    # Set y-axis limits with more headroom and better tick formatting
    ax.set_ylim(0, max_downtime_hours * 0.95)
    ax_twin.set_ylim(0, max_downtime_hours * 1.15)

    # Format y-axis ticks with hour suffix for better readability
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}h'))
    ax_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}h'))

    # Set the color of the right y-axis ticks and label to match the total line
    ax_twin.tick_params(axis='y', colors=total_color)

    # Add grid with better styling for improved readability
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5, zorder=0)

    # Add statistics summary box with enhanced styling and better visibility
    total_hours = df_copy['Tiempo_detenido'].sum()
    avg_daily = daily_totals_filtered['Tiempo_detenido'].mean()

    # Use filtered data for max calculation to avoid extreme outliers
    if len(daily_totals_filtered) > 0:
        max_day = daily_totals_filtered['Tiempo_detenido'].max()
        max_day_date = daily_totals_filtered.loc[daily_totals_filtered['Tiempo_detenido'].idxmax(), 'Date']
    else:
        # Fallback if all days are filtered out
        max_day = daily_totals['Tiempo_detenido'].max()
        max_day_date = daily_totals.loc[daily_totals['Tiempo_detenido'].idxmax(), 'Date']

    day_percent = (df_copy['Tiempo_detenido_D'].sum() / total_hours * 100) if total_hours > 0 else 0
    night_percent = (df_copy['Tiempo_detenido_N'].sum() / total_hours * 100) if total_hours > 0 else 0
    excluded_days = len(daily_totals) - len(daily_totals_filtered)

    # Create a more structured and visually appealing stats box
    stats_title = "RESUMEN DE TIEMPO DETENIDO"
    stats_text = f"\nTotal: {total_hours:.1f}h\n"
    stats_text += f"Promedio Diario: {avg_daily:.1f}h\n"
    stats_text += f"Máximo Normal: {max_day:.1f}h ({max_day_date})\n\n"
    stats_text += f"Distribución por Turno:\n"
    stats_text += f"• Día: {day_percent:.1f}% ({df_copy['Tiempo_detenido_D'].sum():.1f}h)\n"
    stats_text += f"• Noche: {night_percent:.1f}% ({df_copy['Tiempo_detenido_N'].sum():.1f}h)"

    if excluded_days > 0:
        stats_text += f"\n\nNota: {excluded_days} día(s) excluido(s) del promedio"

    # Add enhanced text box with better styling
    props = dict(boxstyle='round4,pad=1.0', facecolor='white', alpha=0.97,
                edgecolor='#bdc3c7', linewidth=1.5)

    # Add title to the stats box
    ax.text(0.02, 0.97, stats_title, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', horizontalalignment='left',
           color='#263238', fontweight='bold', bbox=dict(facecolor='#f5f5f5',
                                                       alpha=0.97, edgecolor='#bdc3c7',
                                                       pad=0.5, boxstyle='round,pad=0.5'))

    # Add stats content
    ax.text(0.02, 0.93, stats_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props, color='#263238', zorder=10)

    # Create a more visually appealing legend with better styling
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_twin.get_legend_handles_labels()

    # Combine legends with better styling
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    # Create custom legend entries with colored markers
    legend = ax.legend(all_handles, all_labels, loc='upper right', frameon=True,
                      fancybox=True, shadow=True, fontsize=11, ncol=3,
                      title="LEYENDA", title_fontsize=12)
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('#bdc3c7')
    legend.get_frame().set_linewidth(1.5)

    # Add a subtle border to the plot for better definition
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#bdc3c7')
        spine.set_linewidth(0.8)

    # Fix the right y-axis to show the spine
    ax_twin.spines['right'].set_visible(True)
    ax_twin.spines['right'].set_color(total_color)
    ax_twin.spines['right'].set_linewidth(0.8)

    # Add a subtle box around the entire plot area
    fig.patch.set_linewidth(1)
    fig.patch.set_edgecolor('#e0e0e0')

    # Add a watermark or subtle footer
    fig.text(0.99, 0.01, 'Análisis de Tiempo Detenido',
             ha='right', va='bottom', color='#9e9e9e', fontsize=8, alpha=0.7)

    # Adjust layout with more padding
    plt.tight_layout(pad=2.0)

    return fig


def create_barplot(df, x_col, y_col, a, b, width, text_size, x_title, y_title, text_angle=45, text_color="black"):
    """
    Create a bar plot with customizable parameters.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    x_col : str
        Name of the column to use for x-axis
    y_col : str
        Name of the column to use for y-axis
    a : int
        Figure width
    b : int
        Figure height
    width : float
        Width of the bars
    text_size : int
        Size of the text labels
    x_title : str
        Title for x-axis
    y_title : str
        Title for y-axis
    text_angle : int, optional
        Angle for text labels (default: 45)
    text_color : str, optional
        Color for text labels (default: "black")

    Returns:
    --------
    None
    """
    import datetime

    # Sort dataframe
    df_sort = df.sort_values(y_col, ascending=True).reset_index(drop=True)

    plt.figure(figsize=(a, b))

    # Create vertical bar plot with specified bar width
    bars = plt.bar(df_sort[x_col], df_sort[y_col], width=width)

    # Add labels to each bar
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{bar.get_height():.2f}',
            ha='center',
            va='bottom',
            fontsize=text_size,
            rotation=text_angle,
            color=text_color
        )

    # Increase y limit to accommodate the text height
    if np.isfinite(df_sort[y_col].max()):
        plt.ylim(0, df_sort[y_col].max() + 5)  # change 5 to a larger number if necessary
    else:
        plt.ylim(0, 1)

    now = datetime.datetime.now()
    current_month = now.month

    plt.ylabel(y_title)
    plt.xlabel(x_title)
    plt.title(f'{x_title} by {y_title}')
    plt.xticks(rotation=45, horizontalalignment='right')

    # Create plots directory if it doesn't exist
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')

    plt.savefig(f'plots/tiempo_programa_{current_month}.png',
                bbox_inches='tight')
    plt.show()
