import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_downtime_analysis(program_summary_df, max_downtime_hours=24):
    """
    Create a plot showing downtime analysis by day and shift.

    This function generates a comprehensive visualization of machine downtime,
    broken down by day and shift (day shift vs night shift). It processes the input
    data to calculate shift-specific downtime and creates a stacked bar chart with
    additional statistical information.

    Parameters:
    ----------
    program_summary_df : pandas.DataFrame
        DataFrame with program summary data. Must contain the following columns:
        - 'Hora_Inicio': Datetime of program start
        - 'Hora_Final': Datetime of program end
        - 'Tiempo_detenido': Downtime in hours

    max_downtime_hours : float, optional
        Maximum downtime hours to display (default: 24). Values exceeding this
        threshold will be capped to prevent extreme outliers from distorting
        the visualization.

    Returns:
    -------
    matplotlib.figure.Figure
        Matplotlib figure object containing the visualization

    Notes:
    -----
    - Shifts are defined as follows:
      - Day shift: 7:00 AM to 8:00 PM
      - Night shift: 8:00 PM to 7:00 AM
    - For programs that span both shifts, downtime is proportionally allocated
      based on the duration in each shift.
    - Weekend patterns are highlighted in the visualization.
    """
    # Set a modern, high-contrast style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.titlesize': 18,
        'axes.labelsize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.spines.top': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3
    })

    # Create a copy to avoid modifying the original
    df = program_summary_df.copy()

    # Extract date from timestamps
    df['Date'] = df['Hora_Inicio'].dt.date

    # Add day of week for week boundary detection
    df['DayOfWeek'] = df['Hora_Inicio'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['DayName'] = df['Hora_Inicio'].dt.day_name()

    # Determine shift based on custom shift definition (20:00 Sunday start)
    # Day shift: 7 AM to 8 PM, Night shift: otherwise
    df['Start_Shift'] = df['Hora_Inicio'].apply(lambda x: 'D' if 7 <= x.hour < 20 else 'N')
    df['End_Shift'] = df['Hora_Final'].apply(lambda x: 'D' if 7 <= x.hour < 20 else 'N')
    df['Shift'] = 'Mixed'
    df.loc[df['Start_Shift'] == df['End_Shift'], 'Shift'] = df['Start_Shift']

    # Initialize shift-specific tiempo columns
    df['Tiempo_detenido_D'] = 0.0
    df['Tiempo_detenido_N'] = 0.0

    # Process mixed shift programs (programs that span both day and night shifts)
    mixed_programs = df[df['Shift'] == 'Mixed']
    for idx in mixed_programs.index:
        start_time = df.loc[idx, 'Hora_Inicio']
        end_time = df.loc[idx, 'Hora_Final']

        # Find shift boundary (the time when shift changes)
        if df.loc[idx, 'Start_Shift'] == 'D':
            # If program starts in day shift, boundary is at 8:00 PM (20:00)
            boundary = pd.Timestamp(start_time.date()) + pd.Timedelta(hours=20)
        else:
            # If program starts in night shift, boundary is at 7:00 AM (07:00)
            boundary = pd.Timestamp(start_time.date()) + pd.Timedelta(hours=7)
            # If the 7:00 AM boundary is earlier than the start time,
            # it means we need to look at the next day's 7:00 AM
            if boundary < start_time:
                boundary = pd.Timestamp(start_time.date() + pd.Timedelta(days=1)) + pd.Timedelta(hours=7)

        # Calculate proportions of time spent in each shift
        total_duration = (end_time - start_time).total_seconds()
        first_shift_duration = (boundary - start_time).total_seconds()
        # Calculate what proportion of the program's time was in the first shift
        first_shift_proportion = first_shift_duration / total_duration

        # Assign proportional tiempo_detenido (downtime) to each shift
        # This allocates downtime proportionally to the time spent in each shift
        tiempo_detenido = float(df.loc[idx, 'Tiempo_detenido'])
        # Allocate to first shift (either D or N)
        df.at[idx, 'Tiempo_detenido_' + df.loc[idx, 'Start_Shift']] = tiempo_detenido * first_shift_proportion
        # Allocate to second shift (either N or D)
        df.at[idx, 'Tiempo_detenido_' + df.loc[idx, 'End_Shift']] = tiempo_detenido * (1 - first_shift_proportion)

    # Process non-mixed programs
    day_programs = df[df['Shift'] == 'D']
    for idx in day_programs.index:
        df.at[idx, 'Tiempo_detenido_D'] = float(df.loc[idx, 'Tiempo_detenido'])

    night_programs = df[df['Shift'] == 'N']
    for idx in night_programs.index:
        df.at[idx, 'Tiempo_detenido_N'] = float(df.loc[idx, 'Tiempo_detenido'])

    # Calculate daily totals
    daily_totals = df.groupby('Date').agg({
        'Tiempo_detenido': 'sum',
        'Tiempo_detenido_D': 'sum',
        'Tiempo_detenido_N': 'sum',
        'DayOfWeek': 'first',
        'DayName': 'first'
    }).reset_index()

    # Cap values at max_downtime_hours to prevent extreme values from distorting the visualization
    daily_totals['Tiempo_detenido'] = daily_totals['Tiempo_detenido'].clip(upper=max_downtime_hours)

    # If the sum of day and night exceeds max_downtime_hours, scale them proportionally
    for idx in daily_totals.index:
        total = daily_totals.loc[idx, 'Tiempo_detenido_D'] + daily_totals.loc[idx, 'Tiempo_detenido_N']
        if total > max_downtime_hours:
            scale_factor = max_downtime_hours / total
            daily_totals.loc[idx, 'Tiempo_detenido_D'] *= scale_factor
            daily_totals.loc[idx, 'Tiempo_detenido_N'] *= scale_factor

    # Sort by date
    daily_totals = daily_totals.sort_values('Date')
    daily_totals['DateDT'] = pd.to_datetime(daily_totals['Date'])

    # Find the first Sunday in the data to use as reference
    sundays = daily_totals[daily_totals['DayOfWeek'] == 6]['DateDT']
    if len(sundays) > 0:
        first_sunday = sundays.min()
        # Calculate weeks since first Sunday
        daily_totals['CustomWeek'] = ((daily_totals['DateDT'] - first_sunday).dt.days // 7) + 1
    else:
        # Fallback if no Sundays in data
        daily_totals['CustomWeek'] = daily_totals['DateDT'].dt.isocalendar().week

    # Create a filtered version that excludes days with downtime >= max_downtime_hours
    daily_totals_filtered = daily_totals[daily_totals['Tiempo_detenido'] < max_downtime_hours].copy()

    # Create a single enhanced figure with better dimensions for readability
    fig, ax = plt.subplots(figsize=(16, 9))

    # Define colors
    day_color = '#1a73e8'  # Vibrant blue for day shift
    night_color = '#9c27b0'  # Vibrant purple for night shift
    total_color = '#e53935'  # Vibrant red for total
    weekend_color = '#f5f5f5'  # Light gray for weekend highlight

    # Convert dates to datetime for better x-axis handling
    x_dates = daily_totals['DateDT'].values

    # Highlight weekends with light background
    for i, row in daily_totals.iterrows():
        if row['DayOfWeek'] >= 5:  # Saturday (5) and Sunday (6)
            ax.axvspan(row['DateDT'] - pd.Timedelta(hours=12),
                       row['DateDT'] + pd.Timedelta(hours=12),
                       alpha=0.2, color=weekend_color, zorder=0)

    # Create stacked bar chart for day and night shifts
    bar_width = 0.65
    day_bars = ax.bar(x_dates, daily_totals['Tiempo_detenido_D'],
                      label='Turno Día', color=day_color, alpha=0.9, width=bar_width,
                      edgecolor='white', linewidth=0.7, zorder=3)
    night_bars = ax.bar(x_dates, daily_totals['Tiempo_detenido_N'], bottom=daily_totals['Tiempo_detenido_D'],
                        label='Turno Noche', color=night_color, alpha=0.9, width=bar_width,
                        edgecolor='white', linewidth=0.7, zorder=3)

    # Add total line
    ax_twin = ax.twinx()
    line = ax_twin.plot(x_dates, daily_totals['Tiempo_detenido'], 'o-',
                        color=total_color, linewidth=3, label='Total',
                        markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                        zorder=5)

    # Add week separators
    for week in daily_totals['CustomWeek'].unique():
        week_data = daily_totals[daily_totals['CustomWeek'] == week]
        if len(week_data) > 0:
            first_day = week_data['DateDT'].min()
            ax.axvline(x=first_day, color='#7f8c8d', linestyle='--', alpha=0.5, linewidth=1.2, zorder=1)

    # Format x-axis dates
    date_format = mdates.DateFormatter('%d-%b')
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontweight='bold')

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='#bdc3c7', linewidth=1, zorder=1)

    # Add value annotations for all points
    for i, (date, total, day_val, night_val) in enumerate(zip(
            x_dates, daily_totals['Tiempo_detenido'],
            daily_totals['Tiempo_detenido_D'], daily_totals['Tiempo_detenido_N'])):

        # Annotate total values
        if total > 0:
            # Style based on value significance
            if total > max(daily_totals['Tiempo_detenido']) * 0.5:  # Major values
                fontsize = 12
                fontweight = 'bold'
                y_offset = 10
                boxstyle = 'round,pad=0.4'
                alpha = 0.95
            elif total > max(daily_totals['Tiempo_detenido']) * 0.2:  # Medium values
                fontsize = 11
                fontweight = 'bold'
                y_offset = 8
                boxstyle = 'round,pad=0.3'
                alpha = 0.9
            else:  # Minor values
                fontsize = 10
                fontweight = 'normal'
                y_offset = 6
                boxstyle = 'round,pad=0.2'
                alpha = 0.85

            # Annotate total value
            ax_twin.annotate(f'{total:.1f}h',
                             xy=(date, total),
                             xytext=(0, y_offset),
                             textcoords='offset points',
                             ha='center', va='bottom',
                             fontsize=fontsize, color=total_color, fontweight=fontweight,
                             bbox=dict(boxstyle=boxstyle, fc='white', ec='#d5dbdb', alpha=alpha),
                             zorder=10)

        # Day shift values
        if day_val > 0.5:
            text_color = 'white' if day_val > max(daily_totals['Tiempo_detenido_D']) * 0.15 else '#1a1a1a'
            fontsize = 10 if day_val > max(daily_totals['Tiempo_detenido_D']) * 0.3 else 9

            ax.annotate(f'{day_val:.1f}h',
                        xy=(date, day_val / 2),
                        ha='center', va='center',
                        fontsize=fontsize, color=text_color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none',
                                  alpha=0.3) if text_color != 'white' else None,
                        zorder=10)

        # Night shift values
        if night_val > 0.5:
            text_color = 'white' if night_val > max(daily_totals['Tiempo_detenido_N']) * 0.15 else '#1a1a1a'
            fontsize = 10 if night_val > max(daily_totals['Tiempo_detenido_N']) * 0.3 else 9

            ax.annotate(f'{night_val:.1f}h',
                        xy=(date, day_val + night_val / 2),
                        ha='center', va='center',
                        fontsize=fontsize, color=text_color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none',
                                  alpha=0.3) if text_color != 'white' else None,
                        zorder=10)

    # Set labels and title
    ax.set_title('Análisis de Tiempo Detenido por Turno y Día', fontweight='bold', pad=25,
                 color='#263238', fontsize=20)
    ax.set_ylabel('Horas Detenido por Turno', fontweight='bold', color='#263238', labelpad=15)
    ax_twin.set_ylabel('Horas Detenido Total', color=total_color, fontweight='bold', labelpad=15)

    # Set y-axis limits and formatting
    ax.set_ylim(0, max_downtime_hours * 0.95)
    ax_twin.set_ylim(0, max_downtime_hours * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}h'))
    ax_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}h'))
    ax_twin.tick_params(axis='y', colors=total_color)
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5, zorder=0)

    # Add statistics summary box
    original_total = df['Tiempo_detenido'].sum()
    total_hours = min(original_total, daily_totals['Tiempo_detenido'].sum())
    avg_daily = daily_totals_filtered['Tiempo_detenido'].mean()

    # Calculate max day stats
    if len(daily_totals_filtered) > 0:
        max_day = daily_totals_filtered['Tiempo_detenido'].max()
        max_day_date = daily_totals_filtered.loc[daily_totals_filtered['Tiempo_detenido'].idxmax(), 'Date']
    else:
        max_day = daily_totals['Tiempo_detenido'].max()
        max_day_date = daily_totals.loc[daily_totals['Tiempo_detenido'].idxmax(), 'Date']

    # Calculate percentages
    day_percent = (df['Tiempo_detenido_D'].sum() / total_hours * 100) if total_hours > 0 else 0
    night_percent = (df['Tiempo_detenido_N'].sum() / total_hours * 100) if total_hours > 0 else 0
    excluded_days = len(daily_totals) - len(daily_totals_filtered)

    # Create stats text
    stats_title = "RESUMEN DE TIEMPO DETENIDO"
    stats_text = f"\nTotal: {total_hours:.1f}h\n"
    stats_text += f"Promedio Diario: {avg_daily:.1f}h\n"
    stats_text += f"Máximo: {max_day:.1f}h ({max_day_date})\n\n"
    stats_text += f"Distribución por Turno:\n"
    stats_text += f"• Día: {day_percent:.1f}% ({df['Tiempo_detenido_D'].sum():.1f}h)\n"
    stats_text += f"• Noche: {night_percent:.1f}% ({df['Tiempo_detenido_N'].sum():.1f}h)"

    if excluded_days > 0:
        stats_text += f"\n\nNota: {excluded_days} día(s) excluido(s) del promedio"

    # Add text box
    props = dict(boxstyle='round4,pad=1.0', facecolor='white', alpha=0.97,
                 edgecolor='#bdc3c7', linewidth=1.5)

    # Add title to the stats box
    # ax.text(0.02, 0.97, stats_title, transform=ax.transAxes, fontsize=12,
    #         verticalalignment='top', horizontalalignment='left',
    #         color='#263238', fontweight='bold', bbox=dict(facecolor='#f5f5f5',
    #                                                       alpha=0.97, edgecolor='#bdc3c7',
    #                                                       pad=0.5, boxstyle='round,pad=0.5'))

    # Add stats content
    ax.text(0.02, 0.93, stats_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props, color='#263238', zorder=10)

    # Create legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_twin.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    legend = ax.legend(all_handles, all_labels, loc='upper right', frameon=True,
                       fancybox=True, shadow=True, fontsize=11, ncol=3,
                       title="LEYENDA", title_fontsize=12)
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('#bdc3c7')
    legend.get_frame().set_linewidth(1.5)

    # Add border to the plot
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

    # Add a watermark
    fig.text(0.99, 0.01, 'Análisis de Tiempo Detenido',
             ha='right', va='bottom', color='#9e9e9e', fontsize=8, alpha=0.7)

    # Adjust layout
    plt.tight_layout(pad=2.0)

    return fig