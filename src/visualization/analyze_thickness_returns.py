## Return Pattern Detection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def analyze_thickness_returns(df, window_days=3):
    """
    Analyze returns to previously used thicknesses in sliding windows of specified days.
    An error is counted for each thickness when we go back to a thickness after switching to a different one.
    For example, in the sequence 8,8,8,10,12,8,12,12,8,10:
    - Thickness 8 has two errors (appears, disappears, reappears twice)
    - Thickness 10 has one error (appears, disappears, reappears)
    - Thickness 12 has one error (appears, disappears, reappears)
    Includes visualization of the error distribution with Harvard-style academic standards.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'Hora_Inicio' and 'Espesor' columns
    window_days : int
        Number of days to include in each sliding window (default=3)

    Returns:
    -----------
    pandas.DataFrame
        DataFrame containing the analysis results and plots a histogram of errors
    """
    # Import additional libraries for enhanced visuals
    import matplotlib
    import matplotlib.patheffects as path_effects
    from matplotlib.ticker import MaxNLocator, AutoMinorLocator
    from matplotlib.colors import LinearSegmentedColormap
    from datetime import datetime

    # Convert Hora_Inicio to datetime if it's not already
    df['Hora_Inicio'] = pd.to_datetime(df['Hora_Inicio'])

    # Sort by date
    df = df.sort_values('Hora_Inicio')

    # Get min and max dates
    min_date = df['Hora_Inicio'].min().date()
    max_date = df['Hora_Inicio'].max().date()

    # Initialize results list
    results = []

    # Create sliding windows
    current_date = min_date
    while current_date <= max_date:
        # Define window boundaries
        window_start = pd.Timestamp(current_date)
        window_end = window_start + pd.Timedelta(days=window_days)

        # Get data for current window
        mask = (df['Hora_Inicio'] >= window_start) & (df['Hora_Inicio'] < window_end)
        window_data = df[mask].copy()

        if not window_data.empty:
            # Get sequence of thicknesses
            thickness_sequence = window_data['Espesor'].tolist()
            dates_sequence = window_data['Hora_Inicio'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()

            # Track thickness appearances and errors
            thickness_errors = {}  # Dictionary to count errors per thickness
            thickness_seen = set()  # Set of thicknesses seen in current sequence
            current_thickness = None
            sequence_details = []
            total_returns_count = 0

            for thickness, date in zip(thickness_sequence, dates_sequence):
                # If we're changing thickness
                if thickness != current_thickness:
                    # If we're returning to a previously seen thickness
                    if thickness in thickness_seen:
                        # Increment error count for this thickness
                        if thickness not in thickness_errors:
                            thickness_errors[thickness] = 1
                        else:
                            thickness_errors[thickness] += 1
                        total_returns_count += 1
                    else:
                        # First time seeing this thickness in this window
                        thickness_seen.add(thickness)

                    current_thickness = thickness

                sequence_details.append(f"{thickness}mm ({date})")

            # Create a summary of errors per thickness
            errors_summary = []
            for thickness, count in thickness_errors.items():
                errors_summary.append(f"{thickness}mm: {count} errors")

            results.append({
                'start_date': window_start.strftime('%Y-%m-%d'),
                'end_date': (window_end - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                'errorPrograma': total_returns_count,
                'errors_per_thickness': thickness_errors,
                'errors_summary': ', '.join(errors_summary) if errors_summary else "No errors",
                'sequence_details': ' -> '.join(sequence_details)
            })

        # Move window forward by one day
        current_date += pd.Timedelta(days=1)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Harvard color palette (based on Harvard's brand colors)
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

    # Create a figure with two subplots with Harvard styling
    fig = plt.figure(figsize=(20, 10), dpi=150, facecolor=harvard_palette['white'])
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Calculate statistics first
    mean_errors = results_df['errorPrograma'].mean()
    median_errors = results_df['errorPrograma'].median()
    std_dev = results_df['errorPrograma'].std()
    max_errors = results_df['errorPrograma'].max()
    min_errors = results_df['errorPrograma'].min()
    total_windows = len(results_df)
    windows_with_errors = (results_df["errorPrograma"] > 0).sum()
    error_rate = (windows_with_errors / total_windows) * 100 if total_windows > 0 else 0

    # Calculate additional academic statistics
    q1 = results_df['errorPrograma'].quantile(0.25)
    q3 = results_df['errorPrograma'].quantile(0.75)
    iqr = q3 - q1

    # Apply Harvard styling to both axes
    for ax in [ax1, ax2]:
        ax.tick_params(colors=harvard_palette['gray_dark'], which='both')
        ax.spines['left'].set_color(harvard_palette['gray_dark'])
        ax.spines['bottom'].set_color(harvard_palette['gray_dark'])
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_facecolor(harvard_palette['background'])
        ax.grid(True, linestyle=':', linewidth=0.8, color=harvard_palette['gray_light'], alpha=0.7)

    # First subplot: Histogram of total errors per window with Harvard styling
    # Calculate optimal bin width using Freedman-Diaconis rule
    if len(results_df) > 1:
        bin_width = 2 * iqr / (len(results_df) ** (1/3)) if iqr > 0 else 1
        n_bins = max(int((results_df['errorPrograma'].max() - results_df['errorPrograma'].min()) / bin_width), 8)
    else:
        n_bins = 5  # Default if not enough data

    # Create custom colormap based on Harvard colors
    cmap = LinearSegmentedColormap.from_list('harvard_blues',
                                           [harvard_palette['slate'], harvard_palette['blue']],
                                           N=256)

    # Create histogram with enhanced styling
    n, bins, patches = ax1.hist(results_df['errorPrograma'],
                              bins=n_bins,
                              color=harvard_palette['blue'],
                              alpha=0.8,
                              edgecolor=harvard_palette['white'],
                              linewidth=1.2)

    # Apply gradient coloring to bars for a more academic look
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = plt.cm.Blues(np.linspace(0.4, 0.8, len(patches)))
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', c)
        p.set_edgecolor(harvard_palette['white'])

        # Add shadow effect to bars
        p.set_path_effects([
            path_effects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
            path_effects.Normal()
        ])

    # Add mean and median lines with Harvard styling
    mean_line = ax1.axvline(mean_errors, color=harvard_palette['crimson'], linestyle='-',
                         linewidth=2.5, alpha=0.9)

    median_line = ax1.axvline(median_errors, color=harvard_palette['ivy'], linestyle='-',
                           linewidth=2.5, alpha=0.9)

    # Add confidence interval shading (±1 std dev)
    ax1.axvspan(mean_errors - std_dev, mean_errors + std_dev, alpha=0.1,
               color=harvard_palette['crimson'])

    # Add count labels on top of each bar with enhanced styling
    max_height = max(n) if len(n) > 0 else 0
    for i in range(len(patches)):
        x = patches[i].get_x() + patches[i].get_width()/2
        y = n[i]
        if y > 0:
            y_pos = y + max_height * 0.05
            # Create a more stylish label with enhanced readability
            count_label = ax1.text(x, y_pos, f'{int(n[i])}',
                               ha='center', va='bottom',
                               color=harvard_palette['black'],
                               fontsize=12,
                               fontweight='bold',
                               bbox=dict(
                                   boxstyle="round,pad=0.5",
                                   facecolor=harvard_palette['white'],
                                   edgecolor=harvard_palette['slate'],
                                   linewidth=1.5,
                                   alpha=1.0
                               ))

            # Add stronger shadow effect for better contrast
            count_label.set_path_effects([
                path_effects.SimplePatchShadow(offset=(2, -2), alpha=0.5, shadow_rgbFace='#CCCCCC'),
                path_effects.Normal()
            ])

    # Create a comprehensive statistics text box for the first subplot
    stats_text1 = (
        f'Estadísticas Descriptivas\n'
        f'───────────────────────\n'
        f'Media: {mean_errors:.2f}\n'
        f'Mediana: {median_errors:.2f}\n'
        f'Desv. Est.: {std_dev:.2f}\n'
        f'IQR: {iqr:.2f}\n'
        f'Rango: [{min_errors:.0f}, {max_errors:.0f}]\n'
        f'Ventanas con Errores: {windows_with_errors} ({error_rate:.1f}%)\n'
        f'Total de Ventanas: {total_windows}'
    )

    # Add the statistics text box with Harvard styling
    stats_box1 = ax1.text(0.97, 0.97, stats_text1,
                       transform=ax1.transAxes,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(
                           boxstyle="round,pad=0.6",
                           facecolor=harvard_palette['white'],
                           edgecolor=harvard_palette['slate'],
                           linewidth=1,
                           alpha=0.95
                       ),
                       fontsize=14,
                       color=harvard_palette['black'],
                       fontfamily='monospace')

    # Add shadow effect to the stats box
    stats_box1.set_path_effects([
        path_effects.SimplePatchShadow(offset=(2, -2), alpha=0.2),
        path_effects.Normal()
    ])

    # Customize the first subplot with Harvard-style labels
    title1 = ax1.set_title('Distribución de Errores Totales por Ventana',
                        pad=10, fontsize=14, fontweight='bold',
                        color=harvard_palette['black'])

    # Add subtitle with window days information
    ax1.text(0.5, 1.05, f'Análisis de Ventanas de {window_days} Días',
           transform=ax1.transAxes,
           ha='center', va='bottom',
           fontsize=14, fontstyle='italic',
           color=harvard_palette['gray_dark'])

    # Add shadow effect to title
    title1.set_path_effects([
        path_effects.withSimplePatchShadow(offset=(1, -1), alpha=0.2)
    ])

    # Set axis labels with Harvard styling
    ax1.set_xlabel('Número Total de Retornos a Espesores Previos',
                 fontsize=14, labelpad=10, color=harvard_palette['black'],
                 fontfamily='serif', fontweight='normal')

    ax1.set_ylabel('Frecuencia',
                 fontsize=14, labelpad=10, color=harvard_palette['black'],
                 fontfamily='serif', fontweight='normal')


    # Second subplot: Errors per thickness with Harvard styling
    # Aggregate errors per thickness across all windows
    thickness_error_counts = {}

    for result in results:
        if 'errors_per_thickness' in result:
            for thickness, count in result['errors_per_thickness'].items():
                if thickness not in thickness_error_counts:
                    thickness_error_counts[thickness] = 0
                thickness_error_counts[thickness] += count

    # Sort thicknesses by error count (descending)
    sorted_thicknesses = sorted(thickness_error_counts.items(), key=lambda x: x[1], reverse=True)

    if sorted_thicknesses:
        thicknesses = [f"{t}mm" for t, _ in sorted_thicknesses]
        counts = [c for _, c in sorted_thicknesses]

        # Calculate additional statistics for the second subplot
        total_errors = sum(counts)
        max_thickness_errors = max(counts) if counts else 0
        avg_errors_per_thickness = total_errors / len(counts) if counts else 0

        # Create a custom colormap for the bars
        cmap = matplotlib.colormaps['Blues']
        colors = [cmap(0.3 + 0.5 * (i / len(counts))) for i in range(len(counts))]

        # Create bar chart with Harvard styling
        bars = ax2.bar(thicknesses, counts,
                     color=colors,
                     alpha=0.8,
                     edgecolor=harvard_palette['white'],
                     linewidth=1.2)

        # Add shadow effect to bars
        for bar in bars:
            bar.set_path_effects([
                path_effects.SimpleLineShadow(offset=(1, -1), alpha=0.3),
                path_effects.Normal()
            ])

        # Add value labels on top of each bar with Harvard styling
        for bar in bars:
            height = bar.get_height()
            value_label = ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{int(height)}',
                               ha='center', va='bottom',
                               fontsize=12,
                               fontweight='bold',
                               color=harvard_palette['black'],
                               bbox=dict(
                                   boxstyle="round,pad=0.5",
                                   facecolor=harvard_palette['white'],
                                   edgecolor=harvard_palette['slate'],
                                   linewidth=1.5,
                                   alpha=1.0
                               ))

            # Add stronger shadow effect for better contrast
            value_label.set_path_effects([
                path_effects.SimplePatchShadow(offset=(2, -2), alpha=0.5, shadow_rgbFace='#CCCCCC'),
                path_effects.Normal()
            ])

        # Create a comprehensive statistics text box for the second subplot
        stats_text2 = (
            f'Resumen de Errores\n'
            f'───────────────\n'
            f'Total de Errores: {total_errors}\n'
            f'Espesores Analizados: {len(counts)}\n'
            f'Promedio por Espesor: {avg_errors_per_thickness:.2f}\n'
            f'Máximo: {max_thickness_errors} errores\n'
            f'Espesor con más errores: {thicknesses[0]}'
        )

        # Add the statistics text box with Harvard styling
        stats_box2 = ax2.text(0.97, 0.97, stats_text2,
                           transform=ax2.transAxes,
                           verticalalignment='top',
                           horizontalalignment='right',
                           bbox=dict(
                               boxstyle="round,pad=0.6",
                               facecolor=harvard_palette['white'],
                               edgecolor=harvard_palette['slate'],
                               linewidth=1,
                               alpha=0.95
                           ),
                           fontsize=14,
                           color=harvard_palette['black'],
                           fontfamily='monospace')

        # Add shadow effect to the stats box
        stats_box2.set_path_effects([
            path_effects.SimplePatchShadow(offset=(2, -2), alpha=0.2),
            path_effects.Normal()
        ])

        # Customize the second subplot with Harvard-style labels
        title2 = ax2.set_title('Errores por Espesor',
                            pad=20, fontsize=14, fontweight='bold',
                            color=harvard_palette['black'])

        # Add subtitle
        ax2.text(0.5, 1.05, 'Total de Errores en Todas las Ventanas',
               transform=ax2.transAxes,
               ha='center', va='bottom',
               fontsize=14, fontstyle='italic',
               color=harvard_palette['gray_dark'])

        # Add shadow effect to title
        title2.set_path_effects([
            path_effects.withSimplePatchShadow(offset=(1, -1), alpha=0.2)
        ])

        # Set axis labels with Harvard styling
        ax2.set_xlabel('Espesor',
                     fontsize=13, labelpad=10, color=harvard_palette['black'],
                     fontfamily='serif', fontweight='normal')

        ax2.set_ylabel('Número de Errores',
                     fontsize=13, labelpad=10, color=harvard_palette['black'],
                     fontfamily='serif', fontweight='normal')

        # Rotate x-axis labels for better readability with Harvard styling
        plt.setp(ax2.get_xticklabels(),
               rotation=45, ha='right',
               fontsize=11, color=harvard_palette['gray_dark'])

        # Add a note about what constitutes an "error"
        ax2.text(0.02, 0.02,
               "Nota: Un error ocurre cuando se vuelve a un espesor\ndespués de haber cambiado a otro diferente.",
               transform=ax2.transAxes,
               verticalalignment='bottom',
               horizontalalignment='left',
               fontsize=9,
               fontstyle='italic',
               color=harvard_palette['gray_medium'],
               bbox=dict(
                   boxstyle="round,pad=0.3",
                   facecolor=harvard_palette['background'],
                   edgecolor='none',
                   alpha=0.7
               ))
    else:
        # Display a message when there are no errors to show
        ax2.text(0.5, 0.5, 'No hay errores por espesor para mostrar',
               ha='center', va='center', fontsize=14,
               color=harvard_palette['gray_dark'],
               fontstyle='italic')

    # Add a sophisticated main title for the entire figure with Harvard styling
    main_title = fig.suptitle('Análisis de Retornos a Espesores Previos',
                           y=0.98, fontsize=20, fontweight='bold',
                           color=harvard_palette['black'])

    # Add shadow effect to main title
    main_title.set_path_effects([
        path_effects.withSimplePatchShadow(offset=(1, -1), alpha=0.2)
    ])

    # Add a subtitle with additional context
    fig.text(0.5, 0.94, 'Evaluación de Patrones de Cambio en Espesores de Material',
           ha='center', fontsize=14, color=harvard_palette['gray_dark'],
           style='italic', fontfamily='serif')

    # Add a horizontal separator line below the title
    fig.text(0.1, 0.925, '_'*150, color=harvard_palette['gray_light'], fontsize=8)

    # Add a footer with the date and citation information
    current_date = datetime.now().strftime("%d de %B, %Y")
    fig.text(0.1, 0.02, f'Generado el {current_date}',
           ha='left', fontsize=9, color=harvard_palette['gray_medium'],
           style='italic', fontfamily='serif')

    # Add a citation/reference footer in Harvard style
    fig.text(0.9, 0.02, 'Análisis de Datos Industriales (2025)',
           ha='right', fontsize=9, color=harvard_palette['gray_medium'],
           style='italic', fontfamily='serif')

    # Add a watermark/logo text
    watermark = fig.text(0.5, 0.5, 'ANÁLISIS INDUSTRIAL',
                      fontsize=60, color=harvard_palette['gray_light'],
                      ha='center', va='center', alpha=0.07,
                      rotation=30, fontweight='bold', fontfamily='serif')

    # Add a caption/description at the bottom of the figure
    caption_text = (
        "Figura 1: Análisis de patrones de retorno a espesores previamente utilizados. El panel izquierdo muestra la distribución "
        "de errores totales por ventana de tiempo, donde un error representa un retorno a un espesor previamente utilizado. "
        "El panel derecho muestra el total de errores por cada espesor específico a lo largo de todo el período analizado."
    )

    fig.text(0.5, 0.01, caption_text, ha='center', va='bottom', fontsize=9,
           color=harvard_palette['gray_dark'], style='italic',
           bbox=dict(facecolor=harvard_palette['background'], edgecolor='none', pad=4),
           wrap=True)

    # Adjust layout for Harvard-style spacing
    plt.subplots_adjust(top=0.88, bottom=0.12, wspace=0.2)

    plt.show()

    return results_df

