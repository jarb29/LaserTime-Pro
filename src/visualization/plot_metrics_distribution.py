import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from src.utils.replace_or_remove_outliers import replace_outliers
## Distribution Visualization
def plot_metrics_distribution(df):
    """Create publication-quality Gaussian distribution plots with Harvard-style visualization standards."""
    # Import additional libraries for enhanced visuals
    import matplotlib.patheffects as path_effects
    import matplotlib.font_manager as fm
    from matplotlib.ticker import MaxNLocator, AutoMinorLocator

    # Create a copy of the dataframe to avoid modifying the original
    plot_df = df.copy()

    # Define the metrics and their display names
    metrics = {
        'Tiempo_Real': 'Tiempo Real de Operación',
        'Tiempo_detenido': 'Tiempo de Detención',
        'Tiempo_Retraso': 'Tiempo de Retraso',
        'D': 'Horas Turno Día',
        'N': 'Horas Turno Noche'
    }

    # Clean the data by handling outliers and negative values
    for metric in metrics.keys():
        if metric in ['D', 'N']:
            # For D and N, only keep positive values
            plot_df[metric] = plot_df[plot_df[metric] > 0][metric]
        else:
            # For other metrics, remove negative values
            plot_df[metric] = plot_df[plot_df[metric] >= 0][metric]

        # Replace outliers with mean values
        if not plot_df[metric].empty:
            print(f"\nProcessing outliers for {metrics[metric]} ({metric}):")
            plot_df[metric] = replace_outliers(plot_df[metric], metrics[metric])

    # Rename columns to use the display names for better readability in the plot
    plot_df = plot_df.rename(columns=metrics)

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

    # Create a custom colormap for the plots based on Harvard colors
    harvard_colors = [harvard_palette['crimson'], harvard_palette['blue'],
                     harvard_palette['ivy'], harvard_palette['gold'],
                     harvard_palette['slate']]

    # Set the seaborn style with customizations
    sns.set_style("whitegrid", {
        'grid.linestyle': ':',
        'axes.edgecolor': harvard_palette['gray_dark'],
        'axes.linewidth': 1.2,
        'xtick.color': harvard_palette['gray_dark'],
        'ytick.color': harvard_palette['gray_dark'],
        'text.color': harvard_palette['black'],
        'font.family': 'serif',
    })

    # Create the pairplot with Gaussian KDE plots on the diagonal
    g = sns.pairplot(
        plot_df[list(metrics.values())],
        diag_kind="kde",
        corner=True,  # Only show the lower triangle to avoid redundancy
        plot_kws={
            "s": 60,
            "alpha": 0.7,
            "edgecolor": harvard_palette['white'],
            "linewidth": 0.8,
            "color": harvard_palette['crimson']
        },
        diag_kws={
            "fill": True,
            "alpha": 0.6,
            "linewidth": 2.5,
            "color": harvard_palette['crimson']
        },
        height=4.0,
    )

    # Enhance the figure with Harvard-style elements
    g.fig.subplots_adjust(top=0.93, bottom=0.07, left=0.08, right=0.92, hspace=0.2, wspace=0.2)

    # Add a sophisticated title with Harvard styling
    title = g.fig.suptitle('Distribucion y Correlacion de Metricas de Tiempo',
                  fontsize=20, fontweight='bold', y=0.98, color=harvard_palette['black'])

    # Add a subtitle with additional context
    g.fig.text(0.5, 0.945, 'Analisis de Patrones Temporales en Procesos de Manufactura',
              ha='center', fontsize=14, color=harvard_palette['gray_dark'],
              style='italic', fontfamily='serif')

    # Add a horizontal separator line below the title
    g.fig.text(0.08, 0.925, '_'*150, color=harvard_palette['gray_light'], fontsize=8)

    # Add a footer with the date and citation information
    current_date = datetime.now().strftime("%d de %B, %Y")
    g.fig.text(0.08, 0.02, f'Generado el {current_date}',
              ha='left', fontsize=9, color=harvard_palette['gray_medium'],
              style='italic', fontfamily='serif')

    # Add a citation/reference footer in Harvard style
    g.fig.text(0.92, 0.02, 'Analisis de Datos Industriales (2025)',
              ha='right', fontsize=9, color=harvard_palette['gray_medium'],
              style='italic', fontfamily='serif')

    # Add a watermark/logo text
    watermark = g.fig.text(0.5, 0.5, 'ANALISIS INDUSTRIAL',
             fontsize=60, color=harvard_palette['gray_light'],
             ha='center', va='center', alpha=0.07,
             rotation=30, fontweight='bold', fontfamily='serif')

    # Enhance each subplot with Harvard-style elements
    for i, metric_i in enumerate(list(metrics.values())):
        for j in range(i+1):
            ax = g.axes[i, j]

            # Apply Harvard styling to all axes
            ax.tick_params(colors=harvard_palette['gray_dark'], which='both')
            ax.spines['left'].set_color(harvard_palette['gray_dark'])
            ax.spines['bottom'].set_color(harvard_palette['gray_dark'])

            # Add minor ticks for a more academic look
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

            # Set background color
            ax.set_facecolor(harvard_palette['background'])

            # Add grid with Harvard styling
            ax.grid(True, linestyle=':', linewidth=0.8, color=harvard_palette['gray_light'], alpha=0.7)

            # If this is a diagonal plot (KDE)
            if i == j:
                # Calculate statistics for the current metric
                data = plot_df[metric_i].dropna()
                if len(data) > 0:
                    mean = data.mean()
                    median = data.median()
                    std_dev = data.std()
                    min_val = data.min()
                    max_val = data.max()

                    # Calculate additional academic statistics
                    q1 = data.quantile(0.25)
                    q3 = data.quantile(0.75)
                    iqr = q3 - q1
                    skewness = data.skew()
                    kurtosis = data.kurtosis()

                    # Create a comprehensive statistics text in Harvard style with improved readability
                    stats_text = (
                        f'Datos:\n\n'
                        f'- Media:      {mean:.2f}h\n'
                        f'- Mediana:    {median:.2f}h\n'
                        f'- Desv. Est.: {std_dev:.2f}\n'
                        # f'- IQR:        {iqr:.2f}\n'
                        f'- Rango:      [{min_val:.2f}, {max_val:.2f}]\n'
                        f'- n =         {len(data)}\n\n'
                        # f'Asimetría: {skewness:.2f}  |  Curtosis: {kurtosis:.2f}'
                    )

                    # Add the statistics text with enhanced Harvard styling for better readability
                    stats_box = ax.text(0.97, 0.97, stats_text,
                                      transform=ax.transAxes,
                                      verticalalignment='top',
                                      horizontalalignment='right',
                                      bbox=dict(
                                          boxstyle="round,pad=0.8",
                                          facecolor=harvard_palette['white'],
                                          edgecolor=harvard_palette['slate'],
                                          linewidth=1,
                                          alpha=0.97
                                      ),
                                      fontsize=14,
                                      color=harvard_palette['black'],
                                      fontfamily='sans-serif',
                                      fontweight='medium',
                                      linespacing=1)

                    # Add enhanced shadow effect for better visibility and depth
                    stats_box.set_path_effects([
                        path_effects.SimplePatchShadow(offset=(2, -2), alpha=0.25, shadow_rgbFace='#CCCCCC'),
                        path_effects.Normal()
                    ])

                    # Add vertical lines for mean and median with Harvard styling (without legend entries)
                    mean_line = ax.axvline(x=mean, color=harvard_palette['crimson'], linestyle='-',
                                         linewidth=2.5, alpha=0.9)

                    median_line = ax.axvline(x=median, color=harvard_palette['ivy'], linestyle='-',
                                           linewidth=2.5, alpha=0.9)

                    # Add confidence interval shading (±1 std dev)
                    ax.axvspan(mean - std_dev, mean + std_dev, alpha=0.1, color=harvard_palette['crimson'])

                    # Add y-axis label for KDE plots with Harvard styling
                    ax.set_ylabel('Densidad de Probabilidad', fontsize=11,
                                color=harvard_palette['black'], labelpad=10,
                                fontfamily='serif', fontweight='normal')

                    # Add x-axis label with the metric name
                    ax.set_xlabel(metric_i, fontsize=11, color=harvard_palette['black'],
                                labelpad=10, fontfamily='serif', fontweight='normal')

                    # Add a small indicator of the variable number in Harvard style
                    ax.text(0.03, 0.97, f'Variable {i+1}', transform=ax.transAxes,
                           color=harvard_palette['gray_medium'], fontsize=9,
                           fontweight='normal', ha='left', va='top',
                           bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor=harvard_palette['white'],
                                   edgecolor=harvard_palette['gray_light'],
                                   alpha=0.7))

            # If this is a scatter plot (correlation)
            elif j < i:
                # Get the data for the two metrics
                x_metric = list(metrics.values())[j]
                y_metric = list(metrics.values())[i]

                # Calculate the correlation coefficient and p-value
                from scipy import stats
                # Create a temporary dataframe with just the two metrics we're comparing
                temp_df = plot_df[[x_metric, y_metric]].dropna()

                # Initialize variables with default values
                corr = float('nan')
                p_value = float('nan')
                sig_marker = ""

                # Only calculate correlation if we have enough data points
                if len(temp_df) >= 2:
                    corr, p_value = stats.pearsonr(temp_df[x_metric], temp_df[y_metric])

                    # Determine significance markers
                    if p_value < 0.001:
                        sig_marker = "***"
                    elif p_value < 0.01:
                        sig_marker = "**"
                    elif p_value < 0.05:
                        sig_marker = "*"

                    # Add the correlation coefficient with significance markers
                    corr_text = ax.text(0.05, 0.95,
                                      f'r = {corr:.2f}{sig_marker}\np = {p_value:.3f}',
                                  transform=ax.transAxes,
                                  verticalalignment='top',
                                  horizontalalignment='left',
                                  bbox=dict(
                                      boxstyle="round,pad=0.4",
                                      facecolor=harvard_palette['white'],
                                      edgecolor=harvard_palette['gray_light'],
                                      alpha=0.9
                                  ),
                                  fontsize=9,
                                  color=harvard_palette['black'],
                                  fontfamily='serif')

                    # Add a regression line with confidence interval
                    sns.regplot(x=x_metric, y=y_metric, data=temp_df,
                              scatter=False, ci=95, line_kws={"color": harvard_palette['blue'],
                                                            "linewidth": 1.5,
                                                            "alpha": 0.7},
                              ax=ax)
                else:
                    # Display a message when there's not enough data
                    ax.text(0.5, 0.5, 'Datos insuficientes\npara correlación',
                          transform=ax.transAxes,
                          verticalalignment='center',
                          horizontalalignment='center',
                          fontsize=10,
                          color=harvard_palette['gray_medium'],
                          style='italic',
                          fontfamily='serif')

                # Set axis labels with Harvard styling
                ax.set_xlabel(x_metric, fontsize=11, color=harvard_palette['black'],
                            labelpad=10, fontfamily='serif')
                ax.set_ylabel(y_metric, fontsize=11, color=harvard_palette['black'],
                            labelpad=10, fontfamily='serif')

                # Add a note about significance levels if this is the last plot
                if i == len(list(metrics.values()))-1 and j == 0:
                    ax.text(0.05, 0.05,
                          "Significancia: * p<0.05, ** p<0.01, *** p<0.001",
                          transform=ax.transAxes,
                          verticalalignment='bottom',
                          horizontalalignment='left',
                          fontsize=8,
                          fontstyle='italic',
                          color=harvard_palette['gray_medium'])

    # Add a caption/description at the bottom of the figure
    caption_text = (
        "Figura 1: Distribucion de metricas temporales y sus correlaciones. Los paneles diagonales muestran la densidad de probabilidad "
        "de cada variable con lineas verticales indicando la media (rojo) y mediana (verde). Los paneles inferiores muestran "
        "las relaciones entre pares de variables con lineas de regresion y coeficientes de correlacion."
    )

    g.fig.text(0.5, 0.005, caption_text, ha='center', va='bottom', fontsize=9,
              color=harvard_palette['gray_dark'], style='italic',
              bbox=dict(facecolor=harvard_palette['background'], edgecolor='none', pad=4),
              wrap=True)

    # Show the plot
    plt.show()
    print('Considerar que horas del turno dia y del turno noche incluyen el retraso')
