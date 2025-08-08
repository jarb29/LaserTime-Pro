import math
import matplotlib.pyplot as plt

def plot_velocity(data):
    # Flatten and organize data
    flattened_data = []
    for year_month, espesors in data.items():
        year, month = year_month.split('_')
        month_num = int(month)
        year_num = int(year)

        # Create a sortable key (year*100 + month)
        sort_key = year_num * 100 + month_num

        # Spanish month names
        month_name = {
            '1': 'Ene', '2': 'Feb', '3': 'Mar', '4': 'Abr',
            '5': 'May', '6': 'Jun', '7': 'Jul', '8': 'Ago',
            '9': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dic'
        }.get(month, month)

        display_month = f"{month_name} {year}"

        for espesor, velocity in espesors.items():
            if velocity > 0:
                flattened_data.append((display_month, int(espesor), velocity, sort_key))

    # Group data by espesor
    espesor_data = {}
    for month, espesor, velocity, sort_key in flattened_data:
        if espesor not in espesor_data:
            espesor_data[espesor] = []
        espesor_data[espesor].append((month, velocity, sort_key))

    # Sort espesors and prepare for plotting
    unique_espesors = sorted(espesor_data.keys())
    num_espesors = len(unique_espesors)
    num_rows = math.ceil(num_espesors / 3)

    # Set up figure with improved style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': '#f8f9fa',
        'axes.facecolor': '#ffffff',
    })

    # Close any existing figures to prevent double plotting
    plt.close('all')

    # Create figure with extra space at top
    fig, axs = plt.subplots(num_rows, 3, figsize=(18, num_rows * 4.5),
                           facecolor='#f8f9fa', squeeze=False, dpi=100)

    # Enhanced color palette with better contrast
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors = [base_colors[i % len(base_colors)] for i in range(len(unique_espesors))]

    # Plot each espesor in its own subplot
    for i, espesor in enumerate(unique_espesors):
        row, col = divmod(i, 3)
        ax = axs[row, col]

        # Sort data chronologically by the sort_key
        month_data = sorted(espesor_data[espesor], key=lambda x: x[2])
        if not month_data:
            continue

        months = [m for m, _, _ in month_data]
        velocities = [v for _, v, _ in month_data]

        # Create bar chart with gradient effect
        bars = ax.bar(range(len(months)), velocities,
                     color=colors[i], alpha=0.85,
                     edgecolor='black', linewidth=0.8, width=0.7)

        # Add subtle shadow effect to bars
        for bar in bars:
            x = bar.get_x()
            width = bar.get_width()
            height = bar.get_height()
            ax.add_patch(plt.Rectangle((x, 0), width, height,
                                      fill=True, alpha=0.1,
                                      color='black', linewidth=0))

        # Add labels with improved styling in Spanish
        ax.set_title(f'Espesor: {espesor} mm', fontweight='bold', pad=15)
        ax.set_xlabel('Mes', fontweight='bold', labelpad=10)
        ax.set_ylabel('Velocidad (m/min)', fontweight='bold', labelpad=10)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, ha='right')

        # Set y-axis to start from 0
        max_velocity = max(velocities) if velocities else 0
        ax.set_ylim(bottom=0, top=max_velocity * 1.15)

        # Add value labels with improved styling
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01 * max_velocity,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10,
                   fontweight='bold', color='#333333')

        # Add grid lines with improved styling
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.set_axisbelow(True)

        # Add a subtle box around the plot
        for spine in ax.spines.values():
            spine.set_edgecolor('#dddddd')
            spine.set_linewidth(1)

    # Remove unused subplots
    for i in range(num_espesors, num_rows * 3):
        row, col = divmod(i, 3)
        if row < len(axs) and col < len(axs[0]):
            fig.delaxes(axs[row, col])

    # Adjust layout with space for title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Add a main title with improved styling in Spanish
    fig.suptitle('Análisis de Velocidad de Corte por Espesor de Material',
                fontsize=16, fontweight='bold', y=0.98,
                fontfamily='serif', color='#333333')

    # Add a footer with improved styling in Spanish
    plt.figtext(0.5, 0.01, 'Fuente: Análisis de Datos de Mecanizado, 2025',
               ha='center', fontsize=10, fontstyle='italic', color='#666666')

    # Add a subtle border to the entire figure
    fig.patch.set_linewidth(1)
    fig.patch.set_edgecolor('#cccccc')

    # Show the figure and return None to prevent double display
    plt.show()
    return None