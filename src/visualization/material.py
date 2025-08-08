import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_espesor_analysis_plots(df, group_size=3, save_dir=None):
    """
    Create a set of plots analyzing the relationship between Espesor, Longitud de Corte, and Tiempo.

    Parameters:
    -----------
    df : pandas.DataFrame or str
        DataFrame containing the data or path to the Excel file containing the data
    group_size : int, optional
        Number of espesores to include in each group of plots (default: 3)
    save_dir : str, optional
        Directory to save the plots. If None, plots are only displayed (default: None)

    Returns:
    --------
    list
        List of matplotlib figure objects created
    """
    # If df is a string (file path), read the Excel file
    if isinstance(df, str):
        df = pd.read_excel(df)

    # Create a copy to avoid modifying the original dataframe
    df = df.copy()

    # Create bins for Tiempo for categorical plots
    df['Tiempo_bins'] = pd.cut(df['Tiempo'], bins=5, labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto'])

    # Get unique espesores and group them in sets
    unique_espesores = sorted(df['Espesor'].unique())
    espesor_groups = [unique_espesores[i:i + group_size] for i in range(0, len(unique_espesores), group_size)]

    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 10})

    # List to store figure objects
    figures = []

    # For each group of espesores, create a set of 4 different plots
    for group_idx, espesor_group in enumerate(espesor_groups):
        # Filter data for current espesor group
        df_filtered = df[df['Espesor'].isin(espesor_group)].copy()

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        figures.append(fig)

        # 1. Scatter plot with Tiempo as color gradient
        scatter = axes[0, 0].scatter(df_filtered['Espesor'],
                                     df_filtered['Longitude Corte (m)'],
                                     c=df_filtered['Tiempo'],
                                     cmap='viridis',
                                     alpha=0.7,
                                     s=80)

        axes[0, 0].set_title('Scatter Plot - Tiempo como Gradiente de Color', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Espesor (mm)')
        axes[0, 0].set_ylabel('Longitud de Corte (m)')
        axes[0, 0].grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[0, 0])
        cbar.set_label('Tiempo (s)')

        # 2. Box plot grouped by Espesor and Tiempo bins
        sns.boxplot(data=df_filtered,
                    x='Espesor',
                    y='Longitude Corte (m)',
                    hue='Tiempo_bins',
                    ax=axes[0, 1])

        axes[0, 1].set_title('Box Plot - Longitud de Corte por Espesor y Tiempo', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Espesor (mm)')
        axes[0, 1].set_ylabel('Longitud de Corte (m)')
        axes[0, 1].legend(title='Tiempo', bbox_to_anchor=(1.02, 1), loc='upper left')

        # 3. Heatmap of average Tiempo by Espesor and binned Longitud de Corte
        df_filtered.loc[:, 'Longitud_bins'] = pd.cut(df_filtered['Longitude Corte (m)'], bins=5)

        # Create pivot table
        heatmap_data = df_filtered.pivot_table(
            values='Tiempo',
            index='Longitud_bins',
            columns='Espesor',
            aggfunc='mean'
        )

        # Plot heatmap
        sns.heatmap(heatmap_data,
                    annot=True,
                    fmt='.1f',
                    cmap='YlOrRd',
                    ax=axes[1, 0])

        axes[1, 0].set_title('Heatmap - Tiempo Promedio por Espesor y Longitud', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Espesor (mm)')
        axes[1, 0].set_ylabel('Rango de Longitud de Corte (m)')

        # 4. Line plot showing relationship between variables
        for espesor in espesor_group:
            espesor_data = df_filtered[df_filtered['Espesor'] == espesor]

            # Sort by Longitud de Corte for smoother line
            espesor_data = espesor_data.sort_values('Longitude Corte (m)')

            # Plot line
            axes[1, 1].plot(espesor_data['Longitude Corte (m)'],
                            espesor_data['Tiempo'],
                            'o-',
                            label=f'Espesor {espesor} mm',
                            alpha=0.7)

        axes[1, 1].set_title('Line Plot - Relación entre Longitud de Corte y Tiempo', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Longitud de Corte (m)')
        axes[1, 1].set_ylabel('Tiempo (s)')
        axes[1, 1].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)

        # Main title
        fig.suptitle(f'Análisis de Espesor, Longitud de Corte y Tiempo - Grupo {group_idx + 1}',
                     fontsize=16, fontweight='bold', y=0.98)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(right=0.85, top=0.92)

        # Save figure if save_dir is provided
        if save_dir:
            import os
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fig.savefig(os.path.join(save_dir, f'espesor_analysis_group_{group_idx+1}.png'),
                        dpi=300, bbox_inches='tight')

        plt.show()

    return figures