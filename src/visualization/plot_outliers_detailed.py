import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def plot_outliers_detailed(df, outliers_mask, diagnostics):
    """
    Create detailed plots to visualize the outliers and regression model fit.

    This function generates a 2x2 grid of plots to help visualize and understand
    the outliers detected by the regression model:
    1. Actual vs Predicted: Shows how well the model predicts machining time
    2. Studentized Residuals vs Predicted: Shows the distribution of residuals
    3. 3D Visualization: Shows the relationship between cutting length, thickness, and time
    4. Leverage vs Studentized Residuals: Helps identify influential outliers

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing machining data. Must include the following columns:
        - 'Longitude Corte (m)': Cutting length in meters
        - 'Espesor': Material thickness
        - 'Tiempo': Machining time in seconds

    outliers_mask : numpy.ndarray
        Boolean mask indicating which rows in the DataFrame are outliers

    diagnostics : dict
        Dictionary containing diagnostic information from the regression model:
        - 'studentized_residuals': Studentized residuals for each observation
        - 'predicted_values': Model predictions for each observation
        - 'actual_values': Actual values for each observation
        - 'leverage': Leverage values for each observation
        - 'model_coefficients': Regression model coefficients

    Returns:
    -------
    None
        The function displays the plots but does not return any values

    Notes:
    -----
    - Outliers are highlighted in red in all plots
    - The 3D visualization helps understand the relationship between
      material properties and machining time
    - The function automatically handles missing values in the data
    """
    # Create a clean dataframe for plotting
    df_clean = df.copy()
    df_clean['Longitude Corte (m)'] = pd.to_numeric(df_clean['Longitude Corte (m)'], errors='coerce')
    df_clean['Espesor'] = pd.to_numeric(df_clean['Espesor'], errors='coerce')
    df_clean['Tiempo'] = pd.to_numeric(df_clean['Tiempo'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Longitude Corte (m)', 'Espesor', 'Tiempo'])

    # Get indices of valid rows
    valid_indices = df_clean.index

    # Filter outliers_mask to only include valid indices
    valid_mask = np.zeros(len(df_clean), dtype=bool)
    for i, idx in enumerate(valid_indices):
        if idx < len(outliers_mask) and outliers_mask[idx]:
            valid_mask[i] = True

    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Actual vs Predicted
    ax1 = fig.add_subplot(221)
    ax1.scatter(diagnostics['predicted_values'][~valid_mask],
                df_clean['Tiempo'][~valid_mask],
                alpha=0.5, label='Normal')
    ax1.scatter(diagnostics['predicted_values'][valid_mask],
                df_clean['Tiempo'][valid_mask],
                color='red', alpha=0.7, label='Outlier')
    ax1.plot([df_clean['Tiempo'].min(), df_clean['Tiempo'].max()],
             [df_clean['Tiempo'].min(), df_clean['Tiempo'].max()],
             'k--', label='Perfect Prediction')
    ax1.set_xlabel('Predicted Machining Time')
    ax1.set_ylabel('Actual Machining Time')
    ax1.legend()
    ax1.set_title('Actual vs Predicted Machining Time')

    # Plot 2: Studentized Residuals vs Predicted
    ax2 = fig.add_subplot(222)
    ax2.scatter(diagnostics['predicted_values'][~valid_mask],
                diagnostics['studentized_residuals'][~valid_mask],
                alpha=0.5, label='Normal')
    ax2.scatter(diagnostics['predicted_values'][valid_mask],
                diagnostics['studentized_residuals'][valid_mask],
                color='red', alpha=0.7, label='Outlier')
    ax2.axhline(y=0, color='k', linestyle='--')
    ax2.axhline(y=3, color='r', linestyle='--')
    ax2.axhline(y=-3, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Machining Time')
    ax2.set_ylabel('Studentized Residuals')
    ax2.legend()
    ax2.set_title('Studentized Residuals vs Predicted')

    # Plot 3: 3D visualization
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(df_clean[~valid_mask]['Longitude Corte (m)'],
                df_clean[~valid_mask]['Espesor'],
                df_clean[~valid_mask]['Tiempo'],
                alpha=0.5, label='Normal')
    ax3.scatter(df_clean[valid_mask]['Longitude Corte (m)'],
                df_clean[valid_mask]['Espesor'],
                df_clean[valid_mask]['Tiempo'],
                color='red', alpha=0.7, label='Outlier')
    ax3.set_xlabel('Longitude Corte (m)')
    ax3.set_ylabel('Espesor')
    ax3.set_zlabel('Tiempo')
    ax3.legend()
    ax3.set_title('3D Visualization')

    # Plot 4: Leverage vs Studentized Residuals
    ax4 = fig.add_subplot(224)
    ax4.scatter(diagnostics['leverage'][~valid_mask],
                diagnostics['studentized_residuals'][~valid_mask],
                alpha=0.5, label='Normal')
    ax4.scatter(diagnostics['leverage'][valid_mask],
                diagnostics['studentized_residuals'][valid_mask],
                color='red', alpha=0.7, label='Outlier')
    ax4.axhline(y=0, color='k', linestyle='--')
    ax4.set_xlabel('Leverage')
    ax4.set_ylabel('Studentized Residuals')
    ax4.legend()
    ax4.set_title('Leverage vs Studentized Residuals')

    plt.tight_layout()
    plt.show()