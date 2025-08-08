import pandas as pd
from sklearn.linear_model import LinearRegression
from IPython.display import display
import numpy as np
import os

def detect_outliers_regression(df, threshold=3, output_dir="data_final"):
    """
    Detect outliers in machining data using linear regression with studentized residuals.

    This function fits a linear regression model to predict machining time based on
    material properties (length and thickness), then identifies outliers using
    studentized residuals. Outliers are programs that take significantly more or less
    time than predicted by the model.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing machining data. Must include the following columns:
        - 'Programa': Program identifier
        - 'Longitude Corte (m)': Cutting length in meters
        - 'Espesor': Material thickness
        - 'Tiempo': Machining time in seconds

    threshold : float, optional
        Threshold for studentized residuals to identify outliers (default: 3).
        Points with absolute studentized residuals greater than this value
        are flagged as outliers.

    output_dir : str, optional
        Directory where to save output files (default: "data_final").
        Will be created if it doesn't exist.

    Returns:
    -------
    tuple
        A tuple containing:
        - numpy.ndarray: Boolean mask indicating outlier rows in the original DataFrame
        - dict: Diagnostic information including studentized residuals, predicted values,
          actual values, leverage values, and model coefficients

    Notes:
    -----
    - The function automatically saves outlier details to an Excel file
    - It also saves a clean version of the data (with outliers removed) to Excel
    - The linear regression model assumes a linear relationship between cutting length,
      material thickness, and machining time
    """
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()

    # Ensure numeric data types
    df_clean['Longitude Corte (m)'] = pd.to_numeric(df_clean['Longitude Corte (m)'], errors='coerce')
    df_clean['Espesor'] = pd.to_numeric(df_clean['Espesor'], errors='coerce')
    df_clean['Tiempo'] = pd.to_numeric(df_clean['Tiempo'], errors='coerce')

    # Drop rows with NaN values
    df_clean = df_clean.dropna(subset=['Longitude Corte (m)', 'Espesor', 'Tiempo'])

    # Create feature matrix X and target variable y
    X = df_clean[['Longitude Corte (m)', 'Espesor']].values  # Explicitly convert to numpy array
    y = df_clean['Tiempo'].values  # Explicitly convert to numpy array

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Get predictions
    y_pred = model.predict(X)

    # Calculate residuals
    residuals = y - y_pred

    # Calculate studentized residuals
    n = len(df_clean)

    # Add intercept column to X matrix for proper hat matrix calculation
    X_with_intercept = np.column_stack([np.ones(n), X])

    # Calculate mean squared error (MSE)
    # We use n-3 degrees of freedom because we have 2 predictors plus intercept
    mse = np.sum(residuals**2) / (n - 3)

    # Calculate hat matrix diagonal (leverage values)
    # The hat matrix H = X(X'X)^(-1)X' maps y to y_hat
    # The diagonal elements h_ii represent the leverage of each observation
    X_transpose_X_inverse = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept))
    hat_matrix = X_with_intercept.dot(X_transpose_X_inverse).dot(X_with_intercept.T)
    h = np.diagonal(hat_matrix)

    # Calculate studentized residuals
    # Studentized residuals account for the fact that observations with high leverage
    # have smaller residual variance
    studentized_residuals = residuals / np.sqrt(mse * (1 - h))

    # Flag as outlier if studentized residual is > threshold or < -threshold
    outliers_mask = abs(studentized_residuals) > threshold

    # Create diagnostics dictionary
    diagnostics = {
        'studentized_residuals': studentized_residuals,
        'predicted_values': y_pred,
        'actual_values': y,
        'leverage': h,
        'model_coefficients': {
            'Longitude Corte (m)': model.coef_[0],
            'Espesor': model.coef_[1],
            'intercept': model.intercept_
        }
    }

    # Print summary
    print(f"Number of outliers detected: {outliers_mask.sum()}")
    print(f"Percentage of outliers: {(outliers_mask.sum()/len(df_clean))*100:.2f}%")
    print("\nModel coefficients:")
    print(f"Longitude Corte (m): {model.coef_[0]:.4f}")
    print(f"Espesor: {model.coef_[1]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")

    # Create a mapping from cleaned indices to original indices
    index_map = dict(zip(range(len(df_clean)), df_clean.index))

    # Map the outliers mask back to the original dataframe
    full_mask = np.zeros(len(df), dtype=bool)
    for i, is_outlier in enumerate(outliers_mask):
        if is_outlier:
            full_mask[index_map[i]] = True

    # Display outlier details
    if outliers_mask.sum() > 0:
        outlier_df = df_clean.iloc[outliers_mask].copy()
        outlier_df['predicted_time'] = y_pred[outliers_mask]
        outlier_df['deviation'] = ((outlier_df['Tiempo'] - outlier_df['predicted_time']) / outlier_df['predicted_time'] * 100).round(2)
        outlier_df['studentized_residual'] = studentized_residuals[outliers_mask].round(2)

        # Sort by absolute deviation
        outlier_df['abs_deviation'] = abs(outlier_df['deviation'])
        outlier_df = outlier_df.sort_values('abs_deviation', ascending=False).drop('abs_deviation', axis=1)

        # Make sure Timestamp column exists before displaying
        columns_to_display = ['Programa', 'Tiempo', 'predicted_time', 'deviation', 'studentized_residual', 'Longitude Corte (m)', 'Espesor']
        if 'Timestamp' in outlier_df.columns:
            columns_to_display.append('Timestamp')

        print("\nOutlier Programs Details:")
        # Display as a formatted DataFrame
        display(outlier_df[columns_to_display])

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save outliers to Excel
        outlier_output_path = os.path.join(output_dir, "outlier_programs.xlsx")
        outlier_df.to_excel(outlier_output_path, index=False)
        print(f"\nOutlier details saved to: {outlier_output_path}")

        # Save clean data (without outliers) to Excel
        clean_df = df.loc[~full_mask].copy()
        clean_output_path = os.path.join(output_dir, "clean_full_tiempo_final.xlsx")
        clean_df.to_excel(clean_output_path, index=False)
        print(f"Clean data (without outliers) saved to: {clean_output_path}")

    return full_mask, diagnostics
