import numpy as np

def remove_outliers(series, n_std=3):
    """Remove outliers from a series using the z-score method."""
    z_scores = np.abs((series - series.mean()) / series.std())
    return series[z_scores < n_std]

def replace_outliers(series, column_name, n_std=3):
    """Replace outliers from a series with the mean value and print information about replacements."""
    # Calculate mean before identifying outliers
    mean_value = series.mean()

    # Identify outliers using z-score method
    z_scores = np.abs((series - mean_value) / series.std())
    outliers_mask = z_scores >= n_std

    # Get the outlier values and their indices
    outlier_values = series[outliers_mask]
    outlier_indices = outlier_values.index

    # Create a copy of the series to avoid modifying the original
    new_series = series.copy()

    # Replace outliers with the mean value
    if len(outlier_values) > 0:
        new_series[outlier_indices] = mean_value

        # Print information about each replacement
        for idx, value in zip(outlier_indices, outlier_values):
            print(f"Row {idx}, Column '{column_name}': Outlier value {value:.2f} replaced with mean {mean_value:.2f}")

    return new_series