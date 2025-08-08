import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_daily(df, date_col, val_col):
    """
    Creates a bar plot showing daily values.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    date_col : str
        Name of the column containing dates
    val_col : str
        Name of the column containing values to plot

    Returns:
    --------
    None
    """
    # Set figure size
    plt.figure(figsize=(12, 6))

    # Create bar plot
    plt.bar(df[date_col], df[val_col], color='skyblue')

    # Add title and labels
    plt.title('Daily Time Usage')
    plt.xlabel('Date')
    plt.ylabel('Time (seconds)')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()

def plot_time(df):
    """
    Creates a line plot showing time values over dates.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data with 'Date' and 'Timestamp' columns

    Returns:
    --------
    None
    """
    # Set figure size
    plt.figure(figsize=(12, 6))

    # Create line plot
    plt.plot(df['Date'], df['Timestamp'], marker='o', linestyle='-', color='blue')

    # Add title and labels
    plt.title('Time Trend')
    plt.xlabel('Date')
    plt.ylabel('Time')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()

def plot_time_usage(time_dict):
    """
    Creates a bar chart comparing machine usage time during day and night periods across multiple dates.

    Parameters:
    -----------
    time_dict : dict
        Dictionary where keys are dates and values are nested dictionaries with 'day' and 'night' usage times

    Returns:
    --------
    None
    """
    # Extract dates and day/night values
    dates = list(time_dict.keys())
    day_values = [time_dict[date]['day'] for date in dates]
    night_values = [time_dict[date]['night'] for date in dates]

    # Set up the figure
    plt.figure(figsize=(12, 6))

    # Set the width of the bars
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(dates))
    r2 = [x + bar_width for x in r1]

    # Create the bars
    bars1 = plt.bar(r1, day_values, width=bar_width, color='blue', label='Day (7:00-21:00)')
    bars2 = plt.bar(r2, night_values, width=bar_width, color='red', label='Night (21:00-7:00)')

    # Add labels on top of each bar
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.1,
                 f'{bar1.get_height():.1f}', ha='center', va='bottom')
        plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.1,
                 f'{bar2.get_height():.1f}', ha='center', va='bottom')

    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Time (seconds)')
    plt.title('Machine Usage Time: Day vs Night')

    # Add xticks on the middle of the group bars
    plt.xticks([r + bar_width/2 for r in range(len(dates))], dates, rotation=45)

    # Add legend
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()