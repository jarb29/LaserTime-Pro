import pandas as pd
import numpy as np
import os
from datetime import datetime
from collections import defaultdict


def convert_to_datetime(ts):
    """
    Convert timestamp(s) to datetime object(s).

    Parameters
    ----------
    ts : str, list
        A single timestamp string or a list of timestamp strings.

    Returns
    -------
    datetime or list
        A single datetime object or a list of datetime objects.
    """
    if isinstance(ts, list):
        return [pd.to_datetime(t) for t in ts]
    else:
        return pd.to_datetime(ts)


def add_time_info_columns(df):
    """
    Adds 'Start Time', 'End Time', 'Steps', 'TiempoCorte', and 'TiempoEspera' columns to the DataFrame
    based on the 'Timestamps' column. Drops rows where the number of steps is less than or equal to 1,
    and calculates time differences in hours.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the 'Timestamps' column with lists of timestamps.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with added columns and filtered rows.
    """
    # Ensure the 'Timestamps' column exists
    if 'Timestamps' not in df.columns:
        raise ValueError("The input DataFrame must contain a column named 'Timestamps'")

    # Convert Timestamps to datetime objects if necessary
    df['Timestamps'] = df['Timestamps'].apply(
        lambda x: [pd.to_datetime(ts) for ts in x] if isinstance(x[0], str) else x)

    # Add new columns
    df['Start Time'] = df['Timestamps'].apply(lambda x: x[0] if len(x) > 0 else None)
    df['End Time'] = df['Timestamps'].apply(lambda x: x[-1] if len(x) > 0 else None)
    df['Steps'] = df['Timestamps'].apply(len)
    df['TiempoCorte'] = round((df['End Time'] - df['Start Time']).dt.total_seconds(), 2)

    # Drop rows where the number of steps is less than or equal to 1
    df = df[df['Steps'] > 1].reset_index(drop=True)

    # Add TiempoEspera column and convert to hours
    df['TiempoEspera'] = round((df['Start Time'].shift(-1) - df['End Time']).dt.total_seconds(), 2)

    return df


def group_timestamps_to_dataframe(df, drop_programas=[]):
    """
    Groups the timestamps in the 'Timestamps' column by day and converts
    them into a DataFrame where each row represents a 'Programa', date,
    list of timestamps for that date, and the 'TotalMachine' value.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame that should contain 'Timestamp' column.
        If 'Programa' column is present, it will be used for grouping.
        If 'Timestamps' column is present, it will be used for grouping by day.
        If 'TotalMachine' column is present, it will be used for calculating totals.
        Missing columns will be handled with default values.
    drop_programas : list, optional
        List of 'Programa' values that should be excluded from the DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame representing the grouped data.
    """
    rows = []
    grouped_data = defaultdict(dict)

    # Make a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Check if 'Programa' column exists, if not, add it with a default value
    if 'Programa' not in df.columns:
        df['Programa'] = 'Default Program'
        print("Warning: 'Programa' column not found in DataFrame. Using 'Default Program' as the program name.")

    # Check if 'Timestamps' column exists, if not, create it from 'Timestamp' column
    if 'Timestamps' not in df.columns:
        if 'Timestamp' in df.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])

            # Group by 'Programa' and collect timestamps into lists
            timestamps_by_programa = df.groupby('Programa')['Timestamp'].apply(list).reset_index()

            # Create a mapping from 'Programa' to 'Timestamps'
            timestamps_map = dict(zip(timestamps_by_programa['Programa'], timestamps_by_programa['Timestamp']))

            # Add 'Timestamps' column to the DataFrame
            df['Timestamps'] = df['Programa'].map(timestamps_map)

            # Drop duplicates to keep one row per 'Programa'
            df = df.drop_duplicates(subset=['Programa']).reset_index(drop=True)

            print("Warning: 'Timestamps' column not found in DataFrame. Created from 'Timestamp' column.")
        elif 'machining_seconds' in df.columns:
            # If 'machining_seconds' exists, use it to create a timestamp
            current_time = pd.Timestamp.now()
            df['Timestamps'] = [[current_time]] * len(df)

            # Add 'machining_seconds' to 'TotalMachine' if it doesn't exist
            if 'TotalMachine' not in df.columns:
                df['TotalMachine'] = df['machining_seconds'].apply(lambda x: {current_time.date(): [str(x)]})

            print("Warning: 'Timestamps' column not found in DataFrame. Created from 'machining_seconds' column.")
        else:
            # If neither 'Timestamps', 'Timestamp', nor 'machining_seconds' exists, create a dummy timestamp
            current_time = pd.Timestamp.now()
            df['Timestamps'] = [[current_time]] * len(df)
            print("Warning: Neither 'Timestamps', 'Timestamp', nor 'machining_seconds' column found in DataFrame. Using current time.")

    # Check if 'TotalMachine' column exists, if not, add it with empty dictionaries
    if 'TotalMachine' not in df.columns:
        df['TotalMachine'] = [{} for _ in range(len(df))]
        print("Warning: 'TotalMachine' column not found in DataFrame. Using empty dictionaries.")

    # Filter out rows with 'Programa' in drop_programas list
    filtered_df = df[~df['Programa'].isin(drop_programas)]

    for _, row in filtered_df.iterrows():
        programa = row['Programa']
        timestamps = row['Timestamps']
        total_machine = row['TotalMachine']

        # Ensure timestamps is a list
        if not isinstance(timestamps, list):
            timestamps = [timestamps]

        # Convert timestamps to datetime objects if they are in string format
        if timestamps and isinstance(timestamps[0], str):
            timestamps = convert_to_datetime(timestamps)

        # Initialize a dictionary to store grouped timestamps for this programa
        daily_group = defaultdict(list)

        for ts in timestamps:
            date = ts.date()
            daily_group[date].append(ts)

        # Convert daily_group to dictionary and add rows to the list
        for date, ts_list in daily_group.items():
            # Handle the case when total_machine is not a dictionary
            if isinstance(total_machine, dict):
                total_machine_for_date = total_machine.get(date, [])
            else:
                total_machine_for_date = []

            from src.utils.file_utils import is_numeric_string
            rows.append({
                'Programa': programa,
                'Date': date,
                'Timestamps': ts_list,
                'TM': sum(float(i) for i in total_machine_for_date if is_numeric_string(i))
            })

    return pd.DataFrame(rows)


def compute_total_time(df):
    tiempo_total = df[df['Message'].str.contains('Total machining', case=False, na=False)].copy()

    # Removing text from Messages and extract time values
    tiempo_total['Message'] = tiempo_total['Message'].str.replace('Total machining:', '')
    tiempo_total['Time'] = pd.to_numeric(tiempo_total['Message'].str.extract(r'(\d+\.\d+) ')[0])

    tiempo_total['Time_Hours'] = tiempo_total['Time'] / 3600

    # Attach a dummy year for date-time conversion
    tiempo_total['Timestamp'] = pd.to_datetime(tiempo_total['Timestamp'])
    tiempo_total.set_index('Timestamp', inplace=True)

    # Grouping by month and calculating the total time for each month
    # 'ME' stands for month end frequency
    total_time_by_month = tiempo_total.resample('M').sum()['Time_Hours']

    for index, value in total_time_by_month.items():
        pp = 'Month: {}, Total Hours: {}'.format(index.month, round(value, 2))
        print(pp)

    return total_time_by_month


def group_by_date(df):
    # Filter the dataframe
    tiempo_total = df[df['Message'].str.contains('Total machining', case=False, na=False)].copy()  # Explicit copy here

    # Removing text from Messages and extract time values
    tiempo_total['Message'] = tiempo_total['Message'].str.replace('Total machining:', '')
    tiempo_total['Time'] = pd.to_numeric(tiempo_total['Message'].str.extract(r'(\d+.\d+) ')[0])

    # Extract the date part from 'Timestamp' without adding "2024-"
    tiempo_total['Timestamp'] = pd.to_datetime(tiempo_total['Timestamp'])
    tiempo_total['Date'] = tiempo_total['Timestamp'].dt.date

    # Group by 'Date' and calculate the sum of 'Time' for each date
    grouped_tiempo_total = tiempo_total.groupby('Date')['Time'].sum().reset_index()

    # Divide each row of 'Time' column by 120
    grouped_tiempo_total['Time'] = grouped_tiempo_total['Time'].apply(lambda x: x / 60)

    return grouped_tiempo_total


def time_between_placas(df, filters):
    # Create a boolean Series for filtering
    mask = pd.Series(False, index=df.index)
    for filter in filters:
        mask |= df['Message'].str.contains(filter)

    df_filtered = df[mask]

    final_df = pd.DataFrame()

    # Iterate over the DataFrame
    for i in range(len(df_filtered) - 1):
        # Check if a row contains the first filter term and the next row contains the second filter term
        if filters[0] in df_filtered.iloc[i]['Message'] and filters[1] in df_filtered.iloc[i + 1]['Message']:
            # Append both rows to final_df using pandas concat
            final_df = pd.concat([final_df, df_filtered.iloc[[i, i + 1]]])

    final_df.reset_index(drop=True, inplace=True)

    # Sort the DataFrame by 'Timestamp'
    final_df = final_df.sort_values('Timestamp')

    # Convert 'Timestamp' to datetime format
    final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp'], format='%Y-%m-%d %H:%M:%S')

    # Create a 'Date' column which is the date part of the 'Timestamp'
    final_df['Date'] = final_df['Timestamp'].dt.date

    # Group by the 'Date' column and calculate the difference within each group
    final_df['Timestamp_Diff'] = final_df.groupby('Date')['Timestamp'].diff()

    # Reset the index for the Timestamp_Diff series and drop na
    timestamp_diff_df = final_df[['Date', 'Timestamp_Diff']].dropna().reset_index(drop=True)

    # Drop the rows where 'Timestamp_Diff' is less than 2 minutes
    timestamp_diff_df = timestamp_diff_df[timestamp_diff_df['Timestamp_Diff'] > pd.Timedelta(minutes=2)]

    # Convert 'Timestamp_Diff' from timedelta object to int, showing minutes
    timestamp_diff_df['Timestamp_Diff'] = (timestamp_diff_df['Timestamp_Diff'].dt.total_seconds() / 60).astype(int)

    # Drop the rows where 'Timestamp_Diff' is greater than 600
    timestamp_diff_df = timestamp_diff_df[timestamp_diff_df['Timestamp_Diff'] <= 600]

    return final_df, timestamp_diff_df


def first_occurrence_per_date(df, column, search_str):
    # Filter rows which contain search string
    df_search = df[df[column].str.contains(search_str, case=False, na=False)].copy()

    # Convert 'Timestamp' to datetime
    df_search['Timestamp'] = pd.to_datetime(df_search['Timestamp'], format='%Y-%m-%d %H:%M:%S')

    # Assign current year
    now = datetime.now()
    df_search.loc[:, 'Timestamp'] = df_search['Timestamp'].map(lambda dt: dt.replace(year=now.year))

    # Extract date and time
    df_search['Date'] = [d.date() for d in df_search['Timestamp']]
    df_search['Time'] = [d.time() for d in df_search['Timestamp']]

    # Group by date, drop 'Timestamp' and get the first occurrence per date
    df_first_occurrence = df_search.sort_values('Time').groupby('Date', as_index=False).first()
    df_first_occurrence = df_first_occurrence.drop(columns=['Timestamp'])

    return df_first_occurrence


def average_time(df):
    avg_time_per_day = df['Time'].mean()
    return round(avg_time_per_day, 2)


def average_time_aws(path):
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    result = {}

    for csv_file in csv_files:
        csv_file_path = os.path.join(path, csv_file)
        dataframe = pd.read_csv(csv_file_path)

        grouped_df = group_by_date(dataframe)
        avg = average_time(grouped_df)

        _, _, year, month = csv_file.rstrip('.csv').split('_')
        year_month = f"{year}_{month}"
        result[year_month] = avg

    return result


def summarize_by_date_and_shift(df):
    """
    Summarize machining times by date and shift (day/night) and calculate totals

    Parameters:
    df (pandas.DataFrame): DataFrame with processed machining data

    Returns:
    tuple: (date_dict, total_dict) where:
           - date_dict is a dictionary with dates as keys and day/night time summaries
           - total_dict is a dictionary with overall day/night sums
    """
    # First process the data if it hasn't been processed yet
    if 'Turno' not in df.columns or 'Tiempo' not in df.columns:
        from src.utils.file_utils import filter_open_file_and_machining
        processed_df = filter_open_file_and_machining(df)
    else:
        processed_df = df

    # Convert Tiempo to numeric if it's not already
    processed_df['Tiempo'] = pd.to_numeric(processed_df['Tiempo'])

    # Create a date string column for grouping
    processed_df['DateStr'] = processed_df['Timestamp'].dt.date.astype(str)

    # Initialize result dictionary and totals
    date_dict = {}
    total_day = 0
    total_night = 0

    # Group by date and shift
    for date_str, group in processed_df.groupby('DateStr'):
        day_time = group[group['Turno'] == 'D']['Tiempo'].sum()
        night_time = group[group['Turno'] == 'N']['Tiempo'].sum()

        # Convert to seconds and round to integers
        day_time = int(round(day_time))
        night_time = int(round(night_time))

        # Add to totals
        total_day += day_time
        total_night += night_time

        date_dict[date_str] = {'day': day_time, 'night': night_time}

    # Create the total dictionary
    total_dict = {'day': total_day, 'night': total_night}

    return date_dict, total_dict


def print_day_night_analysis(total_dict):
    """
    Print analysis of day and night machining times with improved ratio explanation
    including hour differences

    Parameters:
    total_dict (dict): Dictionary with 'day' and 'night' keys containing total times in seconds
    """
    # Get day and night times in seconds
    day_seconds = total_dict['day']
    night_seconds = total_dict['night']
    total_seconds = day_seconds + night_seconds

    # Calculate ratio (avoid division by zero)
    ratio = 0 if day_seconds == 0 else round(night_seconds / day_seconds, 2)

    # Convert seconds to hours for difference calculation
    day_hours = day_seconds / 3600
    night_hours = night_seconds / 3600
    hour_diff = abs(day_hours - night_hours)

    # Convert seconds to hours, minutes, seconds format for display
    def seconds_to_hms(seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs}s"

    # Calculate percentages
    day_percent = 0 if total_seconds == 0 else round((day_seconds / total_seconds) * 100, 1)
    night_percent = 0 if total_seconds == 0 else round((night_seconds / total_seconds) * 100, 1)

    # Determine ratio interpretation with hour differences
    if ratio == 0:
        ratio_explanation = "No night work recorded"
    elif ratio < 0.5:
        ratio_explanation = f"Day shift dominates production by {round(hour_diff, 1)} hours (more than 2x night shift)"
    elif ratio < 0.9:
        ratio_explanation = f"Day shift has higher production by {round(hour_diff, 1)} hours than night shift"
    elif ratio < 1.1:
        ratio_explanation = f"Day and night shifts have similar production levels (only {round(hour_diff, 1)} hours difference)"
    elif ratio < 2:
        ratio_explanation = f"Night shift has higher production by {round(hour_diff, 1)} hours than day shift"
    else:
        ratio_explanation = f"Night shift dominates production by {round(hour_diff, 1)} hours (more than 2x day shift)"

    # Print explanation
    print(f"Day shift (7:00-21:00): {seconds_to_hms(day_seconds)} ({day_percent}% of total)")
    print(f"Night shift (21:00-7:00): {seconds_to_hms(night_seconds)} ({night_percent}% of total)")
    print(f"Total machining time: {seconds_to_hms(total_seconds)}")
    print(f"Night-to-day ratio: {ratio}")
    print(f"Interpretation: {ratio_explanation}")
