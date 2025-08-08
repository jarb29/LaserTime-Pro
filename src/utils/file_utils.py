import codecs
import numpy as np
import re
import os
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict


def rtf_to_dataframe(file_path):
    """
    Convert an RTF file to a pandas DataFrame.

    This function reads an RTF file, extracts timestamps and messages using regex patterns,
    and returns a DataFrame with 'Timestamp' and 'Message' columns.

    Parameters
    ----------
    file_path : str
        Path to the RTF file to be processed.

    Returns
    -------
    pandas.DataFrame
        DataFrame with 'Timestamp' and 'Message' columns containing the extracted data.
    """
    # Initialize an empty dataframe
    df = pd.DataFrame(columns=["Timestamp", "Message"])

    # Read RTF file
    with codecs.open(file_path, "r", "utf-8") as file:
        content = file.read()

        # Split the file content into lines
        rows = content.split("\\par")

        # Regex pattern for timestamp and message
        timestamp_pattern = r'\(([0-9]{2}/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})\)'
        message_pattern = r'\\cf[0-9](.*?)\\cf0'  # Escape the backslashes

        for row in rows:
            # Find timestamp and message
            timestamp = re.search(timestamp_pattern, row)
            message = re.search(message_pattern, row)

            # Append row to dataframe
            if timestamp and message:
                df.loc[len(df)] = [timestamp.group(1), message.group(1)]

    return df


def read_all_rtf_in_dir(directory_path, save_folder, year, month):
    """
    Read all RTF files in a directory, process them, and save the results as CSV files.

    This function reads all RTF files in the specified directory, converts them to
    DataFrames, keeps track of processed files to avoid reprocessing, and saves the
    processed data to CSV files organized by month.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing RTF files to process.
    save_folder : str
        Path to the directory where processed data will be saved.
    year : str or int
        Year for filtering the returned data.
    month : str or int
        Month for filtering the returned data.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the processed data for the specified year and month.
        Returns an empty DataFrame if no data is found.
    """
    data_year = str(datetime.today().year)  # Get the current year for data 'timestamp'.

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    processed_files = set()
    processed_files_filename = f'{save_folder}/processed_files.json'
    if os.path.exists(processed_files_filename):
        with open(processed_files_filename, 'r') as f:
            processed_files = set(json.load(f))

    files = [f for f in os.listdir(directory_path)
             if os.path.isfile(os.path.join(directory_path, f)) and f.endswith('.rtf') and f not in processed_files]

    # If there are no new files, then no new processing is required
    if len(files) == 0:
        print('No new files to process.')

        result_df_filename = f'{save_folder}/saved_df_{year}_{month}.csv'
        if os.path.isfile(result_df_filename):
            return pd.read_csv(result_df_filename)
        else:
            return pd.DataFrame()

    dfs = []
    for num, filename in enumerate(files):
        pp = '{}/{} : {}'.format(num, len(files), filename)
        print(pp)
        full_path = os.path.join(directory_path, filename)
        df = rtf_to_dataframe(full_path)

        df['Timestamp'] = pd.to_datetime(data_year + df['Timestamp'], format='%Y%m/%d %H:%M:%S')

        processed_files.add(filename)

        dfs.append(df)

    with open(processed_files_filename, 'w') as f:
        json.dump(list(processed_files), f)

    if len(dfs) > 0:
        new_data = pd.concat(dfs, ignore_index=True)
    else:
        print("No data to process.")
        new_data = pd.DataFrame()

    # Group new data by month
    groupby_month_new_data = new_data.groupby(new_data['Timestamp'].dt.month)

    for m, group in groupby_month_new_data:
        monthly_filename = f'{save_folder}/saved_df_{data_year}_{m}.csv'

        df_existing = pd.read_csv(monthly_filename) if os.path.isfile(monthly_filename) else pd.DataFrame()
        if not df_existing.empty:
            df_existing['Timestamp'] = pd.to_datetime(df_existing['Timestamp'])
        df_month = pd.concat([df_existing, group], ignore_index=True)
        df_month.to_csv(monthly_filename, index=False)

    # Return the DataFrame for the specified month and year
    result_df_filename = f'{save_folder}/saved_df_{year}_{month}.csv'
    print(result_df_filename, 'result_df_filename')
    if os.path.isfile(result_df_filename):
        return pd.read_csv(result_df_filename)
    else:
        return pd.DataFrame()


# Extract the number from 'Total machining' message
def extract_number(message):
    """
    Extract a floating-point number from a message containing 'Total machining:'.

    Parameters
    ----------
    message : str
        The message to extract the number from.

    Returns
    -------
    float
        The extracted number, or np.nan if no match is found.
    """
    match = re.search(r'Total machining: (\d+\.\d+)', message)
    return float(match.group(1)) if match else np.nan


def filter_open_file_and_machining(df):
    """
    Filter DataFrame to keep only rows where Message contains 'Open File:' or 'Total machining',
    pair each 'Total machining' with its corresponding 'Open File', and extract program name and time

    Parameters:
    df (pandas.DataFrame): DataFrame with 'Message' and 'Timestamp' columns

    Returns:
    pandas.DataFrame: Processed DataFrame with additional columns
    """
    # Create mask for rows containing either 'Open File:' or 'Total machining'
    mask = df['Message'].str.contains('Open File:|Total machining', case=False, na=False)

    # Filter DataFrame
    filtered_df = df[mask].copy()

    # Convert Timestamp to datetime if it's not already
    filtered_df['Timestamp'] = pd.to_datetime(filtered_df['Timestamp'])

    # Add Turno column based on hour
    filtered_df['Turno'] = filtered_df['Timestamp'].apply(lambda x: 'D' if 7 <= x.hour < 21 else 'N')

    # Add separate date component columns
    filtered_df['Year'] = filtered_df['Timestamp'].dt.year
    filtered_df['Month'] = filtered_df['Timestamp'].dt.month
    filtered_df['Day'] = filtered_df['Timestamp'].dt.day

    # Sort by Timestamp
    filtered_df = filtered_df.sort_values('Timestamp')

    # Add a column to track the current file
    current_file = None
    file_column = []

    for _, row in filtered_df.iterrows():

        if 'Open File:' in row['Message'] and 'autosave.slp' not in row['Message'] and 'runner.slp' not in row['Message']:

            current_file = row['Message']
        file_column.append(current_file)

    filtered_df['CurrentFile'] = file_column

    # Keep only rows with 'Total machining' and their file information
    result_df = filtered_df[filtered_df['Message'].str.contains('Total machining')].copy()

    # Extract program name (filename after last backslash)
    result_df['Programa'] = result_df['CurrentFile'].str.extract(r'\\([^\\]+)$')

    # Extract machining time
    result_df['Tiempo'] = result_df['Message'].str.extract(r'Total machining: (\d+\.\d+)')
    result_df['Tiempo'] = pd.to_numeric(result_df['Tiempo'])

    return result_df


def is_numeric_string(s):
    try:
        float(s)
        return True
    except ValueError:
        return False