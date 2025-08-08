import pandas as pd
import os
import json
import re
import codecs
from datetime import datetime
from pathlib import Path

def setup_project_paths(directory_path=None, save_folder=None):
    """Set up project paths and create save folder if needed."""


    if directory_path is None or save_folder is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if directory_path is None:
            directory_path = os.path.join(project_root, 'data', 'raw', 'logs')
        if save_folder is None:
            save_folder = os.path.join(project_root, 'data', 'processed')

    os.makedirs(save_folder, exist_ok=True)
    return directory_path, save_folder


def load_processed_files(save_folder):
    """Load the list of previously processed files and return both the files set and filename.

    This function manages the tracking of processed files by loading previously processed files
    from a JSON file and preparing the path for future updates.

    Parameters
    ----------
    save_folder : str
        Path to the directory where the processed files list is stored

    Returns
    -------
    tuple
        A tuple containing two elements:
        - processed_files (set): A set containing the names of files that have already been
          processed. Used to:
          * Track which files have been processed to avoid duplicate processing
          * Maintain state between program runs
          * Use for filtering new files that need processing

        - processed_files_filename (str): The full path to the JSON file where the list of
          processed files is stored. Used to:
          * Save the updated list of processed files after new files are processed
          * Maintain consistency between program runs
          * Reference the same file location for both reading and writing

    Notes
    -----
    If the JSON file cannot be read, a warning is printed and an empty set is returned
    along with the filename.
    """
    processed_files = set()
    processed_files_filename = os.path.join(save_folder, 'processed_files.json')

    if os.path.exists(processed_files_filename):
        try:
            with open(processed_files_filename, 'r') as f:
                processed_files = set(json.load(f))
        except json.JSONDecodeError:
            print("Warning: Could not read processed_files.json, starting fresh")

    return processed_files, processed_files_filename

def load_existing_data(save_folder):
    """Load existing processed data if available."""


    final_df = pd.DataFrame()
    final_df_filename = os.path.join(save_folder, 'processed_data.csv')

    if os.path.isfile(final_df_filename):
        try:
            final_df = pd.read_csv(final_df_filename)
            if not final_df.empty:
                final_df['Timestamp'] = pd.to_datetime(final_df['Timestamp'])
        except Exception as e:
            print(f"Warning: Could not read existing processed data: {e}")

    return final_df, final_df_filename



def get_rtf_files_to_process(directory_path, processed_files):
    try:
        files = [f for f in os.listdir(directory_path)
                 if os.path.isfile(os.path.join(directory_path, f)) and
                 f.endswith('.rtf') and f not in processed_files]

        print(f"\nFound {len(files)} RTF files to process")
        file_dates = []

        for filename in files:
            try:
                print(f"\nProcessing file: {filename}")
                # Extract year from filename
                year_match = re.search(r'20(24|25|26)', filename)
                if not year_match:
                    print(f"No valid year found in filename '{filename}'")
                    continue

                year = int('20' + year_match.group(1))
                # print(f"Extracted year: {year}")

                # Read file to find timestamp
                file_path = os.path.join(directory_path, filename)
                # print(f"Reading file: {file_path}")

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Read first 1000 characters
                    # print(f"File content (first 100 chars): {content[:100]}")

                    # Look for timestamp pattern (MM/DD HH:MM:SS) - Note: The pattern in file is already MM/DD
                    timestamp_pattern = r'\((\d{2})/(\d{2})\s+(\d{2}:\d{2}:\d{2})\)'
                    timestamp_match = re.search(timestamp_pattern, content)

                    if timestamp_match:
                        month, day, time = timestamp_match.groups()  # First number is month, second is day
                        print(f"Found timestamp parts - Month: {month}, Day: {day}, Time: {time}")

                        # Create datetime string in correct order (YYYY-MM-DD)
                        datetime_str = f"{year}-{month}-{day} {time}"
                        print(f"Created datetime string: {datetime_str}")

                        try:
                            file_date = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                            print(f"Successfully parsed date: {file_date}")
                            file_dates.append((filename, file_date))
                        except ValueError as e:
                            print(f"Error parsing date from file '{filename}': {e}")
                            print(f"Raw datetime string was: {datetime_str}")

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue

        print(f"\nSuccessfully processed {len(file_dates)} files")
        # Sort by datetime and return
        return sorted(file_dates, key=lambda x: x[1])

    except Exception as e:
        print(f"Error accessing directory {directory_path}: {e}")
        return []


def process_rtf_file(filename, directory_path, final_df_filename):

    """Process a single RTF file."""


    try:
        # Extract year from filename
        year_match = re.search(r'20(24|25|26)', filename)
        if not year_match:
            raise ValueError(f"File {filename} does not contain a valid year (2024-2026)")

        file_year = year_match.group(0)
        full_path = os.path.join(directory_path, filename)

        df = rtf_to_dataframe(full_path) # only transfor the data into df

        if not df.empty:
            # Add timestamp using year from filename
            df['Timestamp'] = pd.to_datetime(file_year + df['Timestamp'],
                                             format='%Y%m/%d %H:%M:%S')

            # Filter data
            df = filter_open_file_and_machining(df)

            # Read existing data if it exists
        if os.path.exists(final_df_filename):
            existing_df = pd.read_csv(final_df_filename)
            existing_df['Timestamp'] = pd.to_datetime(existing_df['Timestamp'])

                # Concatenate new data with existing data
            final_df = pd.concat([existing_df, df], ignore_index=True)

                # Remove duplicates and sort
            final_df = final_df.drop_duplicates(subset=['Timestamp', 'Message'], keep='last')
            final_df = final_df.sort_values('Timestamp')
        else:
            final_df = df

            # Save updated data
        final_df.to_csv(final_df_filename, index=False)

        return True


    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

    return pd.DataFrame()


def save_processed_data(filename, processed_files, save_folder):
    """Add a single processed file to the list and save it.

    Args:
        filename: Name of the file to add to the processed list
        processed_files: Set of already processed files
        save_folder: Path to the directory where the processed files list is stored
    """
    try:
        processed_files_filename = os.path.join(save_folder, 'processed_files.json')

        # Ensure the directory exists
        os.makedirs(os.path.dirname(processed_files_filename), exist_ok=True)

        # Add the new filename
        processed_files.add(filename)

        # Save the updated list
        with open(processed_files_filename, 'w', encoding='utf-8') as f:
            json.dump(list(processed_files), f)

    except PermissionError as e:
        print(f"Error: No permission to write to {processed_files_filename}: {e}")
    except FileNotFoundError as e:
        print(f"Error: Directory not found or cannot be created: {e}")
    except Exception as e:
        print(f"Error saving processed files list: {e}")



def rtf_to_dataframe(file_path):
    """
    Convert a log file to a pandas DataFrame.
    """
    # print(f"\nProcessing log file: {file_path}")

    # Initialize an empty dataframe
    df = pd.DataFrame(columns=["Timestamp", "Message"])

    try:
        # Read file
        with codecs.open(file_path, "r", "utf-8") as file:
            content = file.read()
            # print(f"File content length: {len(content)} characters")

            # Split by newlines
            lines = content.split('\n')
            # print(f"Number of lines: {len(lines)}")

            # Regex pattern for timestamp and message
            # Format: (MM/DD HH:MM:SS)Message
            line_pattern = r'\((\d{2}/\d{2} \d{2}:\d{2}:\d{2})\)(.*)'

            matches_found = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                match = re.match(line_pattern, line)
                if match:
                    timestamp = match.group(1)
                    message = match.group(2).strip()
                    df.loc[len(df)] = [timestamp, message]
                    matches_found += 1

            #         # Log first few matches for debugging
            #         if matches_found <= 3:
            #             print(f"\nMatch {matches_found}:")
            #             print(f"Timestamp: {timestamp}")
            #             print(f"Message: {message}")
            #
            # print(f"\nTotal matches found: {matches_found}")
            # print(f"Final DataFrame shape: {df.shape}")

            # if not df.empty:
            #     print("\nFirst few rows of final DataFrame:")
            #     print(df.head())
            #     print("\nData types of columns:")
            #     print(df.dtypes)
            # else:
            #     print("\nWARNING: No data was extracted from the file!")

    except Exception as e:
        print(f"\nError processing file: {str(e)}")
        print(f"Error type: {type(e)}")
        raise

    return df



def filter_open_file_and_machining(df, project_root=None):
    """
    Filter DataFrame to keep only rows where Message contains 'Open File:' or 'Total machining',
    and clean up RTF formatting from the messages.
    """
    # Create mask for rows containing either 'Open File:' or 'Total machining'
    mask = df['Message'].str.contains('Open File:|Total machining', case=False, na=False)
    filtered_df = df[mask].copy()

    # Clean up Total machining messages - remove RTF formatting
    mask_machining = filtered_df['Message'].str.contains('Total machining:', case=False, na=False)
    filtered_df.loc[mask_machining, 'Message'] = filtered_df.loc[mask_machining, 'Message'].apply(
        lambda x: re.search(r'Total machining: \d+\.?\d*', x).group(0) if re.search(r'Total machining: \d+\.?\d*',
                                                                                    x) else x
    )

    # Process timestamps and add columns
    filtered_df['Timestamp'] = pd.to_datetime(filtered_df['Timestamp'])
    filtered_df['Turno'] = filtered_df['Timestamp'].apply(lambda x: 'D' if 7 <= x.hour < 21 else 'N')
    filtered_df['Year'] = filtered_df['Timestamp'].dt.year
    filtered_df['Month'] = filtered_df['Timestamp'].dt.month
    filtered_df['Day'] = filtered_df['Timestamp'].dt.day

    # Sort by Timestamp
    filtered_df = filtered_df.sort_values('Timestamp')

    # Get the last open file from processed data if available
    last_open_file = None
    if project_root:
        processed_data_path = os.path.join(project_root, 'data', 'processed', 'processed_data.csv')
        if os.path.exists(processed_data_path):
            try:
                processed_df = pd.read_csv(processed_data_path)
                # Get the last non-null CurrentFile value
                if 'CurrentFile' in processed_df.columns:
                    last_files = processed_df['CurrentFile'].dropna()
                    if not last_files.empty:
                        last_open_file = last_files.iloc[-1]
            except Exception as e:
                print(f"Error reading processed data: {e}")

    # Track current file
    file_column = []
    current_file = last_open_file  # Initialize with last known file

    # Process files and machining times
    for _, row in filtered_df.iterrows():
        message = row['Message']
        if 'Open File:' in message and 'autosave.slp' not in message and 'runner.slp' not in message:
            file_match = re.search(r'Open File: (.*?)(?:\\cf0|$)', message)
            if file_match:
                current_file = file_match.group(1).strip()
        file_column.append(current_file)

    filtered_df['CurrentFile'] = file_column

    # Process only Total machining rows with valid files
    result_df = filtered_df[
        (filtered_df['Message'].str.contains('Total machining')) &
        (filtered_df['CurrentFile'].notna())
    ].copy()

    # Extract program name from CurrentFile
    # result_df['Programa'] = result_df['CurrentFile'].str.extract(r'\\([^\\]+\.[nN][cC])')
    result_df['Programa'] = result_df['CurrentFile'].str.extract(r'\\([^\\]+$)')

    # Extract time from Total machining message
    result_df['Tiempo'] = result_df['Message'].str.extract(r'Total machining: (\d+\.?\d*)')
    result_df['Tiempo'] = pd.to_numeric(result_df['Tiempo'])

    return result_df

def get_last_current_file(project_root=None):
    """
    Get the last CurrentFile from the processed data CSV file.

    Parameters:
    project_root (str or Path, optional): Project root directory. If None, will attempt to find it.

    Returns:
    str: The last CurrentFile value, or None if not found
    """
    try:
        # If project_root not provided, try to find it
        if project_root is None:
            project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        elif isinstance(project_root, str):
            project_root = Path(project_root)

        # Construct path to processed data file
        processed_file = project_root / 'data' / 'processed' / 'processed_data.csv'

        # Check if file exists
        if not processed_file.exists():
            print(f"Warning: No processed data file found at {processed_file}")
            return None

        # Read the CSV file
        df = pd.read_csv(processed_file)

        # Check if CurrentFile column exists
        if 'CurrentFile' not in df.columns:
            print("Warning: No CurrentFile column in processed data")
            return None

        # Get last non-null CurrentFile
        last_file = df['CurrentFile'].dropna().iloc[-1] if not df['CurrentFile'].dropna().empty else None

        return last_file

    except Exception as e:
        print(f"Error reading last current file: {e}")
        return None




if __name__ == "__main__":
    # Example usage: Read all data from June 2024 to current date
    # combined_data = read_all_rtf_in_di
    print('Helklo')


