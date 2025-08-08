## Data Normalization
def update_missing_values(merged_df, laser_close_df):
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    merged_df = merged_df.copy()

    # Create a clean program column for matching - remove .NC and leading zeros
    merged_df['program_clean'] = merged_df['Programa'].str.replace('.NC', '', case=False)
    merged_df['program_clean'] = merged_df['program_clean'].str.lstrip('0')

    # Create normalized version of cnc column in laser_close_df
    laser_close_df = laser_close_df.copy()
    laser_close_df['cnc_clean'] = laser_close_df['cnc'].astype(str).str.lstrip('0')

    # Convert Decimal values to float in laser_close_df
    laser_close_df['metros'] = laser_close_df['metros'].astype(float)
    laser_close_df['espesor'] = laser_close_df['espesor'].astype(float)

    # Create mapping dictionaries from laser_close_df with normalized keys
    metros_map = dict(zip(laser_close_df['cnc_clean'], laser_close_df['metros']))
    espesor_map = dict(zip(laser_close_df['cnc_clean'], laser_close_df['espesor']))

    # Find rows with NaN values in the target columns
    # First, ensure these columns exist (create them if they don't)
    if 'Longitude Corte (m)' not in merged_df.columns:
        merged_df['Longitude Corte (m)'] = None
    if 'Espesor' not in merged_df.columns:
        merged_df['Espesor'] = None

    # Find rows with NaN values
    nan_mask = merged_df[['Longitude Corte (m)', 'Espesor']].isna().any(axis=1)

    # Update values where matches exist
    merged_df.loc[nan_mask, 'Longitude Corte (m)'] = (
        merged_df.loc[nan_mask, 'program_clean']
        .map(metros_map)
        .astype(float)
    )

    merged_df.loc[nan_mask, 'Espesor'] = (
        merged_df.loc[nan_mask, 'program_clean']
        .map(espesor_map)
        .astype(float)
    )

    # Remove rows where no match was found (values remained NaN)
    merged_df = merged_df.dropna(subset=['Longitude Corte (m)', 'Espesor'])

    # Remove the temporary clean program column
    merged_df = merged_df.drop('program_clean', axis=1)

    return merged_df