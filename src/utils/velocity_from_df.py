## Velocity Calculation
import pandas as pd
def velocity_from_df(df):
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Convert 'timestamp' to datetime format if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Create 'year' and 'month' columns
    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month

    # Group by 'year', 'month', and 'Espesor' and calculate the sum
    df_grouped = df.groupby(['year', 'month', 'Espesor'])[['Tiempo', 'Longitude Corte (m)']].sum().reset_index()

    # Calculate velocity with safe division
    df_grouped['Velocidad (m/min)'] = df_grouped.apply(
        lambda row: round((row['Longitude Corte (m)']) / row['Tiempo']*60, 2)
        if row['Tiempo'] > 0 else 0,
        axis=1
    )

    # Building velocity per month
    result = {}

    for _, row in df_grouped.iterrows():
        year_month = f"{int(row['year'])}_{int(row['month'])}"
        espesor = str(int(row['Espesor']))
        velocidad = row['Velocidad (m/min)']

        if year_month not in result:
            result[year_month] = {}

        result[year_month][espesor] = velocidad

    return result

