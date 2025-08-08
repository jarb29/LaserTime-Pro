## Day/Night Distribution
import pandas as pd
from datetime import timedelta
def process_machining_times_by_period(df, year=None, month=None):
    """
    Process machining times to calculate delay and day/night distribution with period filtering.

    Parameters:
    df (pandas.DataFrame): DataFrame with columns ['Tiempo', 'Hora_Inicio', 'Hora_Final',
                         'Tiempo_Real', 'Tiempo_detenido'] and 'Programa' as index
    year (int, optional): Year to filter for (e.g., 2025)
    month (int, optional): Month to filter for (1-12)

    Returns:
    pandas.DataFrame: DataFrame with columns ['Tiempo_Real', 'Tiempo_detenido', 'Tiempo_Retraso', 'D', 'N', 'Year', 'Month']
    """
    # Create a copy and convert timestamps
    df = df.copy()
    df['Hora_Inicio'] = pd.to_datetime(df['Hora_Inicio'])
    df['Hora_Final'] = pd.to_datetime(df['Hora_Final'])

    # Extract year and month before filtering
    df['Year'] = df['Hora_Inicio'].dt.year
    df['Month'] = df['Hora_Inicio'].dt.month

    # Apply year and month filters if specified
    if year is not None:
        df = df[df['Year'] == year]
    if month is not None:
        df = df[df['Month'] == month]

    # Check if dataframe is empty after filtering
    if df.empty:
        print(f"No data found for period: Year={year}, Month={month}")
        return pd.DataFrame()

    # Convert Tiempo from seconds to hours
    df['Tiempo_horas'] = df['Tiempo'] / 3600


    # Calculate Tiempo Retraso
    df['Tiempo_Retraso'] = abs(df['Tiempo_Real'] - df['Tiempo_horas'])

    # Initialize D and N columns
    df['D'] = 0.0
    df['N'] = 0.0

    # Process each row

    # Process each row
    for idx, row in df.iterrows():
        start_time = row['Hora_Inicio']
        end_time = row['Hora_Final']
        current_time = start_time
        day_hours = 0
        night_hours = 0

        while current_time < end_time:
            hour = current_time.hour
            next_time = min(current_time + timedelta(hours=1), end_time)

            # If the interval crosses a shift boundary, split it
            if hour < 21 and next_time.hour >= 21:  # Day to night transition
                transition_time = current_time.replace(hour=21, minute=0, second=0)
                # Calculate day portion
                day_portion = (transition_time - current_time).total_seconds() / 3600
                day_hours += day_portion
                # Calculate night portion
                night_portion = (next_time - transition_time).total_seconds() / 3600
                night_hours += night_portion
            elif hour < 7 and next_time.hour >= 7:  # Night to day transition
                transition_time = current_time.replace(hour=7, minute=0, second=0)
                # Calculate night portion
                night_portion = (transition_time - current_time).total_seconds() / 3600
                night_hours += night_portion
                # Calculate day portion
                day_portion = (next_time - transition_time).total_seconds() / 3600
                day_hours += day_portion
            else:
                time_diff = (next_time - current_time).total_seconds() / 3600
                if 7 <= hour < 21:  # Day shift (7:00-21:00)
                    day_hours += time_diff
                else:  # Night shift
                    night_hours += time_diff

            current_time = next_time

        df.at[idx, 'D'] = day_hours
        df.at[idx, 'N'] = night_hours

    # Select and reorder final columns
    final_columns = ['Tiempo_horas', 'Hora_Inicio','Tiempo_Real', 'Tiempo_detenido', 'Tiempo_Retraso', 'D', 'N', 'Year', 'Month']
    if 'Espesor' in df.columns:
        final_columns.insert(0, 'Espesor')
    df = df[final_columns]

    # Calculate and print summary statistics
    stats = {
        'Total_Day_Hours': df['D'].sum(),
        'Total_Night_Hours': df['N'].sum(),
        'Total_Hours_cutting':df['Tiempo_horas'].sum(),
        'Average_Delay': df['Tiempo_Retraso'].mean(),
        'Records_Count': len(df)
    }

    print(f"\nResumen del Período (Año: {year}, Mes: {month}):")
    print(f"Retraso Promedio: {stats['Average_Delay']:.2f} horas")
    print(f"Número de Registros: {stats['Records_Count']}")
    print(f"Número de Horas de Corte: {stats['Total_Hours_cutting']:.2f} horas")
    print(f"Total de Horas Diurnas considerando el retraso o error por corte: {stats['Total_Day_Hours']:.2f}")
    print(f"Total de Horas Nocturnas considerando el retraso o error por corte: {stats['Total_Night_Hours']:.2f}")
    return df