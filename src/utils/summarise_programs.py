## Program Summarization by Period
import pandas as pd


def summarise_programs(df, year=None, month=None):
    # Filter by year and month if provided
    if year is not None and month is not None:
        df = df[(df['Year'] == year) & (df['Month'] == month)]
    elif year is not None:
        df = df[df['Year'] == year]
    elif month is not None:
        df = df[df['Month'] == month]

    # Sort the dataframe by timestamp first to ensure chronological order
    # Ensure Timestamp is in datetime format
    df_sorted = df.copy()
    df_sorted['Timestamp'] = pd.to_datetime(df_sorted['Timestamp'])
    df_sorted = df_sorted.sort_values('Timestamp')

    # Start by grouping df by 'Programa' and summarizing 'Tiempo'
    program_summary = df_sorted.groupby('Programa').agg({'Tiempo': 'sum'})

    # Then find the start and end time for each 'Programa'
    program_times = df_sorted.groupby('Programa').agg(
        Hora_Inicio=pd.NamedAgg(column='Timestamp', aggfunc='min'),
        Hora_Final=pd.NamedAgg(column='Timestamp', aggfunc='max')
    )

    # Join the summary dataframes together
    result = pd.concat([program_summary, program_times], axis=1)

    # Calculate real time (duration) for each program
    # Ensure both columns are datetime type before subtraction
    result['Tiempo_Real'] = (pd.to_datetime(result['Hora_Final']) -
                             pd.to_datetime(result['Hora_Inicio'])) / pd.Timedelta(hours=1)

    # Create a chronological list of programs with their start/end times
    chrono_programs = []
    for prog, start, end in zip(result.index, result['Hora_Inicio'], result['Hora_Final']):
        chrono_programs.append({'Programa': prog, 'Start': start, 'End': end})

    # Sort by start time to get chronological order
    chrono_programs.sort(key=lambda x: x['Start'])

    # Calculate downtime between consecutive programs, excluding weekend gaps
    downtime = {}
    for i in range(len(chrono_programs) - 1):
        current_prog = chrono_programs[i]['Programa']
        next_prog = chrono_programs[i + 1]['Programa']
        current_end = chrono_programs[i]['End']
        next_start = chrono_programs[i + 1]['Start']

        # Check if current end time is in weekend off-hours (Friday 16:00 to Sunday 21:00)
        is_weekend_off_hours = False

        # Friday after 16:00
        if current_end.weekday() == 4 and current_end.hour >= 16:
            is_weekend_off_hours = True
        # All day Saturday
        elif current_end.weekday() == 5:
            is_weekend_off_hours = True
        # Sunday before 21:00
        elif current_end.weekday() == 6 and current_end.hour < 21:
            is_weekend_off_hours = True

        # Skip weekend downtime calculation if in off-hours
        if is_weekend_off_hours:
            continue

        # Calculate downtime in hours
        downtime_hours = (next_start - current_end).total_seconds() / 3600
        downtime[current_prog] = downtime_hours

    # Add downtime to results
    result['Tiempo_detenido'] = pd.Series(downtime)

    # The last program won't have downtime
    result['Tiempo_detenido'] = result['Tiempo_detenido'].fillna(0)

    return result