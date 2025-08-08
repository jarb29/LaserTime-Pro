import pandas as pd
## Downtime Calculation
def summarise_programsT(df):
    """
    Summarize program data with chronologically ordered results.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing program data with columns 'Timestamp', 'Programa', and 'Tiempo'

    Returns:
    pandas.DataFrame
        Summarized data with program statistics, ordered by start time
    """
    # Sort the dataframe by timestamp first to ensure chronological order
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df_sorted = df.sort_values('Timestamp')

    # Start by grouping df by 'Programa' and summarizing 'Tiempo'
    program_summary = df_sorted.groupby('Programa').agg({'Tiempo': 'sum',
                                                         'Espesor': 'first'
})

    # Then find the start and end time for each 'Programa'
    program_times = df_sorted.groupby('Programa').agg(
        Hora_Inicio=pd.NamedAgg(column='Timestamp', aggfunc='min'),
        Hora_Final=pd.NamedAgg(column='Timestamp', aggfunc='max')
    )

    # Join the summary dataframes together
    result = pd.concat([program_summary, program_times], axis=1)

    # Calculate real time (duration) for each program
    result['Tiempo_Real'] = result['Hora_Final'] - result['Hora_Inicio']
    result['Tiempo_Real'] = result['Tiempo_Real'] / pd.Timedelta(hours=1)

    # Create a chronological list of programs with their start/end times
    chrono_programs = []
    for prog, start, end in zip(result.index, result['Hora_Inicio'], result['Hora_Final']):
        chrono_programs.append({'Programa': prog, 'Start': start, 'End': end})

    # Sort by start time to get chronological order
    chrono_programs.sort(key=lambda x: x['Start'])

    # Calculate downtime between consecutive programs, excluding weekend gaps
    downtime = {}
    for i in range(len(chrono_programs)-1):
        current_prog = chrono_programs[i]['Programa']
        next_prog = chrono_programs[i+1]['Programa']
        current_end = chrono_programs[i]['End']
        next_start = chrono_programs[i+1]['Start']

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
            downtime[current_prog] = pd.Timedelta(hours=0)
            continue

        # Calculate time between end of current program and start of next
        time_between = next_start - current_end
        downtime[current_prog] = time_between

    # Add last program with zero downtime
    if chrono_programs:
        downtime[chrono_programs[-1]['Programa']] = pd.Timedelta(hours=0)

    # Add downtime to result dataframe
    result['Tiempo_detenido'] = pd.Series(downtime)

    # Convert to hours
    result['Tiempo_detenido'] = result['Tiempo_detenido'] / pd.Timedelta(hours=1)

    # Sort the final result by start time
    result = result.sort_values('Hora_Inicio')

    return result