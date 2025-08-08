# Machining Time Analysis Project

A Python project for analyzing machining time data from laser cutting machines.

## Overview

This project provides tools for loading, processing, analyzing, and visualizing machining time data from laser cutting machines. It includes functionality for:

- Loading and processing historical machining data
- Analyzing machining times, downtime, and performance metrics
- Detecting outliers and anomalies in machining data
- Creating comprehensive visualizations and dashboards
- Analyzing material properties (thickness, length) and their impact on machining time
- Calculating and visualizing cutting velocities
- Interacting with AWS DynamoDB for data storage and retrieval

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab (for running the notebooks)

### Install from source

1. Clone the repository:
   ```bash
   git clone https://github.com/kupfer/machining-time-analysis.git
   cd machining-time-analysis
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up AWS credentials (if using DynamoDB features):
   - Configure your AWS credentials using the AWS CLI:
     ```bash
     aws configure --profile kubeprofile
     ```
   - Or manually create credentials in `~/.aws/credentials`:
     ```
     [kubeprofile]
     aws_access_key_id = YOUR_ACCESS_KEY
     aws_secret_access_key = YOUR_SECRET_KEY
     region = us-east-1
     ```

## Project Structure

The project is organized into the following structure:

```
machining-time-analysis/
│
├── README.md                  # Project documentation
├── requirements.txt           # Project dependencies
│
├── notebooks/                 # Jupyter notebooks
│   ├── analysis/              # Analysis notebooks
│   │   ├── FullProgramTime.ipynb     # Main analysis notebook
│   │   ├── informe.ipynb             # Report notebook
│   │   └── planificador_plant.ipynb  # Plant planning notebook
│   │
│   └── exploratory/           # Exploratory data analysis notebooks
│       └── data_exploration.ipynb    # For initial data exploration
│
├── src/                       # Source code
│   ├── __init__.py            # Makes src a Python package
│   │
│   ├── data/                  # Data processing modules
│   │   ├── __init__.py
│   │   ├── loader.py          # Data loading functions
│   │   ├── processor.py       # Data processing functions
│   │   └── cleaner.py         # Data cleaning functions
│   │
│   ├── visualization/         # Visualization modules
│   │   ├── __init__.py
│   │   ├── program_plots.py   # Program comparison plots
│   │   ├── downtime_plots.py  # Downtime analysis plots
│   │   ├── dashboard.py       # Dashboard visualization
│   │   ├── time_range.py      # Time range plots
│   │   ├── velocity.py        # Velocity plots
│   │   └── material.py        # Material analysis plots
│   │
│   ├── analysis/              # Analysis modules
│   │   ├── __init__.py
│   │   ├── outliers.py        # Outlier detection
│   │   ├── time_analysis.py   # Time-based analysis functions
│   │   └── material_analysis.py # Material analysis functions
│   │
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── file_utils.py      # File handling utilities
│       ├── time_utils.py      # Time-related utilities
│       └── plot_utils.py      # Plotting utilities
│
├── data/                      # Data directory
│   ├── raw/                   # Raw data
│   │   ├── logs/              # Original log files
│   │   └── sap/               # SAP data
│   │
│   ├── processed/             # Processed data
│   │   └── combined_filtered_data.csv  # Combined and filtered data
│   │
│   └── external/              # External data sources
│
├── docs/                      # Documentation
│   ├── improvement.md         # Improvement documentation
│   └── manuals/               # User manuals
│       └── Manual_Panasonic.pdf  # Equipment manual
│
└── plots/                     # Generated plots and visualizations
```

## Main Components

### Data Module (`src/data/`)

This module handles all data loading, processing, and cleaning operations:

- **loader.py**: 
  - Functions for loading data from various sources (RTF files, AWS DynamoDB)
  - Includes functions like `read_all_months_since()`, `read_all_rtf_in_dir()`

- **processor.py**: 
  - Functions for processing and transforming data
  - Includes functions like `group_timestamps_to_dataframe()`, `compute_total_time()`

- **cleaner.py**: 
  - Functions for cleaning and filtering data
  - Includes functions like `filter_open_file_and_machining()`, `filter_by_message_and_extract()`

### Visualization Module (`src/visualization/`)

This module contains all visualization-related code:

- **program_plots.py**: 
  - Functions for plotting program comparisons
  - Includes functions like `plot_program_comparison()`

- **downtime_plots.py**: 
  - Functions for plotting downtime analysis
  - Includes functions like `plot_downtime_analysis()`

- **dashboard.py**: 
  - Functions for creating dashboard visualizations
  - Includes functions like `plot_tiempo_dashboard()`

- **time_range.py**: 
  - Functions for time range analysis
  - Includes functions like `plot_tiempo_range()`

- **velocity.py**: 
  - Functions for velocity analysis
  - Includes functions like `plot_velocity()`

- **material.py**: 
  - Functions for material analysis
  - Includes functions like `create_espesor_analysis_plots()`

### Analysis Module (`src/analysis/`)

This module contains analytical functions:

- **outliers.py**: 
  - Functions for outlier detection and analysis
  - Includes functions like `detect_outliers_regression()`, `plot_outliers_detailed()`

- **time_analysis.py**: 
  - Functions for time-based analysis
  - Includes functions like `summarize_by_date_and_shift()`, `time_between_placas()`

- **material_analysis.py**: 
  - Functions for material property analysis
  - Includes functions for analyzing thickness and length patterns

### Utils Module (`src/utils/`)

This module contains utility functions used across the project:

- **file_utils.py**: 
  - Utility functions for file handling
  - Includes functions like `rtf_to_dataframe()`

- **time_utils.py**: 
  - Utility functions for time calculations
  - Includes functions like `convert_to_datetime()`, `seconds_to_hms()`

- **plot_utils.py**: 
  - Utility functions for plotting
  - Includes common plotting functions used across visualization modules

## Main Analysis Notebook: FullProgramTime.ipynb

The `FullProgramTime.ipynb` notebook is the main analysis tool in this project. It provides a comprehensive analysis of machining time data, including:

1. **Setup and Data Preparation**
   - AWS and data analysis library setup
   - Loading historical machining data
   - Data preprocessing and formatting

2. **Data Integration and Processing**
   - DynamoDB integration for retrieving data from AWS
   - Data cleaning and normalization
   - Program summarization and aggregation

3. **Analysis and Insights**
   - Thickness return analysis
   - Time period analysis (by year, month, day, shift)
   - Metrics distribution analysis
   - Outlier detection and analysis

4. **Visualization**
   - Downtime analysis visualizations
   - Program performance comparison
   - Dashboard creation and export
   - Time range analysis
   - Velocity analysis
   - Material property analysis

This notebook analyzes machining time data from June 2024 to present, processing records of machine operations across different shifts and programs. It provides comprehensive analysis of machine performance, downtime, and material usage patterns to support operational decision-making.

## Usage

### Loading and Processing Data

```python
import pandas as pd
from src.data.loader import read_all_months_since

# Read all data from June 2024 to current date
combined_data = read_all_months_since(2024, 6)

# Standardize program IDs
combined_data['Programa'] = combined_data['Programa'].apply(lambda x: '0' + str(x) if not str(x).startswith('0') else str(x))

# Process and summarize program data
from src.data.processor import summarise_programs

# Summarize programs with chronological ordering
program_summary = summarise_programs(combined_data)
```

### Analyzing Downtime and Performance

```python
# Process machining times by period (year/month)
from src.analysis.time_analysis import process_machining_times_by_period

# Analyze data for May 2025
df_may_2025 = process_machining_times_by_period(program_summary, year=2025, month=5)

# Analyze thickness return patterns
from src.analysis.material_analysis import analyze_thickness_returns

# Analyze with 5-day windows
analysis_results = analyze_thickness_returns(df_may_2025, window_days=5)
```

### Visualizing Data

```python
import matplotlib.pyplot as plt
from src.visualization.program_plots import plot_program_comparison
from src.visualization.downtime_plots import plot_downtime_analysis

# Create a program comparison plot
fig_program = plot_program_comparison(program_summary, top_n=15)
plt.show()

# Create a downtime analysis plot
fig_downtime = plot_downtime_analysis(program_summary, max_downtime_hours=24)
plt.show()

# Create a dashboard visualization
from src.visualization.dashboard import plot_tiempo_dashboard

# Create dashboard for June 2025
fig = plot_tiempo_dashboard(combined_data, 2025, 6)
plt.show()
```

### Detecting Outliers

```python
from src.analysis.outliers import detect_outliers_regression, plot_outliers_detailed

# Detect outliers in the data
outliers_mask, diagnostics = detect_outliers_regression(combined_data)

# Visualize the outliers
plot_outliers_detailed(combined_data, outliers_mask, diagnostics)
```

### Working with AWS DynamoDB

```python
import boto3

# Configure AWS session
session = boto3.Session(profile_name='kubeprofile', region_name='us-east-1')

# Function to retrieve data from DynamoDB
def get_all_items_from_table(table_name):
    dynamo = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamo.Table(table_name)

    response = table.scan()
    items = response['Items']
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response['Items'])

    return items

# Get items from a table
laser_close = get_all_items_from_table('sam-stack-irlaa-LaserClosedTable-6CR5UN27N92Y')
laser_close_df = pd.DataFrame(laser_close)
```

## Data Cleaning Process

The project includes a comprehensive data cleaning process that involves:

1. **Loading Raw Data**: Reading RTF files from the logs directory
2. **Parsing and Formatting**: Converting raw logs into structured dataframes
3. **Filtering**: Removing irrelevant entries and focusing on machining operations
4. **Standardization**: Ensuring consistent formatting of program IDs, timestamps, etc.
5. **Combining**: Merging data from multiple sources (logs, SAP, DynamoDB)
6. **Outlier Detection**: Identifying and handling anomalous data points
7. **Aggregation**: Summarizing data by program, time period, material properties, etc.

The cleaned data is stored in the `data/processed/` directory, with the main combined dataset in `combined_filtered_data.csv`.

## Development

### Running Tests

```bash
pytest
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# LaserTime-Pro
