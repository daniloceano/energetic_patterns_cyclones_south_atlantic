# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_results_periods_energy.py                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/06 10:45:29 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/06 11:17:08 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def process_system_dir(system_dir, fixed_results_path, track_with_periods):
    """
    Process a single system directory to calculate average values for specified periods.

    Parameters:
    - system_dir: The directory name of the current system being processed.
    - fixed_results_path: The base directory path containing all system directories.

    Returns:
    - A tuple containing the system directory name and a DataFrame with average values for each period,
      or None if the required files are not found or an error occurs.
    """
    # Construct the full path to the system directory
    system_path_fixed = os.path.join(fixed_results_path, system_dir)

    # Get track with periods for the current system
    system_id = int(system_dir.split('_')[0])
    track_with_periods_system = track_with_periods[track_with_periods['track_id'] == system_id]

    # Try to find the results CSV file by pattern
    try:
        results_file = next((f for f in os.listdir(system_path_fixed) if f.endswith('fixed_results.csv')), None)
    except FileNotFoundError:
        # Return None if the file is not found
        return system_dir, None

    # Construct the full paths to the results and periods CSV files
    results_path = os.path.join(system_path_fixed, results_file)

    try:
        # Attempt to read the CSV files into DataFrames
        results_df = pd.read_csv(results_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        # Return None if any of the files are not found
        return system_dir, None
    
    # Get valid periods for the current system
    periods_system = track_with_periods_system[['period', 'date']]
    valid_periods = periods_system.loc[pd.notna(periods_system['period']), 'period'].unique()
    
    # Initialize a DataFrame to hold the average values for each period
    averages_df = pd.DataFrame(columns=results_df.columns)

    # Calculate the average values for each period
    for period in valid_periods:
        start_time, end_time = periods_system[periods_system['period'] == period]['date'].min(), periods_system[periods_system['period'] == period]['date'].max()
        period_data = results_df.loc[start_time:end_time].mean()
        averages_df.loc[period] = period_data

    return system_dir, averages_df

def main(fixed_results_path, track_with_periods_path, output_path):
    """
    Main function to process all system directories in parallel and save the average values
    for specified periods to CSV files.

    Parameters:
    - fixed_results_path: The base directory path containing all system directories.
    """
    # List all directories that match the expected pattern
    system_dirs = [d for d in os.listdir(fixed_results_path) if d.endswith('_ERA5_fixed')]
    system_averages = {}

    # Open tracks with periods
    track_with_periods = pd.read_csv(track_with_periods_path)

    # Use a ProcessPoolExecutor to process directories in parallel
    with ProcessPoolExecutor() as executor:
        # Map each system directory to a future task
        future_to_system_dir = {executor.submit(process_system_dir, system_dir, fixed_results_path, track_with_periods): system_dir for system_dir in system_dirs}
        # Monitor the progress of tasks with a progress bar
        for future in tqdm(as_completed(future_to_system_dir), total=len(system_dirs)):
            system_dir = future_to_system_dir[future]
            try:
                # Attempt to retrieve the result of each task
                _, averages_df = future.result()
                if averages_df is not None:
                    system_averages[system_dir] = averages_df
            except Exception as e:
                print(f"Error processing {system_dir}: {e}")

    # Save the computed averages to CSV files
    os.makedirs(output_path, exist_ok=True)
    for system_dir, averages_df in system_averages.items():
        system_id = system_dir.split('_')[0]
        output_file_path = os.path.join(output_path, f"{system_id}_averages.csv")
        averages_df.to_csv(output_file_path)

if __name__ == "__main__":
    fixed_results_path = '../../LEC_Results'
    track_with_periods_path = '..//tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv'
    output_path = '../csv_fixed_framework_database_energy_by_periods'
    main(fixed_results_path, track_with_periods_path, output_path)
