# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    automate_fixed_domain.py                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/03 14:41:39 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/03 23:18:45 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Automate run LEC with fixed domain for the selected systems.
"""

import sys
import os
import glob
import subprocess
import time
import logging
import random
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from datetime import timedelta
import cdsapi

global subprocess_counter
subprocess_counter = 0

def get_cdsapi_keys():
    """
    Lists all files in the home directory that match the pattern 'cdsapirc-*'.

    Returns:
    list: A list of filenames matching the pattern.
    """
    home_dir = os.path.expanduser('~')
    pattern = os.path.join(home_dir, '.cdsapirc-*')
    files = glob.glob(pattern)
    logging.info(f"CDSAPIRC files available at '{home_dir}': {files}")
    # Extract file suffixes from the full paths
    suffixes = [os.path.basename(file) for file in files]
    return suffixes

def copy_cdsapirc(suffix):
    """
    Copies a specific .cdsapirc file to the default .cdsapirc location.
    Args:
    suffix (str): The suffix of the .cdsapi file to be copied.
    """
    try:
        source_path = os.path.expanduser(f'~/{suffix}')
        subprocess.run(['cp', source_path, CDSAPIRC_PATH], check=True)
        logging.info(f"Copied {source_path} to {CDSAPIRC_PATH}")
    except Exception as e:
        logging.error(f"Error copying {source_path} to {CDSAPIRC_PATH}: {e}")

def prepare_box_limits_file(tracks_filtered_ids, track_id):
    """
    """
    try:
        track_data = tracks_filtered_ids[tracks_filtered_ids['track_id'] == track_id]
        # Get mininum and maximum latitude and longitude and create a 15x15 degree bounding box
        min_lat = track_data['lat vor'].min()
        max_lat = track_data['lat vor'].max()
        min_lon = track_data['lon vor'].min()
        max_lon = track_data['lon vor'].max()
        bbox_lat_min = np.floor(min_lat - 7.5)
        bbox_lat_max = np.ceil(max_lat + 7.5)
        bbox_lon_min = np.floor(min_lon - 7.5)
        bbox_lon_max = np.ceil(max_lon + 7.5)
        bbox = [bbox_lon_min, bbox_lon_max, bbox_lat_min, bbox_lat_max]
        df_bbox = pd.DataFrame([bbox], columns=['min_lon', 'max_lon', 'min_lat', 'max_lat']).T
        box_limits_file_path = f'inputs/track_{track_id}.csv'
        df_bbox.to_csv(box_limits_file_path, header=False, sep=';')
        return box_limits_file_path
    except Exception as e:
        logging.error(f"Error preparing track data for ID {track_id}: {e}")
        return None

def check_results_exist(system_id):
    """
    Check if results for the given system ID already exist.

    Args:
    system_id (int): The system ID to check.

    Returns:
    bool: True if results exist, False otherwise.
    """
    results_file_path = os.path.join(LEC_RESULTS_DIR, f"{system_id}_ERA5_track", f"{system_id}_ERA5_track_results.csv")
    return os.path.exists(results_file_path)

def get_cdsapi_data(track, box_limits_file_path):

    # Open box limits file
    box_limits = pd.read_csv(box_limits_file_path, sep=';', index_col=0, header=None)

    # Get mininum and maximum latitude and longitude 
    min_lat = box_limits.loc['min_lat'].iloc[0]
    max_lat = box_limits.loc['max_lat'].iloc[0]
    min_lon = box_limits.loc['min_lon'].iloc[0]
    max_lon = box_limits.loc['max_lon'].iloc[0]

    # Define the area for the request
    area = f"{max_lat}/{min_lon}/{min_lat}/{max_lon}" # North, West, South, East. Nort/West/Sout/East

    pressure_levels = ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70',
                       '100', '125', '150', '175', '200', '225', '250', '300', '350',
                       '400', '450', '500', '550', '600', '650', '700', '750', '775',
                       '800', '825', '850', '875', '900', '925', '950', '975', '1000']
    
    variables = ["u_component_of_wind", "v_component_of_wind", "temperature",
                 "vertical_velocity", "geopotential"]
    
    # Convert track index to DatetimeIndex
    track.index = pd.to_datetime(track['date'])

    # Convert track index to DatetimeIndex and find the last date & time
    track_datetime_index = pd.DatetimeIndex(track.index)
    last_track_timestamp = track_datetime_index.max()
    
    # Calculate if the additional day is needed by comparing last track timestamp with the last possible data timestamp for that day
    last_possible_data_timestamp_for_day = pd.Timestamp(f"{last_track_timestamp.strftime('%Y-%m-%d')} 21:00:00")
    need_additional_day = last_track_timestamp > last_possible_data_timestamp_for_day
    
    # Include additional day in dates if needed
    dates = track_datetime_index.strftime('%Y%m%d').unique()
    if need_additional_day:
        additional_day = (last_track_timestamp + timedelta(days=1)).strftime('%Y%m%d')
        dates = np.append(dates, additional_day)

    # Convert unique dates to string format for the request
    # dates = track.index.strftime('%Y%m%d').unique().tolist()
    time_range = f"{dates[0]}/{dates[-1]}"
    time_step = str(int((track.index[1] - track.index[0]).total_seconds() / 3600))
    time_step = '3' if time_step < '3' else time_step

    # File name
    infile = f"{track['track_id'].unique()[0]}_ERA5.nc"

    # Log track file bounds and requested data bounds
    logging.info(f"File Limits: max_lon (east):  min_lon (west): {min_lon}, max_lon (west): {max_lon}, min_lat (south): {min_lat}, max_lat (north): {max_lat}")
    logging.info(f"Requesting data for time range: {time_range}, and time step: {time_step}...")

    # Load ERA5 data
    logging.info("Retrieving data from CDS API...")
    c = cdsapi.Client(timeout=600)
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "pressure_level": pressure_levels,
            "date": time_range,
            "area": area,
            'time': f'00/to/23/by/{time_step}',
            "variable": variables,
        }, infile # save file as passed in arguments
    )

    if not os.path.exists(infile):
        raise FileNotFoundError("CDS API file not created.")
    return infile

def run_lorenz_cycle(track_id, tracks_filtered_ids):
    global subprocess_counter
    subprocess_counter += 1

    print(f"Process {subprocess_counter} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    if check_results_exist(track_id):
        logging.info(f"Results already exist for system ID {track_id}, skipping.")
        return track_id

    # Pick a random .cdsapirc file for each process
    if CDSAPIRC_SUFFIXES:
        chosen_suffix = random.choice(CDSAPIRC_SUFFIXES)
        copy_cdsapirc(chosen_suffix)
        logging.info(f"Switched .cdsapirc file to {chosen_suffix}")
    else:
        logging.error("No .cdsapirc files found. Please check the configuration.")

    box_limits_file_path = prepare_box_limits_file(tracks_filtered_ids, track_id)
    
    if box_limits_file_path:
        track = tracks_filtered_ids[tracks_filtered_ids['track_id'] == track_id]
        infile = get_cdsapi_data(track, box_limits_file_path)
        try:
            arguments = [infile, '-f', '-r', '-g', '-v', '-p', '--box_limits', f'{box_limits_file_path}']
            command = f"python {LEC_PATH} " + " ".join(arguments)
            subprocess.run(command, shell=True, executable='/bin/bash')
            logging.info(f"Successfully ran Lorenz Cycle script for ID {track_id}")
        except Exception as e:
            logging.error(f"Error running Lorenz Cycle script for ID {track_id}: {e}")
    else:
        logging.error(f"Error running Lorenz Cycle script for ID {track_id}: Could not prepare track data")

    return track_id

def count_evaluated_systems():
    """
    Counts the number of systems that have been evaluated based on the presence of results files.

    Returns:
    int: The number of evaluated systems.
    """
    evaluated_count = 0
    for dirname in os.listdir(LEC_RESULTS_DIR):
        if dirname.endswith('_ERA5_track') and os.path.exists(os.path.join(LEC_RESULTS_DIR, dirname, f"{dirname}_results.csv")):
            evaluated_count += 1
    return evaluated_count

# Define global variables
CDSAPIRC_SUFFIXES = get_cdsapi_keys()   
FILTERED_TRACKS = os.path.abspath('../tracks_SAt_filtered/tracks_SAt_filtered.csv') # Path to filtered tracks
LEC_RESULTS_DIR = os.path.abspath('../../LEC_Results')  # Get absolute PATH
CDSAPIRC_PATH = os.path.expanduser('~/.cdsapirc')
LEC_PATH = os.path.abspath('../../lorenz-cycle/lorenz_cycle.py')  # Get absolute path

def main():
    # Set logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("debug.log"),  # Log to a file
                            logging.StreamHandler(sys.stdout)  # Log to standard output
                        ])
    
    # Start timer
    overall_start_time = time.time()
    logging.info(f"Script start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start_time))}")
    print("Script start time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start_time)))

    # Initialize subprocess_counter with the number of already evaluated systems
    subprocess_counter = count_evaluated_systems()

    # Get all results directories
    results_directories = sorted(glob.glob('../../LEC_Results_energetic-patterns/*'))

    # Select track to process
    selected_systems = pd.read_csv('../src_barotropic_instability/systems_to_be_analysed.txt', header=None)[0].tolist()

    # Convert selected systems to string for easier comparison
    selected_systems_str = [str(system) for system in selected_systems]

    # Filter directories
    filtered_directories = [directory for directory in results_directories if any(system_id in directory for system_id in selected_systems_str)]

    system_ids = [int(os.path.basename(directory).split('_')[0]) for directory in filtered_directories]
    tracks = pd.read_csv(FILTERED_TRACKS)
    tracks_filtered_ids = tracks[tracks['track_id'].isin(system_ids)]
    print(f"Selected systems: {selected_systems}")

    # Change directory to the Lorenz Cycle program directory
    try:
        lec_dir = os.path.dirname(LEC_PATH)
        os.chdir(lec_dir)
        logging.info(f"Changed directory to {lec_dir}")
        print(f"Changed directory to {lec_dir}")
    except Exception as e:
        logging.error(f"Error changing directory: {e}")
        exit(1)

    # Pull the latest changes from Git
    try:
        subprocess.run(["git", "pull"])
        logging.info("Successfully pulled latest changes from Git")
        print("Successfully pulled latest changes from Git")
    except Exception as e:
        logging.error(f"Error pulling latest changes from Git: {e}")
        exit(1)

    # Determine the number of CPU cores to use
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    else:
        num_workers = 1

    # Log start time and total number of systems
    start_time = time.time()
    formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    total_systems_count = len(filtered_directories)
    remaining_systems_count = total_systems_count - subprocess_counter
    logging.info(f"Starting {total_systems_count} cases at {formatted_start_time}")
    logging.info(f"{subprocess_counter} cases already evaluated")
    logging.info(f"{remaining_systems_count} cases remaining to be evaluated")

    logging.info(f"Using {num_workers} CPU cores for processing")
    print(f"Using {num_workers} CPU cores for processing")

    # Adjust the function to use partial
    run_lorenz_cycle_partial = partial(run_lorenz_cycle, tracks_filtered_ids=tracks_filtered_ids)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, completed_id in enumerate(executor.map(run_lorenz_cycle_partial, system_ids), 1):
            current_time = time.time()
            elapsed_time = current_time - overall_start_time
            average_time_per_system = elapsed_time / idx
            estimated_total_time = average_time_per_system * (remaining_systems_count - idx)
            estimated_completion_time = overall_start_time + estimated_total_time
            formatted_estimated_completion_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(estimated_completion_time))

            logging.info(f"Completed {idx}/{total_systems_count} cases (ID {completed_id}). Estimated completion time: {formatted_estimated_completion_time}")
    
    # Log end time
    end_time = time.time()
    formatted_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    logging.info(f"Finished {len(system_ids)} cases at {formatted_end_time}")

    # Calculate and log execution times
    total_time_seconds = end_time - start_time
    total_time_minutes = total_time_seconds / 60
    total_time_hours = total_time_seconds / 3600
    mean_time_minutes = total_time_minutes / len(system_ids)
    mean_time_hours = total_time_hours / len(system_ids)

    logging.info(f'Total time for {len(system_ids)} cases: {total_time_hours:.2f} hours ({total_time_minutes:.2f} minutes)')
    logging.info(f'Mean time per case: {mean_time_hours:.2f} hours ({mean_time_minutes:.2f} minutes)')


if __name__ == '__main__':
    main()