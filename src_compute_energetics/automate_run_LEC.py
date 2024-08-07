# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    automate_run_LEC.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/01/22 13:52:26 by daniloceano       #+#    #+#              #
#    Updated: 2024/02/02 16:48:32 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

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

# Update logging configuration to use the custom handler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('log.automate_run_LEC.txt', mode='w')])

overall_start_time = time.time()

REGION = sys.argv[1] # Region to process
FILTERED_TRACKS = '../tracks_SAt_filtered/tracks_SAt_filtered.csv' # Path to filtered tracks
LEC_PATH = os.path.abspath('../../lorenz-cycle/lorenz_cycle.py')  # Get absolute path
LEC_RESULTS_DIR = os.path.abspath('../../LEC_Results')  # Get absolute PATH
CDSAPIRC_PATH = os.path.expanduser('~/.cdsapirc')

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


def prepare_track_data(system_id):
    """
    Prepare and save track data for a given system ID in the required format.
    Each system ID will have its own input file.

    Args:
    system_id (int): The ID of the system for which to prepare the track data.
    """
    try:
        track_data = tracks_region[tracks_region['track_id'] == system_id]
        # Explicitly create a copy of the DataFrame to avoid SettingWithCopyWarning
        formatted_data = track_data[['date', 'lat vor', 'lon vor', 'vor42']].copy()
        formatted_data.columns = ['time', 'Lat', 'Lon', 'min_max_zeta_850']
        formatted_data['min_max_zeta_850'] = - np.abs(formatted_data['min_max_zeta_850'])
        # Create a unique input file for each system ID
        input_file_path = f'inputs/track_{system_id}.csv'
        formatted_data.to_csv(input_file_path, index=False, sep=';')
        return input_file_path
    except Exception as e:
        logging.error(f"Error preparing track data for ID {system_id}: {e}")
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

def run_lorenz_cycle(id):
    global subprocess_counter
    subprocess_counter += 1

    if check_results_exist(id):
        logging.info(f"Results already exist for system ID {id}, skipping.")
        return id

    # Pick a random .cdsapirc file for each process
    if CDSAPIRC_SUFFIXES:
        chosen_suffix = random.choice(CDSAPIRC_SUFFIXES)
        copy_cdsapirc(chosen_suffix)
        logging.info(f"Switched .cdsapirc file to {chosen_suffix}")
    else:
        logging.error("No .cdsapirc files found. Please check the configuration.")

    input_track_path = prepare_track_data(id)
    if input_track_path:
        try:
            arguments = [f'{id}_ERA5.nc', '-t', '-r', '-g', '-v', '-p', '-z', '--cdsapi', '--trackfile', input_track_path]
            command = f"python {LEC_PATH} " + " ".join(arguments)
            subprocess.run(command, shell=True, executable='/bin/bash')
            logging.info(f"Successfully ran Lorenz Cycle script for ID {id}")
        except Exception as e:
            logging.error(f"Error running Lorenz Cycle script for ID {id}: {e}")
    else:
        logging.error(f"Error running Lorenz Cycle script for ID {id}: Could not prepare track data")

    return id

CDSAPIRC_SUFFIXES = get_cdsapi_keys()

# Initialize subprocess_counter with the number of already evaluated systems
subprocess_counter = count_evaluated_systems()

logging.info(f"Starting automate_run_LEC.py for region: {REGION}")

tracks = pd.read_csv(FILTERED_TRACKS)
tracks_region = tracks[tracks['region'] == REGION]
system_ids = tracks_region['track_id'].unique()

# Change directory to the Lorenz Cycle program directory
try:
    lec_dir = os.path.dirname(LEC_PATH)
    os.chdir(lec_dir)
    logging.info(f"Changed directory to {lec_dir}")
except Exception as e:
    logging.error(f"Error changing directory: {e}")
    exit(1)

# Pull the latest changes from Git
try:
    subprocess.run(["git", "pull"])
    logging.info("Successfully pulled latest changes from Git")
except Exception as e:
    logging.error(f"Error pulling latest changes from Git: {e}")
    exit(1)

# Determine the number of CPU cores to use
max_cores = os.cpu_count()
num_workers = max(1, max_cores - 4) if max_cores else 1
logging.info(f"Using {num_workers} CPU cores")

# Process each system ID in parallel and log progress
start_time = time.time()
formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
total_systems_count = len(system_ids)
remaining_systems_count = total_systems_count - subprocess_counter
logging.info(f"Starting {total_systems_count} cases at {formatted_start_time}")
logging.info(f"{subprocess_counter} cases already evaluated")
logging.info(f"{remaining_systems_count} cases remaining to be evaluated")

# Inside the loop, after processing each system, calculate and log the estimated completion time
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    for idx, completed_id in enumerate(executor.map(run_lorenz_cycle, system_ids), 1):
        current_time = time.time()
        elapsed_time = current_time - overall_start_time
        average_time_per_system = elapsed_time / idx
        estimated_total_time = average_time_per_system * (remaining_systems_count - idx)
        estimated_completion_time = overall_start_time + estimated_total_time
        formatted_estimated_completion_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(estimated_completion_time))
        
        logging.info(f"Completed {idx}/{total_systems_count} cases (ID {completed_id}). Estimated completion time: {formatted_estimated_completion_time}")
        
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