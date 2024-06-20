# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    automate_Lorenz.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/20 11:33:00 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/21 16:29:03 by daniloceano      ###   ########.fr        #
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
import pandas as pd
import numpy as np
from datetime import timedelta
import cdsapi
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('log.automate_run_LEC.txt', mode='w')])

def get_cdsapi_keys():
    home_dir = os.path.expanduser('~')
    pattern = os.path.join(home_dir, '.cdsapirc-*')
    files = glob.glob(pattern)
    logging.info(f"CDSAPIRC files available at '{home_dir}': {files}")
    suffixes = [os.path.basename(file) for file in files]
    return suffixes

def check_results_exist(system_id):
    results_file_path = os.path.join(LEC_RESULTS_DIR, f"{system_id}_ERA5_fixed", f"{system_id}_ERA5_fixed_results.csv")
    return os.path.exists(results_file_path)    

def get_region_key(track_id):
    if '2005' in str(track_id):
        return 'Reg1'
    elif '2007' in str(track_id):
        return 'Reg2'
    elif '2008' in str(track_id):
        return 'Reg3'

def prepare_box_limits_file(region_key):
    try:
        region_limits = DOMAINS[region_key]
        bbox_lat_min = region_limits["lat_min"]
        bbox_lat_max = region_limits["lat_max"]
        bbox_lon_min = region_limits["lon_min"]
        bbox_lon_max = region_limits["lon_max"]
        
        bbox = [bbox_lon_min, bbox_lon_max, bbox_lat_min, bbox_lat_max]
        df_bbox = pd.DataFrame([bbox], columns=['min_lon', 'max_lon', 'min_lat', 'max_lat']).T
        
        box_limits_file_path = f'inputs/box_limits_{region_key}.csv'
        df_bbox.to_csv(box_limits_file_path, header=False, sep=';')
        
        return box_limits_file_path
    except Exception as e:
        logging.error(f"Error preparing box limits file for region {region_key}: {e}")
        return None
    
def prepare_track_data(track):
    """
    Prepare and save track data for a given system ID in the required format.
    Each system ID will have its own input file.

    Args:
    system_id (int): The ID of the system for which to prepare the track data.
    """
    try:
        track_id = int(*track['track_id'].unique())
        # Explicitly create a copy of the DataFrame to avoid SettingWithCopyWarning
        formatted_data = track[['date', 'lat vor', 'lon vor', 'vor42']].copy()
        formatted_data.columns = ['time', 'Lat', 'Lon', 'min_max_zeta_850']
        formatted_data['min_max_zeta_850'] = - np.abs(formatted_data['min_max_zeta_850'])
        # Create a unique input file for each system ID
        input_file_path = f'inputs/track_{track_id}.csv'
        formatted_data.to_csv(input_file_path, index=False, sep=';')
        return input_file_path
    except Exception as e:
        logging.error(f"Error preparing track data for ID {track_id}: {e}")
        return None

def get_cdsapi_data(track, box_limits_file_path):
    # Open box limits file
    box_limits = pd.read_csv(box_limits_file_path, sep=';', index_col=0, header=None)

    # Get minimum and maximum latitude and longitude 
    min_lat = np.floor(float(track['lat vor'].min()) - 15)
    max_lat = np.ceil(float(track['lat vor'].max()) + 15)
    min_lon = np.floor(float(track['lon vor'].min()) - 15)
    max_lon = np.ceil(float(track['lon vor'].max()) + 15)
    
    # Define the area for the request
    area = [max_lat, min_lon, min_lat, max_lon]  # North, West, South, East

    pressure_levels = ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70',
                       '100', '125', '150', '175', '200', '225', '250', '300', '350',
                       '400', '450', '500', '550', '600', '650', '700', '750', '775',
                       '800', '825', '850', '875', '900', '925', '950', '975', '1000']
    
    variables = ["u_component_of_wind", "v_component_of_wind", "temperature",
                 "vertical_velocity", "geopotential"]

    # Convert track index to DatetimeIndex
    track.index = pd.to_datetime(track['date'])

    # Calculate if the additional day is needed
    last_track_timestamp = track.index.max()
    last_possible_data_timestamp_for_day = pd.Timestamp(f"{last_track_timestamp.strftime('%Y-%m-%d')} 21:00:00")
    need_additional_day = last_track_timestamp > last_possible_data_timestamp_for_day
    
    dates = track.index.strftime('%Y%m%d').unique().tolist()
    if need_additional_day:
        additional_day = (last_track_timestamp + timedelta(days=1)).strftime('%Y%m%d')
        dates.append(additional_day)

    time_range = f"{dates[0]}/{dates[-1]}"
    time_step = int((track.index[1] - track.index[0]).total_seconds() / 3600)
    time_step = '3' if time_step < 3 else str(time_step)

    infile = f"{track['track_id'].unique()[0]}_ERA5.nc"

    logging.info(f"File Limits: min_lon (west): {min_lon}, max_lon (east): {max_lon}, min_lat (south): {min_lat}, max_lat (north): {max_lat}")
    logging.info(f"Requesting data for time range: {time_range}, and time step: {time_step}...")

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
            "time": [f"{hour:02d}:00" for hour in range(0, 24, int(time_step))],
            "variable": variables,
        },
        infile
    )

    if not os.path.exists(infile):
        raise FileNotFoundError("CDS API file not created.")
    return infile

def slice_netcdf_to_match_track(infile, track):
    # Load the NetCDF file
    ds = xr.open_dataset(infile)

    # Convert track dates to datetime objects
    track_dates = pd.to_datetime(track['date'])

    # Get the initial and final dates from the track
    initial_date = track_dates.min()
    final_date = track_dates.max()

    # Slice the dataset to match the initial and final dates
    ds_sliced = ds.sel(time=slice(initial_date, final_date))

    # Define the output file name
    outfile = infile.replace('.nc', '_sliced.nc')

    # Save the sliced dataset to a new NetCDF file
    ds_sliced.to_netcdf(outfile)

    return outfile

def run_lorenz_cycle(track_file):
    subprocess_counter = count_evaluated_systems()
    logging.info(f"Process {subprocess_counter} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"Process {subprocess_counter} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    track = pd.read_csv(track_file)

    try:
        track_id = int(*track['track_id'].unique())
    except Exception as e:
        logging.error(f"Multiple track IDs found for {track_file}: {e}")
        return None

    if check_results_exist(track_id):
        logging.info(f"Results already exist for system ID {track_id}, skipping.")
        print(f"Results already exist for system ID {track_id}, skipping.")
        return track_id

    region_key = get_region_key(track_id)
    box_limits_file_path = prepare_box_limits_file(region_key)
    input_track_path = prepare_track_data(track)
    
    if box_limits_file_path:
        logging.info(f"Preparing data for ID {track_id}...")
        infile = get_cdsapi_data(track, box_limits_file_path)
        infile = slice_netcdf_to_match_track(infile, track)
        logging.info(f"Successfully prepared data for ID {track_id}")

        try:
            logging.info(f"Running Lorenz Cycle script for ID {track_id} with fixed framework...")

            # 1) Copy fvars_ERA5-copernicus to fvars
            command = f"cp inputs/fvars_ERA5-cdsapi inputs/fvars"
            subprocess.run(command, shell=True, executable='/bin/bash')

            # 2) Run the Lorenz Cycle script with fixed domain
            arguments = [infile, '-f', '-r', '-g', '-v', '-p', '--box_limits', f'{box_limits_file_path}']
            command = f"python {LEC_PATH} " + " ".join(arguments)
            subprocess.run(command, shell=True, executable='/bin/bash')
            logging.info(f"Successfully ran Lorenz Cycle script for ID {track_id} with fixed domain")

            # 3) Run the Lorenz Cycle script with track
            arguments = [f'{infile}', '-t', '-r', '-g', '-v', '-p', '-z', '--trackfile', input_track_path]
            command = f"python {LEC_PATH} " + " ".join(arguments)
            subprocess.run(command, shell=True, executable='/bin/bash')
            logging.info(f"Successfully ran Lorenz Cycle script for ID {track_id} with track")

            logging.info(f"Successfully processed {track_id}, process {subprocess_counter}")
            print(f"Successfully processed ID {track_id}, process {subprocess_counter}")

        except Exception as e:
            logging.error(f"Error running Lorenz Cycle script for ID {track_id}: {e}")
    else:
        logging.error(f"Error running Lorenz Cycle script for ID {track_id}: Could not prepare track data")

    return track_id

def count_evaluated_systems():
    evaluated_count = 0
    for dirname in os.listdir(LEC_RESULTS_DIR):
        if dirname.endswith('_ERA5_track') and os.path.exists(os.path.join(LEC_RESULTS_DIR, dirname, f"{dirname}_results.csv")):
            evaluated_count += 1
    return evaluated_count

# Define global variables
TRACKS_DIR = os.path.abspath('../results_test_Dias_Pinto') 
LEC_RESULTS_DIR = os.path.abspath('../../LEC_Results')  
CDSAPIRC_PATH = os.path.expanduser('~/.cdsapirc')
LEC_PATH = os.path.abspath('../../lorenz-cycle/lorenz_cycle.py') 

DOMAINS = {
    "Reg1": {
        "lat_min": -42.5,
        "lat_max": -17.5,
        "lon_min": -60,
        "lon_max": -30
    },
    "Reg2": {
        "lat_min": -65,
        "lat_max": -20,
        "lon_min": -65,
        "lon_max": -20
    },
    "Reg3": {
        "lat_min": -60,
        "lat_max": -30,
        "lon_min": -82.5,
        "lon_max": -15
    }
}

def main():
    overall_start_time = time.time()
    logging.info(f"Script start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start_time))}")
    print("Script start time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start_time)))

    subprocess_counter = count_evaluated_systems()

    track_files = glob.glob(os.path.join(TRACKS_DIR, '*.csv'))

    try:
        lec_dir = os.path.dirname(LEC_PATH)
        os.chdir(lec_dir)
        logging.info(f"Changed directory to {lec_dir}")
        print(f"Changed directory to {lec_dir}")
    except Exception as e:
        logging.error(f"Error changing directory: {e}")
        exit(1)

    try:
        subprocess.run(["git", "pull"])
        logging.info("Successfully pulled latest changes from Git")
        print("Successfully pulled latest changes from Git")
    except Exception as e:
        logging.error(f"Error pulling latest changes from Git: {e}")
        exit(1)

    # Parallel execution with ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_lorenz_cycle, track_file): track_file for track_file in track_files}
        for future in as_completed(futures):
            track_file = futures[future]
            try:
                track_id = future.result()
                logging.info(f"Successfully processed {track_id}")
            except Exception as e:
                logging.error(f"Error processing {track_file}: {e}")

    overall_end_time = time.time()
    logging.info(f"Script end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_end_time))}")
    print("Script end time: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_end_time)))


if __name__ == '__main__':
    main()
