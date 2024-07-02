# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    automate_composites.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/24 14:42:50 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/01 19:31:38 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


"""
This script will automate the process of creating composites for each system selected.
"""

import sys
import os
import time
import math
import random
import cdsapi
import logging
import subprocess
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from metpy.units import units
from metpy.interpolate import interpolate_1d, log_interpolate_1d


# Update logging configuration to use the custom handler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('log.automate_run_LEC.txt', mode='w')])

# REGION = sys.argv[1] # Region to process
LEC_RESULTS_DIR = os.path.abspath('../../LEC_Results_energetic-patterns')  # Get absolute PATH
CDSAPIRC_PATH = os.path.expanduser('~/.cdsapirc')
OUTPUT_DIR = '../results_nc_files/composites_fluxes_downstream/'

DEBUG_CODE = False
DEBUG_CDSAPI = False

def get_cdsapi_keys():
    """
    Lists all files in the home directory that match the pattern 'cdsapirc-*'.

    Returns:
    list: A list of filenames matching the pattern.
    """
    home_dir = os.path.expanduser('~')
    pattern = os.path.join(home_dir, '.cdsapirc-*')
    files = glob(pattern)
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

def get_cdsapi_era5_data(filename: str, track: pd.DataFrame, pressure_levels: list, variables: list) -> xr.Dataset:

    # Pick a random .cdsapirc file for each process
    # if DEBUG_CDSAPI == False:
    #     if CDSAPIRC_SUFFIXES:
    #         chosen_suffix = random.choice(CDSAPIRC_SUFFIXES)
    #         copy_cdsapirc(chosen_suffix)
    #         logging.info(f"Switched .cdsapirc file to {chosen_suffix}")
    #     else:
    #         logging.error("No .cdsapirc files found. Please check the configuration.")

    # Extract bounding box (lat/lon limits) from track
    min_lat, max_lat = track['Lat'].min(), track['Lat'].max()
    min_lon, max_lon = track['Lon'].min(), track['Lon'].max()

    # Apply a 20-degree buffer and round to nearest integer
    buffered_max_lat = math.ceil(max_lat + 20)
    buffered_min_lon = math.floor(min_lon - 20)
    buffered_min_lat = math.floor(min_lat - 20)
    buffered_max_lon = math.ceil(max_lon + 20)

    # Define the area for the request
    area = f"{buffered_max_lat}/{buffered_min_lon}/{buffered_min_lat}/{buffered_max_lon}" # North, West, South, East. Nort/West/Sout/East
    
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
    time_range = f"{dates[0]}/{dates[-1]}"
     
    if len(track.index) > 1:
        time_step = str(int((track.index[1] - track.index[0]).total_seconds() / 3600))
        time_step = '3' if int(time_step) < 3 else time_step
    else:
        time_step = '3'  # Default value when only one time step

    # Load ERA5 data
    infile = f"{filename}.nc"

    if not os.path.exists(infile):
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
    
    else:
        logging.info("CDS API file already exists.")
        return infile
    
def create_pressure_array(pressure_levels, time, latitude, longitude, level):
    # Create a 4D DataArray for pressure levels
    pressure_values = np.repeat(pressure_levels.magnitude[:, np.newaxis, np.newaxis, np.newaxis],
                                len(time), axis=1)
    pressure_values = np.repeat(pressure_values, len(latitude), axis=2)
    pressure_values = np.repeat(pressure_values, len(longitude), axis=3)

    pressure_da = xr.DataArray(
        pressure_values,
        dims=['level', 'time', 'latitude', 'longitude'],
        coords={
            'level': level,
            'time': time,
            'latitude': latitude,
            'longitude': longitude
        },
        attrs={'units': pressure_levels.units}
    )

    return pressure_da

def create_bae_composite(infile, track):
    # Load the dataset
    ds = xr.open_dataset(infile)

    # Open variables for calculations and assign units
    temperature = ds['t'] * units.kelvin
    pressure = ds.level * units.hPa
    u = ds['u'] * units('m/s')
    v = ds['v'] * units('m/s')
    hgt = (ds['z'] / 9.8) * units('gpm')
    latitude = ds.latitude

    # Empty lists to store the slices
    u_slices_system = []
    v_slices_system = []
    hgt_slices_system = []
    temperature_slices_system = []

    for time_step in track.index:
        # Select the time step
        u_time = u.sel(time=time_step)
        v_time = v.sel(time=time_step)
        hgt_time = hgt.sel(time=time_step)
        temperature_time = temperature.sel(time=time_step)

        # Select the track limits
        min_lon, max_lon = track.loc[time_step, 'min_lon'], track.loc[time_step, 'max_lon']
        min_lat, max_lat = track.loc[time_step, 'min_lat'], track.loc[time_step, 'max_lat']

        # Slice for track limits
        u_time_slice = u_time.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))
        v_time_slice = v_time.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))
        hgt_time_slice = hgt_time.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))
        temperature_time_slice = temperature_time.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))

        # Append to the lists
        u_slices_system.append(u_time_slice)
        v_slices_system.append(v_time_slice)
        hgt_slices_system.append(hgt_time_slice)
        temperature_slices_system.append(temperature_time_slice)
    
    # Calculate the composites for this system
    u_mean = np.mean(u_slices_system, axis=0)
    v_mean = np.mean(v_slices_system, axis=0)
    hgt_mean = np.mean(hgt_slices_system, axis=0)
    temperature_mean = np.mean(temperature_slices_system, axis=0)

    # Create a DataArray using an extra dimension
    x_size, y_size = u_mean.shape[2], u_mean.shape[1]
    x = np.linspace(- x_size / 2, (x_size / 2) - 1 , x_size)
    y = np.linspace(- y_size / 2, (y_size / 2) - 1, y_size)
    level = u.level
    track_id = int(infile.split('.')[0].split('-')[0])

    # Create DataArrays
    da_u = xr.DataArray(
        u_mean,
        dims=['level', 'y', 'x'],
        coords={'level': level, 'y': y, 'x': x},
        name='u',
        attrs={'units': str(u_time_slice.metpy.units)}
    )

    da_v = xr.DataArray(
        v_mean,
        dims=['level', 'y', 'x'],
        coords={'level': level, 'y': y, 'x': x},
        name='v',
        attrs={'units': str(v_time_slice.metpy.units)}
    )

    da_hgt = xr.DataArray(
        hgt_mean,
        dims=['level', 'y', 'x'],
        coords={'level': level, 'y': y, 'x': x},
        name='hgt',
        attrs={'units': str(hgt_time_slice.metpy.units)}
    )

    da_temperature = xr.DataArray(
        temperature_mean,
        dims=['level', 'y', 'x'],
        coords={'level': level, 'y': y, 'x': x},
        name='temperature',
        attrs={'units': str(temperature_time_slice.metpy.units)}
    )

    # Combine into a Dataset and add track_id as a coordinate
    ds_composite = xr.Dataset({
        'u': da_u,
        'v': da_v,
        'hgt': da_hgt,
        't': da_temperature
    })

    # Assigning track_id as a coordinate
    ds_composite = ds_composite.assign_coords(track_id=track_id)  # Assigning track_id as a coordinate

    return ds_composite

def save_composite(composites, output_dir, total_systems_count):
    os.makedirs(output_dir, exist_ok=True)

    # Create a composite across all systems
    logging.info("Finished processing all systems. Creating composite...")
    composites = [composite for composite in composites if composite is not None]

    # Concatenate the composites and calculate the mean
    da_composite = xr.concat(composites, dim='track_id')
    ds_composite_mean = da_composite.mean(dim='track_id')

    # Save the composites
    da_composite.to_netcdf(f'{output_dir}/bae_composite_track_ids.nc')
    ds_composite_mean.to_netcdf(f'{output_dir}/bae_composite_mean.nc')
    logging.info("Saved bae_composite_track_ids.nc and bae_composite_mean.nc")

    # Log end time
    end_time = time.time()
    formatted_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    logging.info(f"Finished {total_systems_count} cases at {formatted_end_time}")


def process_system(system_dir):
    """
    Process the selected system.
    """
    # Get track and periods data
    try:
        logging.info(f"Processing started for {system_dir}")
        track_file = glob(f'{system_dir}/*trackfile')[0]  # Assuming filename pattern
        periods_file = glob(f'{system_dir}/*periods.csv')[0]
    except Exception as e:
        logging.error(f"Did not find files for {system_dir}: {e}")
        return None     

    # Load track and periods data
    try:
        track = pd.read_csv(track_file, index_col=0, sep=';')
        track.index = pd.to_datetime(track.index)
        periods = pd.read_csv(periods_file, index_col=0, parse_dates=['start', 'end'])
    except Exception as e:
        logging.error(f"Error reading files for {system_dir}: {e}")
        return None
    
    # Check if both track and periods data are available
    if track.empty or periods.empty:
        logging.info(f"No track or periods data for {system_dir}")
        return None
    if 'incipient' not in periods.index:
        logging.info(f"No incipient phase data for {system_dir}")
        return None

    # Filter for incipient phase only
    incipient_start = periods.loc['incipient', 'start']
    incipient_end = periods.loc['incipient', 'end']
    track = track[(track.index >= incipient_start) & (track.index <= incipient_end)]
    if track.empty:
        logging.info(f"No incipient phase data for {system_dir}")
        return None

    system_id = os.path.basename(system_dir).split('_')[0] # Get system ID

    # Get ERA5 data for interest variables
    pressure_levels = ['250', '300', '350', '550', '500', '450', '700', '750', '800', '950', '975', '1000']
    variables = ["u_component_of_wind", "v_component_of_wind", "temperature", "geopotential"]
    infile_bae = get_cdsapi_era5_data(f'{system_id}-bae', track, pressure_levels, variables) 

    # Make composite
    ds_composite = create_bae_composite(infile_bae, track) 
    logging.info(f"Processing completed for {system_dir}")

    # # Delete infile
    # if os.path.exists(infile_bae):
    #     os.remove(infile_bae)

    return ds_composite 

CDSAPIRC_SUFFIXES = get_cdsapi_keys()

def main():
    # Get all directories in the LEC_RESULTS_DIR
    results_directories = sorted(glob(f'{LEC_RESULTS_DIR}/*'))

    # Select track to process
    selected_systems = pd.read_csv('systems_to_be_analysed.txt', header=None)[0].tolist()

    # Convert selected systems to string for easier comparison
    selected_systems_str = [str(system) for system in selected_systems]

    # Filter directories
    filtered_directories = [directory for directory in results_directories if any(system_id in directory for system_id in selected_systems_str)]

    # # Determine the number of CPU cores to use
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    else:
        max_cores = os.cpu_count()
        num_workers = max(1, max_cores - 4) if max_cores else 1
        logging.info(f"Using {num_workers} CPU cores")
    
    if DEBUG_CDSAPI == True:
        num_workers = 1

    if DEBUG_CODE == True:
        logging.info(f"Debug mode!")
        composites = [process_system(results_directories[0])]
        print(composites)
        sys.exit(0)

    # Log start time and total number of systems
    start_time = time.time()
    formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    total_systems_count = len(filtered_directories)
    logging.info(f"Starting {total_systems_count} cases at {formatted_start_time}")

    logging.info(f"Using {num_workers} CPU cores for processing")

    composites = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_system, dir_path) for dir_path in filtered_directories]
        for future in as_completed(futures):
            result = future.result()
            if result is not None and np.any(result):
                composites.append(result)

    # Assuming a function to aggregate and save the results
    if composites:
        logging.info("Aggregating and saving BAe composites...")
        save_composite(composites, OUTPUT_DIR, total_systems_count)

if __name__ == "__main__":
    main()
