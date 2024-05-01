# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    automate_composites.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/24 14:42:50 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/01 01:35:02 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


"""
This script will automate the process of creating composites for PV for each system selected.
"""

import sys
import os
import time
import logging
import random
import pandas as pd
import numpy as np
import subprocess
import xarray as xr
import cdsapi
import math
import metpy.calc
from datetime import timedelta
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from metpy.units import units

# Update logging configuration to use the custom handler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('log.automate_run_LEC.txt', mode='w')])

# REGION = sys.argv[1] # Region to process
LEC_RESULTS_DIR = os.path.abspath('../../LEC_Results_energetic-patterns')  # Get absolute PATH
CDSAPIRC_PATH = os.path.expanduser('~/.cdsapirc')

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

def select_systems(results_directory):
    """
    """
    clusters_to_use = ["ARG_DJF_cl_2", "ARG_JJA_cl_1",
                       "LA-PLATA_DJF_cl_2", "LA-PLATA_JJA_cl_2",
                       "SE-BR_DJF_cl_2", "SE-BR_JJA_cl_3"]
    
    selected_systems = []

    for cluster in clusters_to_use:
        # Get information about the cluster
        region, season = cluster.split('_')[:2]
        cluster_number = cluster.split('_')[3]

        # Open cluster json file
        system_dir = os.path.join(results_directory, f"{region}_{season}", "IcItMD")
        json_file = glob(f"{system_dir}/*.json")[0]
        df_system = pd.read_json(json_file)

        # Get system IDs
        cluster_ids = df_system[f'Cluster {cluster_number}']['Cyclone IDs']

        for system_id in cluster_ids:
            selected_systems.append(int(system_id))

    return selected_systems

def process_system(system_dir):

    # # Pick a random .cdsapirc file for each process
    # if CDSAPIRC_SUFFIXES:
    #     chosen_suffix = random.choice(CDSAPIRC_SUFFIXES)
    #     copy_cdsapirc(chosen_suffix)
    #     logging.info(f"Switched .cdsapirc file to {chosen_suffix}")
    # else:
    #     logging.error("No .cdsapirc files found. Please check the configuration.")


    try:
        logging.info(f"Processing started for {system_dir}")
        track_file = glob(f'{system_dir}/*trackfile')[0]  # Assuming filename pattern
        periods_file = glob(f'{system_dir}/*periods.csv')[0]
    except Exception as e:
        logging.error(f"Did not find files for {system_dir}: {e}")
        return None     

    try:
        track = pd.read_csv(track_file, index_col=0, sep=';')
        track.index = pd.to_datetime(track.index)
        periods = pd.read_csv(periods_file, index_col=0, parse_dates=['start', 'end'])
    except Exception as e:
        logging.error(f"Error reading files for {system_dir}: {e}")
        return None
    
    if track.empty or periods.empty:
        logging.info(f"No track or periods data for {system_dir}")
        return None

    if 'intensification' not in periods.index:
        logging.info(f"No intensification phase data for {system_dir}")
        return None

    # Filter for intensification phase only
    intensification_start = periods.loc['intensification', 'start']
    intensification_end = periods.loc['intensification', 'end']
    track = track[(track.index >= intensification_start) & (track.index <= intensification_end)]
    if track.empty:
        logging.info(f"No intensification phase data for {system_dir}")
        return None

    # Load ERA5 data
    system_id = os.path.basename(system_dir).split('_')[0]
    
    infile = get_cdsapi_era5_data(system_id, track)
    
    pv_mean = create_pv_composite(infile, track)

    logging.info(f"Processing completed for {system_dir}")

    # Delete infile
    if os.path.exists(infile):
        os.remove(infile)

    return pv_mean  # or any data structure you are processing

def get_cdsapi_era5_data(track_id: str, track: pd.DataFrame) -> xr.Dataset:

    # Extract bounding box (lat/lon limits) from track
    min_lat, max_lat = track['Lat'].min(), track['Lat'].max()
    min_lon, max_lon = track['Lon'].min(), track['Lon'].max()

    # Apply a 15-degree buffer and round to nearest integer
    buffered_max_lat = math.ceil(max_lat + 15)
    buffered_min_lon = math.floor(min_lon - 15)
    buffered_min_lat = math.floor(min_lat - 15)
    buffered_max_lon = math.ceil(max_lon + 15)

    # Define the area for the request
    area = f"{buffered_max_lat}/{buffered_min_lon}/{buffered_min_lat}/{buffered_max_lon}" # North, West, South, East. Nort/West/Sout/East

    pressure_levels = ['200', '250', '300']
    
    variables = ["u_component_of_wind", "v_component_of_wind", "temperature"]
    
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

    # Load ERA5 data
    infile = f"{track_id}.nc"

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

def create_pv_composite(infile, track):
    # Load the dataset
    ds = xr.open_dataset(infile)

    # Open variables for calculations and assign units
    temperature = ds['t'] * units.kelvin
    pressure = ds.level * units.hPa
    u = ds['u'] * units('m/s')
    v = ds['v'] * units('m/s')
    latitude = ds.latitude

    # Calculate potential temperature, vorticity and Coriolis parameter
    potential_temperature = metpy.calc.potential_temperature(pressure, temperature)
    zeta = metpy.calc.vorticity(u, v)
    f = metpy.calc.coriolis_parameter(latitude)

    # Calculate baroclinic and barotropic potential vorticity
    pv_baroclinic = metpy.calc.potential_vorticity_baroclinic(potential_temperature, pressure, u, v)
    pv_barotropic = zeta + f

    # Select the 250 hPa level
    pv_baroclinic_250 = pv_baroclinic.sel(level=250)
    pv_barotropic_250 = pv_barotropic.sel(level=250)

    pv_baroclinic_slices_system, pv_barotropic_slices_system = [], []
    for time_step in track.index:
        # Select the time step
        pv_baroclinic_250_time = pv_baroclinic_250.sel(time=time_step)
        pv_barotropic_250_time = pv_barotropic_250.sel(time=time_step)

        # Slice for track limits
        min_lon, max_lon = track.loc[time_step, 'min_lon'], track.loc[time_step, 'max_lon']
        min_lat, max_lat = track.loc[time_step, 'min_lat'], track.loc[time_step, 'max_lat']
        pv_baroclinic_250_time_slice = pv_baroclinic_250_time.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))
        pv_baroclinic_slices_system.append(pv_baroclinic_250_time_slice)
        pv_barotropic_250_time_slice = pv_barotropic_250_time.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))
        pv_barotropic_slices_system.append(pv_barotropic_250_time_slice)
    
    # Calculate the PV composites for this system
    pv_baroclinic_mean = np.mean(pv_baroclinic_slices_system, axis=0)
    pv_barotropic_mean = np.mean(pv_barotropic_slices_system, axis=0)

    # Create a DataArray using an extra dimension for the type of PV
    x, y = np.arange(pv_baroclinic_mean.shape[1]), np.arange(pv_baroclinic_mean.shape[0])
    track_id = int(infile.split('.')[0])

    # Create DataArrays
    da_baroclinic = xr.DataArray(
        pv_baroclinic_mean,
        dims=['y', 'x'],
        coords={'y': y, 'x': x},
        name='baroclinic'
    )

    da_barotropic = xr.DataArray(
        pv_barotropic_mean,
        dims=['y', 'x'],
        coords={'y': y, 'x': x},
        name='barotropic'
    )

    # Combine into a Dataset and add track_id as a coordinate
    ds_pv = xr.Dataset({
        'pv_baroclinic': da_baroclinic,
        'pv_barotropic': da_barotropic
    })
    ds_pv = ds_pv.assign_coords(track_id=track_id)  # Assigning track_id as a coordinate

    return ds_pv

def save_composite(pv_composites, results_directories):
    # Create a composite across all systems
    logging.info("Finished processing all systems. Creating composite...")
    pv_composites = [composite for composite in pv_composites if composite is not None]

    # Concatenate the composites and calculate the mean
    da_pv_composite = xr.concat(pv_composites, dim='track_id')
    ds_pv_composite_mean = da_pv_composite.mean(dim='track_id')

    # Save the composites
    da_pv_composite.to_netcdf('pv_composite_track_ids.nc')
    ds_pv_composite_mean.to_netcdf('pv_composite_mean.nc')
    logging.info("Saved pv_composite_track_ids.nc and pv_composite_mean.nc")

    # Log end time
    end_time = time.time()
    formatted_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    logging.info(f"Finished {len(results_directories)} cases at {formatted_end_time}")

def main():
    # Get CDS API keys
    cdsapi_keys = get_cdsapi_keys()

    # Create a list to store the PV composites for all systems
    pv_composites = []

    # Select track to process
    selected_systems = select_systems("../results_kmeans/")

    # Get all directories in the LEC_RESULTS_DIR
    results_directories = sorted(glob(f'{LEC_RESULTS_DIR}/*'))

    # Convert selected systems to string for easier comparison
    selected_systems_str = [str(system) for system in selected_systems]

    # Filter directories
    filtered_directories = [directory for directory in results_directories if any(system_id in directory for system_id in selected_systems_str)]

    # # Determine the number of CPU cores to use
    # max_cores = os.cpu_count()
    # num_workers = max(1, max_cores - 4) if max_cores else 1
    # logging.info(f"Using {num_workers} CPU cores")
    num_workers = 20

    # Log start time and total number of systems
    start_time = time.time()
    formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    total_systems_count = len(filtered_directories)
    logging.info(f"Starting {total_systems_count} cases at {formatted_start_time}")

    logging.info(f"Using {num_workers} CPU cores for processing")

    pv_composites = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_system, dir_path) for dir_path in filtered_directories]
        for future in as_completed(futures):
            result = future.result()
            if result is not None and np.any(result):
                pv_composites.append(result)

    # Assuming a function to aggregate and save the results
    if pv_composites:
        logging.info("Aggregating and saving PV composites...")
        save_composite(pv_composites, results_directories)

if __name__ == "__main__":
    CDSAPIRC_SUFFIXES = get_cdsapi_keys()
    main()