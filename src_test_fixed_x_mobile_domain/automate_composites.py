# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    automate_composites.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/06 16:40:35 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/07 17:52:56 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


"""
This script will automate the process of creating composites for PV for each system selected.
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
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
                    handlers=[logging.FileHandler('log.composite.txt', mode='w')])

# REGION = sys.argv[1] # Region to process
LEC_RESULTS_DIR = os.path.abspath('../../LEC_Results_fixed_framework_test')  # Get absolute PATH

def get_cdsapi_era5_data(filename: str, track: pd.DataFrame, pressure_levels: list, variables: list) -> xr.Dataset:

    track.set_index('date', inplace=True)
    track.index = pd.to_datetime(track.index)

    # Extract bounding box (lat/lon limits) from track
    min_lat, max_lat = track['lat vor'].min(), track['lat vor'].max()
    min_lon, max_lon = track['lon vor'].min(), track['lon vor'].max()

    # Apply a 15-degree buffer and round to nearest integer
    buffered_max_lat = math.ceil(max_lat + 15)
    buffered_min_lon = math.floor(min_lon - 15)
    buffered_min_lat = math.floor(min_lat - 15)
    buffered_max_lon = math.ceil(max_lon + 15)

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
    time_step = str(int((track.index[1] - track.index[0]).total_seconds() / 3600))
    time_step = '3' if time_step < '3' else time_step

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
    
def calculate_eady_growth_rate(u, theta, pressure, f, hgt):
    # Calculate the derivative of U with respect to log-pressure
    dudp = u.differentiate("level")
    
    # Calculate the derivative of theta with respect to log-pressure
    dthetadp = theta.sortby("level", ascending=True).differentiate("level")
    
    # Calculate Brunt-Väisälä Frequency (N)
    N = metpy.calc.brunt_vaisala_frequency(hgt, theta)
    
    # Calculate Eady Growth Rate
    EGR = 0.3098 * (np.abs(f) *  np.abs(dudp)) / N

    return EGR

def create_pv_composite(infile, track):
    logging.info(f"Creating PV composite for {infile}")

    # Load the dataset
    ds = xr.open_dataset(infile)

    # Open variables for calculations and assign units
    temperature = ds['t'] * units.kelvin
    pressure = ds.level * units.hPa
    u = ds['u'] * units('m/s')
    v = ds['v'] * units('m/s')
    hgt = (ds['z'] / 9.8) * units('gpm')
    latitude = ds.latitude
    lat, lon = ds.latitude.values, ds.longitude.values

    # Calculate potential temperature, vorticity and Coriolis parameter
    potential_temperature = metpy.calc.potential_temperature(pressure, temperature)
    zeta = metpy.calc.vorticity(u, v)
    f = metpy.calc.coriolis_parameter(latitude)

    # Calculate baroclinic and absolute vorticity
    logging.info("Calculating baroclinic and absolute vorticity...")
    pv_baroclinic = metpy.calc.potential_vorticity_baroclinic(potential_temperature, pressure, u, v)
    absolute_vorticity = zeta + f
    logging.info("Done.")

    # Calculate Eady Growth Rate
    logging.info("Calculating Eady Growth Rate...")
    eady_growth_rate = calculate_eady_growth_rate(u, potential_temperature, pressure, f, hgt)
    logging.info("Done.")

    # Select the 250 hPa level
    pv_baroclinic_250 = pv_baroclinic.sel(level=250)
    absolute_vorticity_250 = absolute_vorticity.sel(level=250)
    eady_growth_rate_400 = eady_growth_rate.sel(level=250)
    
    # Calculate the composites for this system
    logging.info("Calculating means...")
    pv_baroclinic_mean = pv_baroclinic_250.mean(dim='time')
    absolute_vorticity_mean = absolute_vorticity_250.mean(dim='time')
    eady_growth_rate_mean = eady_growth_rate_400.mean(dim='time')
    logging.info("Done.")

    # Create a DataArray using an extra dimension for the type of PV
    logging.info("Creating DataArray...")
    x, y = np.arange(pv_baroclinic_mean.shape[1]), np.arange(pv_baroclinic_mean.shape[0])
    track_id = int(infile.split('.')[0].split('-')[0])

    # Create DataArrays
    da_baroclinic = xr.DataArray(
        pv_baroclinic_mean.values,
        dims=['latitude', 'longitude'],
        coords={'latitude': lat, 'longitude': lon},
        name='pv_baroclinic'
    )

    da_absolute_vorticity = xr.DataArray(
        absolute_vorticity_mean.values,
        dims=['latitude', 'longitude'],
        coords={'latitude': lat, 'longitude': lon},
        name='absolute_vorticity'
    )

    da_edy = xr.DataArray(
        eady_growth_rate_mean.values,
        dims=['latitude', 'longitude'],
        coords={'latitude': lat, 'longitude': lon},
        name='EGR'
    )

    # Combine into a Dataset and add track_id as a coordinate
    ds_composite = xr.Dataset({
        'pv_baroclinic': da_baroclinic,
        'absolute_vorticity': da_absolute_vorticity,
        'EGR': da_edy
    })

    # Assigning track_id as a coordinate
    ds_composite = ds_composite.assign_coords(track_id=track_id)  # Assigning track_id as a coordinate
    logging.info(f"Finished creating PV composite for {infile}")

    return ds_composite

# def save_composite(composites, total_systems_count):
#     # Create a composite across all systems
#     logging.info("Finished processing all systems. Creating composite...")
#     composites = [composite for composite in composites if composite is not None]

#     # Concatenate the composites and calculate the mean
#     da_composite = xr.concat(composites, dim='track_id')

#     # Calculate the mean
#     ds_composite_mean = da_composite.mean(dim='track_id')

#     # Save the composites
#     da_composite.to_netcdf('pv_egr_composite_track_ids_fixed.nc')
#     ds_composite_mean.to_netcdf('pv_egr_composite_mean_fixed.nc')
#     logging.info("Saved pv_egr_composite_track_ids.nc and pv_egr_composite_mean.nc")

#     # Log end time
#     end_time = time.time()
#     formatted_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
#     logging.info(f"Finished {total_systems_count} cases at {formatted_end_time}")

def save_composite(composites, total_systems_count):
    # Create a composite across all systems
    logging.info("Finished processing all systems. Creating composite...")
    composites = [composite for composite in composites if composite is not None]

    if not composites:
        logging.info("No valid composites found. Skipping composite creation.")
        return

    # Calculate weights based on spatial extent (area)
    weights = [composite.latitude.size * composite.longitude.size for composite in composites]
    weights /= np.sum(weights)

    # Apply weighted average
    weighted_composite = create_weighted_composite(composites, weights)

    # Save the composite
    weighted_composite.to_netcdf('pv_egr_weighted_composite.nc')
    logging.info("Saved pv_egr_weighted_composite.nc")

    # Log end time
    end_time = time.time()
    formatted_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    logging.info(f"Finished {total_systems_count} cases at {formatted_end_time}")

def create_weighted_composite(composites, weights):
    # Normalize weights to sum to 1
    weights /= np.sum(weights)
    
    # Initialize empty arrays to store weighted variables
    weighted_variables = []
    
    # Apply weights to each variable in each composite
    for composite, weight in zip(composites, weights):
        weighted_variables.append(composite * weight)
    
    # Combine weighted variables into composite
    composite = sum(weighted_variables)
    
    return composite

def process_system(system_dir, tracks_with_periods):
    """
    Process the selected system.
    """
    # Get track and periods data
    system_id = os.path.basename(system_dir).split('_')[0] # Get system ID
    logging.info(f"Processing {system_id}")

    # Get track data
    sliced_track = tracks_with_periods[tracks_with_periods['track_id'] == int(system_id)]
    if sliced_track.empty:
        logging.info(f"No track data for {system_dir}")
        return None

    # Filter for intensification phase only
    intensification_start = sliced_track[sliced_track['period'] == 'intensification']['date'].min()
    intensification_end = sliced_track[sliced_track['period'] == 'intensification']['date'].max()
    track = sliced_track[(sliced_track['date'] >= intensification_start) & (sliced_track['date'] <= intensification_end)]
    if track.empty:
        logging.info(f"No intensification phase data for {system_dir}")
        return None

    # Get ERA5 data for computing PV and EGR
    pressure_levels = ['200', '250', '300', '350', '400', '450']
    variables = ["u_component_of_wind", "v_component_of_wind", "temperature", "geopotential"]
    infile_pv_egr = get_cdsapi_era5_data(f'{system_id}-pv-egr', track, pressure_levels, variables) 

    # Make PV composite
    ds_composite = create_pv_composite(infile_pv_egr, track) 
    logging.info(f"Processing completed for {system_dir}")

    # Delete infile
    # if os.path.exists(infile_pv_egr):
    #     os.remove(infile_pv_egr)

    return ds_composite 

def main():

    # Get all directories in the LEC_RESULTS_DIR
    results_directories = sorted(glob(f'{LEC_RESULTS_DIR}/*'))

    # Get track and periods data
    tracks_with_periods = pd.read_csv('../tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv')

    # # Determine the number of CPU cores to use
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    else:
        max_cores = os.cpu_count()
        num_workers = max(1, max_cores - 4) if max_cores else 1
        logging.info(f"Using {num_workers} CPU cores")

    # Log start time and total number of systems
    start_time = time.time()
    formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    total_systems_count = len(results_directories)
    logging.info(f"Starting {total_systems_count} cases at {formatted_start_time}")

    logging.info(f"Using {num_workers} CPU cores for processing")

    composites = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_system, dir_path, tracks_with_periods) for dir_path in results_directories]
        for future in as_completed(futures):
            result = future.result()
            if result is not None and np.any(result):
                composites.append(result)

    # Assuming a function to aggregate and save the results
    if composites:
        logging.info("Aggregating and saving PV composites...")
        save_composite(composites, total_systems_count)

if __name__ == "__main__":
    main()