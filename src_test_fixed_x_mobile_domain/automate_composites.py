# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    automate_composites.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/06 16:40:35 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/11 00:23:39 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


"""
This script will automate the process of creating composites for PV for each system selected.
"""

import sys
import os
import time
import random
import subprocess
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
from metpy.interpolate import interpolate_1d

# Update logging configuration to use the custom handler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('log.composite.txt', mode='w')])

# REGION = sys.argv[1] # Region to process
LEC_RESULTS_DIR = os.path.abspath('../../LEC_Results_fixed_framework_test')  
CDSAPIRC_PATH = os.path.expanduser('~/.cdsapirc')
OUTPUT_DIR = '../results_nc_files/composites_test_fixed_x_mobile/'

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
    if CDSAPIRC_SUFFIXES:
        chosen_suffix = random.choice(CDSAPIRC_SUFFIXES)
        copy_cdsapirc(chosen_suffix)
        logging.info(f"Switched .cdsapirc file to {chosen_suffix}")
    else:
        logging.error("No .cdsapirc files found. Please check the configuration.")

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
    
def calculate_eady_growth_rate(u, theta, f, hgt):

    # Define new target geopotential heights
    new_geopotential_heights = np.linspace(float(hgt.min()), float(hgt.max()), 6) * units.meters  # Example range

    # Interpolate u from pressure levels to new geopotential heights
    u_on_geopotential_height_levels = interpolate_1d(new_geopotential_heights, hgt, u, axis=1)
    hgt_on_geopotential_height_levels = interpolate_1d(new_geopotential_heights, hgt, hgt, axis=1)

    theta_reordered = theta.transpose('time', 'level', 'latitude', 'longitude')
    theta_on_geopotential_height_levels = interpolate_1d(new_geopotential_heights, hgt, theta_reordered, axis=1)

    # Convert u to DataArray
    u_height_levels = xr.DataArray(u_on_geopotential_height_levels, dims=["time", "level", "latitude", "longitude"],
                    coords={"time": u.time, "level": new_geopotential_heights, "latitude": u.latitude, "longitude": u.longitude})
    hgt_height_levels = xr.DataArray(hgt_on_geopotential_height_levels, dims=["time", "level", "latitude", "longitude"],
                    coords={"time": u.time, "level": new_geopotential_heights, "latitude": u.latitude, "longitude": u.longitude})
    theta_height_levels = xr.DataArray(theta_on_geopotential_height_levels, dims=["time", "level", "latitude", "longitude"],
                    coords={"time": theta.time, "level": new_geopotential_heights, "latitude": theta.latitude, "longitude": theta.longitude})
    
    # Drop NaN values
    u_height_levels = u_height_levels.dropna(dim="level")
    hgt_height_levels = hgt_height_levels.dropna(dim="level")
    theta_height_levels = theta_height_levels.dropna(dim="level")

    # Calculate the derivative of U with respect to log-pressure
    dudz = u_height_levels.differentiate("level") / units('m')

    # Calculate Brunt-Väisälä Frequency (N)
    N = metpy.calc.brunt_vaisala_frequency(hgt_height_levels, theta_height_levels, vertical_dim=1)
    
    # Calculate Eady Growth Rate
    EGR = 0.3098 * (np.abs(f) *  np.abs(dudz)) / N

    # Convert units for simplicity
    EGR = EGR.metpy.convert_units(' 1 / day')

    return EGR

def create_pv_composite(infile, track):
    logging.info(f"Creating PV composite for {infile}")
    print(f"Creating PV composite for {infile}")

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
    eady_growth_rate = calculate_eady_growth_rate(u, potential_temperature, f, hgt)
    logging.info("Done.")

    # Select the 250 hPa level
    pv_baroclinic_250 = pv_baroclinic.sel(level=250)
    absolute_vorticity_1000 = absolute_vorticity.sel(level=250)
    eady_growth_rate_400 = eady_growth_rate.isel(level=0) # 1000 hPa level is the 1st vertical level
    
    # Calculate the composites for this system
    logging.info("Calculating means...")
    pv_baroclinic_mean = pv_baroclinic_250.mean(dim='time')
    absolute_vorticity_mean = absolute_vorticity_1000.mean(dim='time')
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
        name='pv_baroclinic',
        attrs={'units': str(pv_baroclinic_mean.metpy.units)},
    )

    da_absolute_vorticity = xr.DataArray(
        absolute_vorticity_mean.values,
        dims=['latitude', 'longitude'],
        coords={'latitude': lat, 'longitude': lon},
        name='absolute_vorticity',
        attrs={'units': str(absolute_vorticity_mean.metpy.units)},
    )

    da_edy = xr.DataArray(
        eady_growth_rate_mean.values,
        dims=['latitude', 'longitude'],
        coords={'latitude': lat, 'longitude': lon},
        name='EGR',
        attrs={'units': str(eady_growth_rate_mean.metpy.units)},
    )

    # Combine into a Dataset and add track_id as a coordinate
    ds_composite = xr.Dataset({
        'pv_baroclinic': da_baroclinic,
        'absolute_vorticity': da_absolute_vorticity,
        'EGR': da_edy})

    # Assigning track_id as a coordinate
    ds_composite = ds_composite.assign_coords(track_id=track_id)  # Assigning track_id as a coordinate
    logging.info(f"Finished creating PV composite for {infile}")

    return ds_composite

def find_smallest_domain(composites):
    """
    Finds the smallest domain that includes all given composites by calculating the intersection
    of their latitude and longitude ranges.
    """
    if not composites:
        return None, None

    # Initialize with the range of the first composite
    min_lat, max_lat = composites[0].latitude.min(), composites[0].latitude.max()
    min_lon, max_lon = composites[0].longitude.min(), composites[0].longitude.max()

    # Update the bounds based on all other composites
    for comp in composites[1:]:
        comp_min_lat, comp_max_lat = comp.latitude.min(), comp.latitude.max()
        comp_min_lon, comp_max_lon = comp.longitude.min(), comp.longitude.max()

        # Find the intersection of the ranges
        min_lat = max(min_lat, comp_min_lat)
        max_lat = min(max_lat, comp_max_lat)
        min_lon = max(min_lon, comp_min_lon)
        max_lon = min(max_lon, comp_max_lon)

        # Check if there's no overlap
        if min_lat > max_lat or min_lon > max_lon:
            logging.warning("No overlapping region found among composites.")
            return None, None

    # Return the smallest domain's latitude and longitude ranges
    common_lat = np.arange(min_lat, max_lat + 0.25, 0.25)  # Increment might need adjustment based on actual data resolution
    common_lon = np.arange(min_lon, max_lon + 0.25, 0.25)

    return common_lat, common_lon

def find_largest_domain(composites):
    max_lat = max([comp.latitude.max() for comp in composites])
    min_lat = min([comp.latitude.min() for comp in composites])
    max_lon = max([comp.longitude.max() for comp in composites])
    min_lon = min([comp.longitude.min() for comp in composites])

    return np.arange(min_lat, max_lat + 0.25, 0.25), np.arange(min_lon, max_lon + 0.25, 0.25)
    
def interpolate_to_common_grid(composites, common_lat, common_lon):
    interpolated_composites = []
    for composite in composites:
        interpolated = composite.interp(latitude=common_lat, longitude=common_lon, method='cubic')
        interpolated_composites.append(interpolated)
    return interpolated_composites

def compute_mean_composite(interpolated_composites):
    # Concatenate the datasets along a new dimension
    combined = xr.concat(interpolated_composites, dim='composite')
    # Compute the mean along the new dimension
    mean_composite = combined.mean(dim='composite')
    return mean_composite

def save_composite(composites, total_systems_count, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    if not composites:
        logging.info("No valid composites found. Skipping composite creation.")
        return

    common_lat, common_lon = find_smallest_domain(composites)
    interpolated_composites = interpolate_to_common_grid(composites, common_lat, common_lon)
    mean_composite = compute_mean_composite(interpolated_composites)

    mean_composite.to_netcdf(os.path.join(output_dir, 'pv_egr_mean_composite.nc'))
    logging.info("Saved pv_egr_mean_composite.nc")
    
    end_time = time.time()
    formatted_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    logging.info(f"Finished {total_systems_count} cases at {formatted_end_time}")

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
    pressure_levels = ['250', '300', '350', '975', '1000']
    variables = ["u_component_of_wind", "v_component_of_wind", "temperature", "geopotential"]
    infile_pv_egr = get_cdsapi_era5_data(f'{system_id}-pv-egr', track, pressure_levels, variables) 

    # Make PV composite
    ds_composite = create_pv_composite(infile_pv_egr, track) 
    logging.info(f"Processing completed for {system_dir}")

    # # Delete infile
    # if os.path.exists(infile_pv_egr):
    #     os.remove(infile_pv_egr)

    return ds_composite 

CDSAPIRC_SUFFIXES = get_cdsapi_keys()

def main():

    # Get all directories in the LEC_RESULTS_DIR
    results_directories = sorted(glob(f'{LEC_RESULTS_DIR}/*'))[:3]

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
        save_composite(composites, total_systems_count, OUTPUT_DIR)

if __name__ == "__main__":
    main()