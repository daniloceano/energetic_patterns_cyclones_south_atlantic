# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    automate_composites.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/24 14:42:50 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/02 18:01:56 by daniloceano      ###   ########.fr        #
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
                    handlers=[logging.FileHandler('log.automate_run_LEC.txt', mode='w')])

# REGION = sys.argv[1] # Region to process
LEC_RESULTS_DIR = os.path.abspath('../../LEC_Results_energetic-patterns')  # Get absolute PATH

def get_cdsapi_era5_data(filename: str, track: pd.DataFrame, pressure_levels: list, variables: list) -> xr.Dataset:

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
    
def calculate_eady_growth_rate(u, theta, pressure, f):
    # Calculate the derivative of U with respect to log-pressure
    dudp = u.differentiate("level")
    
    # Calculate the derivative of theta with respect to log-pressure
    dthetadp = theta.differentiate("level")
    
    # Calculate Brunt-Väisälä Frequency (N)
    N = np.sqrt((9.81 / theta) * (-dthetadp))
    
    # Calculate Eady Growth Rate
    EGR = 0.31 * (np.abs(f) / N) * np.abs(dudp)

    return EGR

def calculate_thermal_wind(ds, lower_level, upper_level):
    """
    Calculate the thermal wind between two pressure levels.

    Parameters:
        ds (xarray.Dataset): Dataset containing the temperature and geopotential height.
        lower_level (int): Lower pressure level in hPa.
        upper_level (int): Upper pressure level in hPa.

    Returns:
        xarray.DataArray: Thermal wind vector components (u, v) at mid-level between input levels.
    """
    # Ensure the data is properly attached with units
    ds = ds.metpy.quantify()

    # Select the temperature and geopotential height at specified levels
    T_lower = ds['t'].metpy.sel(vertical=lower_level * units.hPa)
    T_upper = ds['t'].metpy.sel(vertical=upper_level * units.hPa)
    Z_lower = ds['z'].metpy.sel(vertical=lower_level * units.hPa)
    Z_upper = ds['z'].metpy.sel(vertical=upper_level * units.hPa)

    # Calculate the temperature gradient
    grad_T_lower = metpy.calc.gradient(T_lower, coordinates=['lat', 'lon'])
    grad_T_upper = metpy.calc.gradient(T_upper, coordinates=['lat', 'lon'])

    # Average the temperature gradients
    avg_grad_T = (grad_T_lower + grad_T_upper) / 2

    # Calculate the geopotential height difference
    delta_Z = Z_upper - Z_lower

    # Compute the Coriolis parameter (f)
    f = metpy.calc.coriolis_parameter(ds['lat'])

    # Calculate the thermal wind
    thermal_wind = (f / 9.81 * np.cross(avg_grad_T, [0, 0, 1]) * delta_Z).to_base_units()

    return thermal_wind

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

    # Calculate baroclinic and absolute vorticity
    pv_baroclinic = metpy.calc.potential_vorticity_baroclinic(potential_temperature, pressure, u, v)
    absolute_vorticity = zeta + f

    # Calculate Eady Growth Rate
    eady_growth_rate = calculate_eady_growth_rate(u, potential_temperature, pressure, f)

    # Select the 250 hPa level
    pv_baroclinic_250 = pv_baroclinic.sel(level=250)
    absolute_vorticity_250 = absolute_vorticity.sel(level=250)
    eady_growth_rate_400 = eady_growth_rate.sel(level=250)

    # Empty lists to store the slices
    pv_baroclinic_slices_system, absolute_vorticity_slices_system = [], []
    eady_growth_rate_slices_system = []

    for time_step in track.index:
        # Select the time step
        pv_baroclinic_250_time = pv_baroclinic_250.sel(time=time_step)
        absolute_vorticity_250_time = absolute_vorticity_250.sel(time=time_step)
        eady_growth_rate_400_time = eady_growth_rate_400.sel(time=time_step)

        # Select the track limits
        min_lon, max_lon = track.loc[time_step, 'min_lon'], track.loc[time_step, 'max_lon']
        min_lat, max_lat = track.loc[time_step, 'min_lat'], track.loc[time_step, 'max_lat']

        # Slice for track limits
        pv_baroclinic_250_time_slice = pv_baroclinic_250_time.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))
        pv_baroclinic_slices_system.append(pv_baroclinic_250_time_slice)

        absolute_vorticity_250_time_slice = absolute_vorticity_250_time.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))
        absolute_vorticity_slices_system.append(absolute_vorticity_250_time_slice)

        eady_growth_rate_400_time_slice = eady_growth_rate_400_time.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))
        eady_growth_rate_slices_system.append(eady_growth_rate_400_time_slice)
    
    # Calculate the composites for this system
    pv_baroclinic_mean = np.mean(pv_baroclinic_slices_system, axis=0)
    absolute_vorticity_mean = np.mean(absolute_vorticity_slices_system, axis=0)
    eady_growth_rate_mean = np.mean(eady_growth_rate_slices_system, axis=0)

    # Create a DataArray using an extra dimension for the type of PV
    x, y = np.arange(pv_baroclinic_mean.shape[1]), np.arange(pv_baroclinic_mean.shape[0])
    track_id = int(infile.split('.')[0].split('-')[0])

    # Create DataArrays
    da_baroclinic = xr.DataArray(
        pv_baroclinic_mean,
        dims=['y', 'x'],
        coords={'y': y, 'x': x},
        name='pv_baroclinic'
    )

    da_absolute_vorticity = xr.DataArray(
        absolute_vorticity_mean,
        dims=['y', 'x'],
        coords={'y': y, 'x': x},
        name='absolute_vorticity'
    )

    da_edy = xr.DataArray(
        eady_growth_rate_mean,
        dims=['y', 'x'],
        coords={'y': y, 'x': x},
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

    return ds_composite

def save_composite(composites, total_systems_count):
    # Create a composite across all systems
    logging.info("Finished processing all systems. Creating composite...")
    composites = [composite for composite in composites if composite is not None]

    # Concatenate the composites and calculate the mean
    da_composite = xr.concat(composites, dim='track_id')
    ds_composite_mean = da_composite.mean(dim='track_id')

    # Save the composites
    da_composite.to_netcdf('pv_egr_composite_track_ids.nc')
    ds_composite_mean.to_netcdf('pv_egr_composite_mean.nc')
    logging.info("Saved pv_egr_composite_track_ids.nc and pv_egr_composite_mean.nc")

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

    system_id = os.path.basename(system_dir).split('_')[0] # Get system ID

    # Get ERA5 data for computing PV and EGR
    pressure_levels = ['200', '250', '300', '350', '400', '450']
    variables = ["u_component_of_wind", "v_component_of_wind", "temperature"]
    infile_pv_egr = get_cdsapi_era5_data(f'{system_id}-pv-egr', track, pressure_levels, variables) 

    # # Get ERA5 data for computing Thermal Wind
    # pressure_levels = ['900', '600', '300']
    # variables = ["temperature", "geopotential"]
    # infile_thermal_wind = get_cdsapi_era5_data(f'{system_id}-pv-egr', track, pressure_levels, variables) 

    # Make PV composite
    ds_composite = create_pv_composite(infile_pv_egr, track) 
    logging.info(f"Processing completed for {system_dir}")

    # Delete infile
    if os.path.exists(infile_pv_egr):
        os.remove(infile_pv_egr)

    # if os.path.exists(infile_thermal_wind):
    #     os.remove(infile_thermal_wind)

    return ds_composite 

def main():
    # Create a list to store the PV composites for all systems
    pv_composites = []

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
        logging.info("Aggregating and saving PV composites...")
        save_composite(composites, total_systems_count)

if __name__ == "__main__":
    main()