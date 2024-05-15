# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    automate_composites.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/24 14:42:50 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/15 12:08:13 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


"""
This script will automate the process of creating composites for PV for each system selected.
"""

import sys
import os
import time
import math
import random
import cdsapi
import logging
import subprocess
import metpy.calc
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from metpy.units import units
from metpy.constants import Rd, g, Cp_d
from metpy.interpolate import interpolate_1d, log_interpolate_1d


# Update logging configuration to use the custom handler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('log.automate_run_LEC.txt', mode='w')])

# REGION = sys.argv[1] # Region to process
LEC_RESULTS_DIR = os.path.abspath('../../LEC_Results_energetic-patterns')  # Get absolute PATH
CDSAPIRC_PATH = os.path.expanduser('~/.cdsapirc')
OUTPUT_DIR = '../results_nc_files/composites_barotropic_baroclinic/'

DEBUG_CODE = False
DEBUG_CDSAPI = True

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
    if DEBUG_CDSAPI == False:
        if CDSAPIRC_SUFFIXES:
            chosen_suffix = random.choice(CDSAPIRC_SUFFIXES)
            copy_cdsapirc(chosen_suffix)
            logging.info(f"Switched .cdsapirc file to {chosen_suffix}")
        else:
            logging.error("No .cdsapirc files found. Please check the configuration.")

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

def calculate_eady_growth_rate(u, potential_temperature, f, hgt):

    # Create a 3D array for pressure
    time = u.coords['time']
    latitude = u.coords['latitude']
    longitude = u.coords['longitude']
    level = u.coords['level']
    pressure_levels = u.level.values * units.hPa
    pressure = create_pressure_array(pressure_levels, time, latitude, longitude, level) * units.hPa
    pressure = pressure.transpose(*u.dims)

    # Define new target geopotential heights
    new_geopotential_heights = np.linspace(float(hgt.min()) * 0.8 , float(hgt.max()) * 0.8, len(u.level)) * units.meters  # Example range

    # Interpolate u from pressure levels to new geopotential heights
    u_on_geopotential_height_levels = interpolate_1d(new_geopotential_heights, hgt, u, axis=1)
    hgt_on_geopotential_height_levels = interpolate_1d(new_geopotential_heights, hgt, hgt, axis=1)
    pressure_on_geopotential_height_levels = interpolate_1d(new_geopotential_heights, hgt, pressure, axis=1)

    theta_reordered = potential_temperature.transpose('time', 'level', 'latitude', 'longitude')
    theta_on_geopotential_height_levels = interpolate_1d(new_geopotential_heights, hgt, theta_reordered, axis=1)

    # Convert u to DataArray
    u_height_levels = xr.DataArray(u_on_geopotential_height_levels, dims=["time", "level", "latitude", "longitude"],
                    coords={"time": u.time, "level": new_geopotential_heights, "latitude": u.latitude, "longitude": u.longitude})
    hgt_height_levels = xr.DataArray(hgt_on_geopotential_height_levels, dims=["time", "level", "latitude", "longitude"],
                    coords={"time": u.time, "level": new_geopotential_heights, "latitude": u.latitude, "longitude": u.longitude})
    theta_height_levels = xr.DataArray(theta_on_geopotential_height_levels, dims=["time", "level", "latitude", "longitude"],
                    coords={"time": potential_temperature.time, "level": new_geopotential_heights, "latitude": potential_temperature.latitude, "longitude": potential_temperature.longitude})
    pressure_height_levels = xr.DataArray(pressure_on_geopotential_height_levels, dims=["time", "level", "latitude", "longitude"],
                    coords={"time": u.time, "level": new_geopotential_heights, "latitude": u.latitude, "longitude": u.longitude})

    # Calculate the derivative of U with respect to log-pressure
    dudz = u_height_levels.differentiate("level") / units('m')

    # Calculate Brunt-Väisälä Frequency (N)
    N = metpy.calc.brunt_vaisala_frequency(hgt_height_levels, theta_height_levels, vertical_dim=1)
    
    # Calculate Eady Growth Rate
    EGR = 0.3098 * (np.abs(f) *  np.abs(dudz)) / N

    # Convert units for simplicity
    EGR = EGR.metpy.convert_units(' 1 / day')

    # Make data dimensions match
    EGR = EGR.transpose(*u.dims)
    hgt_height_levels = hgt_height_levels.transpose(*u.dims)

    # Prepare data for interpolation
    pressure_quant = units.Quantity(pressure_height_levels.values, 'hPa')
    hgt_quant = units.Quantity(hgt_height_levels.values, 'meter')
    EGR_quant = units.Quantity(EGR.values, '1 / day')

    # Interpolate EGR to the pressure levels
    height, EGR_on_pressure_levels = log_interpolate_1d(pressure_levels, pressure_quant, hgt_quant, EGR_quant, axis=1)
    
    # Convert interpolated EGR to DataArray
    EGR_pressure_levels = xr.DataArray(EGR_on_pressure_levels, dims=["time", "level", "latitude", "longitude"],
                    coords={"time": u.time, "level": pressure_levels, "latitude": u.latitude, "longitude": u.longitude},
                    name="EGR")

    # Interpolate missing values
    EGR_pressure_levels_interp = EGR_pressure_levels.interpolate_na(dim='longitude', method='linear')

    return EGR_pressure_levels


def create_pv_composite(infile, track):
    # Load the dataset
    ds = xr.open_dataset(infile)

    # Open variables for calculations and assign units
    temperature = ds['t'] * units.kelvin
    pressure = ds.level * units.hPa
    u = ds['u'] * units('m/s')
    v = ds['v'] * units('m/s')
    hgt = (ds['z'] / 9.8) * units('gpm')
    latitude = ds.latitude

    # Calculate potential temperature, vorticity and Coriolis parameter
    potential_temperature = metpy.calc.potential_temperature(pressure, temperature)
    zeta = metpy.calc.vorticity(u, v)
    f = metpy.calc.coriolis_parameter(latitude)

    # Calculate baroclinic and absolute vorticity
    pv_baroclinic = metpy.calc.potential_vorticity_baroclinic(potential_temperature, pressure, u, v)
    absolute_vorticity = zeta + f

    # Calculate Eady Growth Rate
    eady_growth_rate = calculate_eady_growth_rate(u, potential_temperature, f, hgt)

    # Empty lists to store the slices
    pv_baroclinic_slices_system, absolute_vorticity_slices_system = [], []
    eady_growth_rate_slices_system = []
    u_slices_system, v_slices_system, hgt_slices_system = [], [], []

    for time_step in track.index:
        # Select the time step
        pv_baroclinic_time = pv_baroclinic.sel(time=time_step)
        absolute_vorticity_time = absolute_vorticity.sel(time=time_step)
        eady_growth_rate_time = eady_growth_rate.sel(time=time_step)
        u_time, v_time, hgt_time = u.sel(time=time_step), v.sel(time=time_step), hgt.sel(time=time_step)

        # Select the track limits
        min_lon, max_lon = track.loc[time_step, 'min_lon'], track.loc[time_step, 'max_lon']
        min_lat, max_lat = track.loc[time_step, 'min_lat'], track.loc[time_step, 'max_lat']

        # Slice for track limits
        pv_baroclinic_time_slice = pv_baroclinic_time.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))
        pv_baroclinic_slices_system.append(pv_baroclinic_time_slice)

        absolute_vorticity_time_slice = absolute_vorticity_time.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))
        absolute_vorticity_slices_system.append(absolute_vorticity_time_slice)

        eady_growth_rate_time_slice = eady_growth_rate_time.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))
        eady_growth_rate_slices_system.append(eady_growth_rate_time_slice)

        u_time_slice = u_time.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))
        v_time_slice = v_time.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))
        hgt_time_slice = hgt_time.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))
        u_slices_system.append(u_time_slice)
        v_slices_system.append(v_time_slice)
        hgt_slices_system.append(hgt_time_slice)
    
    # Calculate the composites for this system
    pv_baroclinic_mean = np.mean(pv_baroclinic_slices_system, axis=0)
    absolute_vorticity_mean = np.mean(absolute_vorticity_slices_system, axis=0)
    eady_growth_rate_mean = np.mean(eady_growth_rate_slices_system, axis=0)

    u_mean = np.mean(u_slices_system, axis=0)
    v_mean = np.mean(v_slices_system, axis=0)
    hgt_mean = np.mean(hgt_slices_system, axis=0)

    # Create a DataArray using an extra dimension for the type of PV
    x_size, y_size = pv_baroclinic_mean.shape[2], pv_baroclinic_mean.shape[1]
    x = np.linspace(- x_size / 2, (x_size / 2) - 1 , x_size)
    y = np.linspace(- y_size / 2, (y_size / 2) - 1, y_size)
    level = u.level
    track_id = int(infile.split('.')[0].split('-')[0])

    # Create DataArrays
    da_baroclinic = xr.DataArray(
        pv_baroclinic_mean,
        dims=['level', 'y', 'x'],
        coords={'level': level, 'y': y, 'x': x},
        name='pv_baroclinic',
        attrs={'units': str(pv_baroclinic_time_slice.metpy.units)}
    )

    da_absolute_vorticity = xr.DataArray(
        absolute_vorticity_mean,
        dims=['level', 'y', 'x'],
        coords={'level': level, 'y': y, 'x': x},
        name='absolute_vorticity',
        attrs={'units': str(absolute_vorticity_time_slice.metpy.units)}
    )

    da_edy = xr.DataArray(
        eady_growth_rate_mean,
        dims=['level', 'y', 'x'],
        coords={'level': level, 'y': y, 'x': x},
        name='EGR',
        attrs={'units': str(eady_growth_rate_time_slice.metpy.units)}
    )

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

    # Combine into a Dataset and add track_id as a coordinate
    ds_composite = xr.Dataset({
        'pv_baroclinic': da_baroclinic,
        'absolute_vorticity': da_absolute_vorticity,
        'EGR': da_edy,
        'u': da_u,
        'v': da_v,
        'hgt': da_hgt,
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
    da_composite.to_netcdf(f'{output_dir}/pv_egr_composite_track_ids.nc')
    ds_composite_mean.to_netcdf(f'{output_dir}/pv_egr_composite_mean.nc')
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
    pressure_levels = ['250', '300', '350', '550', '500', '450', '700', '750', '800', '950', '975', '1000']
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
        logging.info("Aggregating and saving PV composites...")
        save_composite(composites, OUTPUT_DIR, total_systems_count)

if __name__ == "__main__":
    main()