# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    automate_study_case.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/08 14:17:01 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/14 15:39:49 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import sys
import math
import random
import cdsapi
import logging
import metpy.calc
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from metpy.units import units
from concurrent.futures import ProcessPoolExecutor
from automate_composites import calculate_eady_growth_rate, get_cdsapi_keys,copy_cdsapirc

LEC_RESULTS_DIR = '../../LEC_Results_fixed_framework_test'
OUTPUT_DIR = '../results_nc_files/composites_test_fixed_x_mobile/'


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler('log.study-case.txt', mode='w')])
    logger = logging.getLogger()
    logger.handlers.clear()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler('log.study-case.txt', mode='w')])

def get_lowest_ck_date(results_directory, system_id, tracks_with_periods):
    # Open the results file
    results = sorted(glob(f'{results_directory}/*{system_id}*.csv'))

    # Subset the track data
    track = tracks_with_periods[tracks_with_periods['track_id'] == int(system_id)]
    track_intensification_period = track[track['period'] == 'intensification']
    intensification_dates = pd.to_datetime(track_intensification_period['date'].values)

    # Subset the results file for the intensification period
    df = pd.read_csv(results[0], index_col=0)
    df.index = pd.to_datetime(df.index)
    df_intensification = df.loc[df.index.intersection(intensification_dates)]

    # Get the lowest Ck value during the intensification period
    lowest_ck = df_intensification['Ck'].min()
    lowest_ck_row = df_intensification[df_intensification['Ck'] == lowest_ck]

    # Get the datetime index of the lowest Ck value
    lowest_ck_date = lowest_ck_row.index[0]

    logging.info(f"Lowest Ck value for system {system_id}: {lowest_ck} on {lowest_ck_date}")
    return lowest_ck_date


def get_cdsapi_era5_data(filename: str,
                         track: pd.DataFrame,
                         pressure_levels: list,
                         variables: list,
                         lowest_ck_date: pd.Timestamp) -> xr.Dataset:
    
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
    
    # Use just the date of the lowest Ck value
    lowest_ck_date_timestamp = pd.to_datetime(lowest_ck_date)[0]
    date = lowest_ck_date_timestamp.strftime('%Y%m%d')

    # Convert unique dates to string format for the request
    time_range = f"{date}/{date}"
    initial_hour = (lowest_ck_date_timestamp - pd.Timedelta(hours=1)).strftime('%H')
    final_hour = (lowest_ck_date_timestamp + pd.Timedelta(hours=1)).strftime('%H')

    # Load ERA5 data
    if not os.path.exists(filename):
        print("Retrieving data from CDS API...")
        c = cdsapi.Client(timeout=600)
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "pressure_level": pressure_levels,
                "date": time_range,
                "area": area,
                'time': f'{initial_hour}/to/{final_hour}/by/1',
                "variable": variables,
            }, filename # save file as passed in arguments
        )

        if not os.path.exists(filename):
            raise FileNotFoundError("CDS API file not created.")
        return filename
    
    else:
        print("CDS API file already exists.")
        return filename

def process_results(system_dir, tracks_with_periods, lowest_ck_date, file_path_study_case):
        
    # Get track and periods data
    system_id = os.path.basename(system_dir).split('_')[0] # Get system ID
    logging.info(f"Processing {system_id}")

    # Get track data
    track = tracks_with_periods[tracks_with_periods['track_id'] == int(system_id)]

    # Get ERA5 data for computing PV and EGR
    pressure_levels = ['250', '300', '350', '975', '1000']
    variables = ["u_component_of_wind", "v_component_of_wind", "temperature", "geopotential"]
    tmp_file = os.path.basename(file_path_study_case)
    tmp_file = get_cdsapi_era5_data(tmp_file, track, pressure_levels, variables, lowest_ck_date) 

    # Load the dataset
    ds = xr.open_dataset(tmp_file)

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
    print("Calculating baroclinic and absolute vorticity...")
    pv_baroclinic = metpy.calc.potential_vorticity_baroclinic(potential_temperature, pressure, u, v)
    absolute_vorticity = zeta + f
    print("Done.")

    # Calculate Eady Growth Rate
    print("Calculating Eady Growth Rate...")
    eady_growth_rate = calculate_eady_growth_rate(u, potential_temperature, f, hgt)
    print("Done.")

    # Select variables in their corresponding levels for composites 
    pv_baroclinic_1000 = pv_baroclinic.sel(time=lowest_ck_date).sel(level=1000)
    absolute_vorticity_1000 = absolute_vorticity.sel(time=lowest_ck_date).sel(level=1000)
    eady_growth_rate_1000 = eady_growth_rate.sel(time=lowest_ck_date).isel(level=0)
    u_250, v_250, hgt_250 = u.sel(level=250, time=lowest_ck_date), v.sel(level=250, time=lowest_ck_date), hgt.sel(level=250, time=lowest_ck_date)
    u_1000, v_1000, hgt_1000 = u.sel(level=1000, time=lowest_ck_date), v.sel(level=1000, time=lowest_ck_date), hgt.sel(level=1000, time=lowest_ck_date)

    # Create a DataArray using an extra dimension for the type of PV
    print("Creating DataArray...")
    track_id = int(os.path.basename(system_dir).split('_')[0])

    # Create DataArrays
    da_baroclinic = xr.DataArray(
        pv_baroclinic_1000.values,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': pv_baroclinic_1000.time, 'latitude': lat, 'longitude': lon},
        name='pv_baroclinic',
        attrs={'units': str(pv_baroclinic_1000.metpy.units), 'description': 'PV Baroclinic'}
    )

    da_absolute_vorticity = xr.DataArray(
        absolute_vorticity_1000.values,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': absolute_vorticity_1000.time, 'latitude': lat, 'longitude': lon},
        name='absolute_vorticity',
        attrs={'units': str(absolute_vorticity_1000.metpy.units), 'description': 'Absolute Vorticity'}
    )

    da_edy = xr.DataArray(
        eady_growth_rate_1000.values,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': eady_growth_rate_1000.time, 'latitude': lat, 'longitude': lon},
        name='EGR',
        attrs={'units': str(eady_growth_rate_1000.metpy.units), 'description': 'Eady Growth Rate'}
    )

    da_u_250 = xr.DataArray(
        u_250.values,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': u_250.time, 'latitude': lat, 'longitude': lon},
        name='u_250',
        attrs={'units': str(u_250.metpy.units), 'description': '250 hPa Wind Speed'}
    )

    da_v_250 = xr.DataArray(
        v_250.values,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': v_250.time, 'latitude': lat, 'longitude': lon},
        name='v_250',
        attrs={'units': str(v_250.metpy.units), 'description': '250 hPa Wind Speed'}
    )

    da_u_1000 = xr.DataArray(
        u_1000.values,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': u_1000.time, 'latitude': lat, 'longitude': lon},
        name='u_1000',
        attrs={'units': str(u_1000.metpy.units), 'description': '1000 hPa Wind Speed'}
    )

    da_v_1000 = xr.DataArray(
        v_1000.values,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': v_1000.time, 'latitude': lat, 'longitude': lon},
        name='v_1000',
        attrs={'units': str(v_1000.metpy.units), 'description': '1000 hPa Wind Speed'}
    )

    da_hgt_250 = xr.DataArray(
        hgt_250.values,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': hgt_250.time, 'latitude': lat, 'longitude': lon},
        name='hgt_250',
        attrs={'units': str(hgt_250.metpy.units), 'description': '250 hPa Geopotential Height'}
    )

    da_hgt_1000 = xr.DataArray(
        hgt_1000.values,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': hgt_1000.time, 'latitude': lat, 'longitude': lon},
        name='hgt_1000',
        attrs={'units': str(hgt_1000.metpy.units), 'description': '1000 hPa Geopotential Height'}
    )

    # Combine into a Dataset and add track_id as a coordinate
    ds = xr.Dataset({
        'pv_baroclinic': da_baroclinic,
        'absolute_vorticity': da_absolute_vorticity,
        'EGR': da_edy,
        'u_250': da_u_250,
        'v_250': da_v_250,
        'u_1000': da_u_1000,
        'v_1000': da_v_1000,
        'hgt_250': da_hgt_250,
        'hgt_1000': da_hgt_1000
    })

    # Assigning track_id as a coordinate
    ds = ds.assign_coords(track_id=track_id)  # Assigning track_id as a coordinate
    # Assign date
    ds = ds.assign_coords(time=lowest_ck_date)
    print(f"Finished creating PV composite for {file_path_study_case}")
    return ds

def process_single_case(system_dir, tracks_with_periods):
    system_id = int(os.path.basename(system_dir).split('_')[0])
    logging.info(f"Processing {system_id}")
    print(f"Processing {system_dir}")

    # Process study case
    file_path_study_case = f'{OUTPUT_DIR}/{system_id}_results_study_case.nc'
    lowest_ck_date = get_lowest_ck_date(system_dir, system_id, tracks_with_periods)

    if not os.path.exists(file_path_study_case):
        ds = process_results(system_dir, tracks_with_periods, lowest_ck_date, file_path_study_case)

        # Save study case
        ds.to_netcdf(file_path_study_case)
        logging.info(f"Saved {file_path_study_case}")
        print(f"Finished processing {system_dir}")
        return f"Finished {system_id}"
    
    else:
        logging.info(f"{file_path_study_case} already exists")
        return f"{system_id} already exists"

CDSAPIRC_SUFFIXES = get_cdsapi_keys()

def main():
    setup_logging()

    # Get all directories in the LEC_RESULTS_DIR
    results_directories = sorted(glob(f'{LEC_RESULTS_DIR}/*'))

    completed_systems = glob(f'{OUTPUT_DIR}/*_results_study_case.nc')
    completed_ids = [os.path.basename(dir_path).split('_')[0] for dir_path in completed_systems]

    results_directories = [dir_path for dir_path in results_directories if os.path.basename(dir_path).split('_')[0] not in completed_ids]
    logging.info(f"Found {len(completed_ids)} cases already processed")
    logging.info(f"Found {len(results_directories)} cases to process: {results_directories}")
   
    # Get track and periods data
    tracks_with_periods = pd.read_csv('../tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv')

    # ##### TEST CASE #####
    # test_system = '19910624'
    # test_results_dir = glob(f'{LEC_RESULTS_DIR}/{test_system}*')[0]
    # process_single_case(test_results_dir, tracks_with_periods)
    # sys.exit()

    # Get CPU count 
    max_cores = os.cpu_count()
    num_workers = max(1, max_cores - 4) if max_cores else 1
    logging.info(f"Using {num_workers} CPU cores")

    # Using ProcessPoolExecutor to handle parallel processing
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        futures = [executor.submit(process_single_case, dir_path, tracks_with_periods) for dir_path in results_directories]

        # Collecting results (if necessary)
        for future in futures:
            print(future.result())

 
if __name__ == '__main__':
    main()