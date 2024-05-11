# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    automate_study_case.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/08 14:17:01 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/11 17:39:19 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import math
import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob

import metpy.calc
from metpy.units import units

from automate_composites import calculate_eady_growth_rate


LEC_RESULTS_DIR = '../../LEC_Results_fixed_framework_test'
OUTPUT_DIR = '../results_nc_files/composites_test_fixed_x_mobile/'

def choose_study_case(LEC_RESULTS_DIR, tracks_with_periods):
    """
    Determine which case had the lowest Ck values for plotting.
    """
    min_ck_values = {}
    
    for directory in os.listdir(LEC_RESULTS_DIR):
        if 'ERA5_fixed' in directory:
            results_file = glob(f"{LEC_RESULTS_DIR}/{directory}/*fixed_results.csv")[0]
        else:
            continue
        track_id = directory.split('_')[0]
        
        track = tracks_with_periods[tracks_with_periods['track_id'] == int(track_id)]
        track_intensification_period = track[track['period'] == 'intensification']
        intensification_dates = pd.to_datetime(track_intensification_period['date'].values)

        df = pd.read_csv(results_file, index_col=0)
        df.index = pd.to_datetime(df.index)
        df_intensification = df.loc[df.index.intersection(intensification_dates)]

        # Find the maximum Ck value during the intensification period
        if not df_intensification.empty:
            min_ck = df_intensification['Ck'].min()
            min_ck_values[track_id] = min_ck
    
    # Find the track ID with the highest Ck value
    if min_ck_values:
        # System id with the lowest Ck value and corresponding Ck value
        lowest_ck_track_id = min(min_ck_values, key=min_ck_values.get)
        lowest_ck_value = min_ck_values[lowest_ck_track_id]

        # Find the track corresponding to the system with lowest Ck value
        lowest_track = tracks_with_periods[tracks_with_periods['track_id'] == int(lowest_ck_track_id)]
        
        # Determine in which date and time the highest Ck value occurred
        results_file = glob(f"{LEC_RESULTS_DIR}/{lowest_ck_track_id}*/*fixed_results.csv")[0]
        df = pd.read_csv(results_file, index_col=0)
        df.index = pd.to_datetime(df.index)

        lowest_ck_date = df[df['Ck'] == lowest_ck_value].index[0]

        return lowest_track, lowest_ck_date
    else:
        return None, None

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
    lowest_ck_date = pd.to_datetime(df[df['Ck'] == lowest_ck].index)
    return lowest_ck_date

def get_cdsapi_era5_data(filename: str,
                         track: pd.DataFrame,
                         pressure_levels: list,
                         variables: list,
                         lowest_ck_date: pd.Timestamp) -> xr.Dataset:

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
    date = lowest_ck_date.strftime('%Y%m%d')
    hour = lowest_ck_date.strftime('%H')

    # Convert unique dates to string format for the request
    time_range = f"{date}/{date}"
    initial_hour = (lowest_ck_date - pd.Timedelta(hours=1)).strftime('%H')
    final_hour = (lowest_ck_date + pd.Timedelta(hours=1)).strftime('%H')

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

def process_system(system_dir, file_study_case, tracks_with_periods, lowest_ck_date):
    """
    Process the selected system.
    """
    # Get track and periods data
    system_id = os.path.basename(system_dir).split('_')[0] # Get system ID
    print(f"Processing {system_id}")

    # Get track data
    sliced_track = tracks_with_periods[tracks_with_periods['track_id'] == int(system_id)]
    if sliced_track.empty:
        print(f"No track data for {system_dir}")
        return None

    # Filter for intensification phase only
    intensification_start = sliced_track[sliced_track['period'] == 'intensification']['date'].min()
    intensification_end = sliced_track[sliced_track['period'] == 'intensification']['date'].max()
    track = sliced_track[(sliced_track['date'] >= intensification_start) & (sliced_track['date'] <= intensification_end)]
    if track.empty:
        print(f"No intensification phase data for {system_dir}")
        return None

    # Get ERA5 data for computing PV and EGR
    pressure_levels = ['250', '300', '350', '975', '1000']
    variables = ["u_component_of_wind", "v_component_of_wind", "temperature", "geopotential"]
    infile = f"{system_id}-pv-egr.nc"
    infile = get_cdsapi_era5_data(infile, track, pressure_levels, variables, lowest_ck_date) 
    
    return infile

def process_results(nc_file, lowest_ck_date):
        # Load the dataset
        ds = xr.open_dataset(nc_file)

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
        track_id = int(os.path.basename(nc_file).split('.')[0].split('-')[0])

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
        
        print(f"Finished creating PV composite for {nc_file}")

        return ds

def main():
    # Get track and periods data
    tracks_with_periods = pd.read_csv('../tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv')

    nc_files = glob(f'*-pv-egr.nc')

    for nc_file in nc_files:
        # Get system_id
        system_id = nc_file.split('-')[0]
        print(f"Processing {system_id}")

        # Process study case
        results_directory = glob(f'{LEC_RESULTS_DIR}/{system_id}*')[0]
        file_study_case = f'{OUTPUT_DIR}/{system_id}_results_study_case.nc'
        lowest_ck_date = get_lowest_ck_date(results_directory, system_id, tracks_with_periods)
        ds = process_results(nc_file, lowest_ck_date)
        
        # Save study case
        ds.to_netcdf(file_study_case)
        print(f"Saved {file_study_case}")
 
if __name__ == '__main__':
    main()