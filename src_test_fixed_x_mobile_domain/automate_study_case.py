# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    automate_study_case.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/08 14:17:01 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/08 15:26:28 by daniloceano      ###   ########.fr        #
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


LEC_RESULTS_DIR = '../../LEC_Results_fixed_framework_test'

def choose_study_case(LEC_RESULTS_DIR, tracks_with_periods):
    """
    Determine which case had the lowest Ck values for plotting.
    """
    min_ck_values = {}
    
    for directory in os.listdir(LEC_RESULTS_DIR):
        results_file = glob(f"{LEC_RESULTS_DIR}/{directory}/*fixed_results.csv")[0]
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
    infile = f"{filename}.nc"

    if not os.path.exists(infile):
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
            }, infile # save file as passed in arguments
        )

        if not os.path.exists(infile):
            raise FileNotFoundError("CDS API file not created.")
        return infile
    
    else:
        print("CDS API file already exists.")
        return infile
    
def calculate_eady_growth_rate(u, theta, f):
    # Calculate the derivative of U with respect to log-pressure
    dudp = u.differentiate("level")
    
    # Calculate the derivative of theta with respect to log-pressure
    dthetadp = theta.differentiate("level")
    
    # Calculate Brunt-Väisälä Frequency (N)
    N_sqrd = (9.81 / theta) * (-dthetadp)
    N_sqrd = N_sqrd.where(N_sqrd > 0, 0)
    N = np.sqrt(N_sqrd)
    
    # Calculate Eady Growth Rate
    EGR = 0.31 * (np.abs(f) / N) * np.abs(dudp)

    return EGR
    
def process_system(system_dir, tracks_with_periods, lowest_ck_date):
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
    pressure_levels = ['200', '250', '300', '350', '400', '450']
    variables = ["u_component_of_wind", "v_component_of_wind", "temperature"]
    file_study_case = get_cdsapi_era5_data(f'{system_id}-study_case', track, pressure_levels, variables, lowest_ck_date) 
    
    return file_study_case

def process_results(file_study_case, lowest_ck_date):
        # Load the dataset
        ds = xr.open_dataset(file_study_case)

        # Open variables for calculations and assign units
        temperature = ds['t'] * units.kelvin
        pressure = ds.level * units.hPa
        u = ds['u'] * units('m/s')
        v = ds['v'] * units('m/s')
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
        eady_growth_rate = calculate_eady_growth_rate(u, potential_temperature, f)
        print("Done.")

        # Select the 250 hPa level
        pv_baroclinic_250 = pv_baroclinic.sel(time=lowest_ck_date).sel(level=250)
        absolute_vorticity_250 = absolute_vorticity.sel(time=lowest_ck_date).sel(level=250)
        eady_growth_rate_400 = eady_growth_rate.sel(time=lowest_ck_date).sel(level=400)

        # Create a DataArray using an extra dimension for the type of PV
        print("Creating DataArray...")
        track_id = int(file_study_case.split('.')[0].split('-')[0])

        # Create DataArrays
        da_baroclinic = xr.DataArray(
            pv_baroclinic_250.values,
            dims=['latitude', 'longitude'],
            coords={'latitude': lat, 'longitude': lon},
            name='pv_baroclinic'
        )

        da_absolute_vorticity = xr.DataArray(
            absolute_vorticity_250.values,
            dims=['latitude', 'longitude'],
            coords={'latitude': lat, 'longitude': lon},
            name='absolute_vorticity'
        )

        da_edy = xr.DataArray(
            eady_growth_rate_400.values,
            dims=['latitude', 'longitude'],
            coords={'latitude': lat, 'longitude': lon},
            name='EGR'
        )

        # Combine into a Dataset and add track_id as a coordinate
        ds = xr.Dataset({
            'pv_baroclinic': da_baroclinic,
            'absolute_vorticity': da_absolute_vorticity,
            'EGR': da_edy
        })

        # Assigning track_id as a coordinate
        ds = ds.assign_coords(track_id=track_id)  # Assigning track_id as a coordinate

        # Assign date
        ds = ds.assign_coords(time=lowest_ck_date)
        
        print(f"Finished creating PV composite for {file_study_case}")

        return ds

def main():
    # Get track and periods data
    tracks_with_periods = pd.read_csv('../tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv')

    # Choose study case
    lowest_track, lowest_ck_date = choose_study_case(LEC_RESULTS_DIR, tracks_with_periods)
    system_id = lowest_track['track_id'].unique()[0]

    system_dir = os.path.join(LEC_RESULTS_DIR, f"{system_id}_ERA5_fixed")

    file_study_case = process_system(system_dir, tracks_with_periods, lowest_ck_date)

    ds = process_results(file_study_case, lowest_ck_date)

    filename = f'{system_id}_results_study_case.nc'
    ds.to_netcdf(filename)
    print(f"Saved {filename}")
 
if __name__ == '__main__':
    main()