# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_advh_temp_integrated.py                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/07/03 13:09:28 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/03 13:14:47 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import json
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from metpy.units import units
import pandas as pd
from datetime import datetime
import cartopy.feature as cfeature

# Configuration constants
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
GRID_LABEL_SIZE = 10
FIGURES_DIR = '../figures_bae_fluxes/study_cases'
CRS = ccrs.PlateCarree()
NC_PATH = '../results_nc_files/composites_bae/'
JSON_PATH = '../results_nc_files/composites_bae/'

def read_latlon_time_data(cyclone_id, phase):
    """
    Read latitude, longitude, and date data from a JSON file.
    """
    json_file = os.path.join(JSON_PATH, f'{cyclone_id}_latlon_{phase}.json')
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {json_file} not found")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from file {json_file}")
    
    latitudes = data['latitude']
    longitudes = data['longitude']
    date_str = data['date']

    # Process the date string to extract the date
    date_str = date_str.strip("[]'\"")
    # Remove the nanoseconds part
    date_str = date_str.split('.')[0]
    try:
        date = datetime.fromisoformat(date_str)
    except ValueError:
        raise ValueError(f"Invalid date format in file {json_file}")

    return latitudes, longitudes, date

def plot_map(ax, temp_advection_integrated, **kwargs):
    """
    Plot integrated temperature advection with Geopotential height contours and wind vectors.
    """
    transform = ccrs.PlateCarree()
    cmap, levels, title, units = kwargs['cmap'], kwargs['levels'], kwargs['title'], kwargs['units']
    latitudes, longitudes = temp_advection_integrated.latitude, temp_advection_integrated.longitude

    # Create the contour plot for temperature advection
    levels_min, levels_max = np.min(levels), np.max(levels)
    norm = colors.TwoSlopeNorm(vmin=levels_min, vcenter=0, vmax=levels_max) if levels_min < 0 and levels_max > 0 else colors.Normalize(vmin=levels_min, vmax=levels_max)
    cf = ax.contourf(longitudes, latitudes, temp_advection_integrated, cmap=cmap, norm=norm, levels=levels, transform=transform, extend='both')

    # Add colorbar
    try:
        colorbar = plt.colorbar(cf, ax=ax, pad=0.1, orientation='horizontal', shrink=0.5, label=units)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        colorbar.ax.xaxis.set_major_formatter(formatter)
        # colorbar.set_ticks(colorbar.get_ticks()[::2])
        colorbar.update_ticks()
    except ValueError:
        pass

    # Add coastlines, country borders, and state borders
    ax.coastlines(linewidth=1, color='gray')
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='gray')  # Change 'black' to your desired color
    ax.add_feature(cfeature.STATES, linestyle='-', linewidth=1, edgecolor='gray')  # Change 'blue' to your desired color

    # Customize grid and ticks
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.5, color='k')
    ax.set_xticks(np.arange(longitudes.min(), longitudes.max(), 5))
    ax.set_yticks(np.arange(latitudes.min(), latitudes.max(), 5))
    ax.tick_params(axis='both', which='major', labelsize=GRID_LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)

    # Draw the 15x15 square centered on the domain
    idx_lon_mean = int(len(longitudes) / 2)
    idx_lat_mean = int(len(latitudes) / 2)
    lon_center, lat_center = longitudes[idx_lon_mean], latitudes[idx_lat_mean]
    square_half_size = 7.5
    square_lon = [lon_center - square_half_size, lon_center + square_half_size, lon_center + square_half_size, lon_center - square_half_size, lon_center - square_half_size]
    square_lat = [lat_center - square_half_size, lat_center - square_half_size, lat_center + square_half_size, lat_center + square_half_size, lat_center - square_half_size]
    ax.plot(square_lon, square_lat, transform=transform, color='r', linewidth=2)

def plot_variable(temp_advection_integrated, track_id, output_dir, **map_attrs):
    """
    Plot a single variable and save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': CRS})
    plot_map(ax, temp_advection_integrated, **map_attrs)
    plt.tight_layout()
    track_dir = os.path.join(output_dir, str(track_id))
    os.makedirs(track_dir, exist_ok=True)
    plt.savefig(os.path.join(track_dir, map_attrs['filename']))
    plt.close(fig)
    print(f'Saved {map_attrs["filename"]} in {track_dir}')

def plot_study_cases(phase):
    """
    Plot study cases for a given phase.
    """
    ds = xr.open_dataset(os.path.join(NC_PATH, f'bae_composite_{phase}_track_ids.nc'))
    output_dir = os.path.join(FIGURES_DIR, phase)
    os.makedirs(output_dir, exist_ok=True)

    for track_id in ds.track_id:
        # Get the data for the current track
        id_data = ds.sel(track_id=track_id)

        # Read latitude and longitude data
        latitudes, longitudes, date = read_latlon_time_data(track_id.values, phase)
        date = pd.to_datetime(date).strftime('%Y-%m-%d %HZ')

        # Replace the values of x and y with latitude and longitude
        id_data = id_data.assign_coords({'x': longitudes, 'y': latitudes}).rename({'x': 'longitude', 'y': 'latitude'})

        # Integrate temperature advection over all vertical levels
        temp_advection = id_data['temp_advection'] * units('K/s')
        temp_advection_integrated = temp_advection.integrate('level')
        temp_advection_integrated = temp_advection_integrated.metpy.convert_units('K/day')

        # Calculate the maximum absolute value for symmetric levels
        min_val = temp_advection_integrated.min(skipna=True).metpy.dequantify().item()
        max_val = temp_advection_integrated.max(skipna=True).metpy.dequantify().item()
        abs_max = max(abs(min_val), abs(max_val))

        # Create symmetric contour levels
        contour_levels = np.linspace(-abs_max, abs_max, 12)

        map_attrs = {
            'cmap': 'RdBu_r',
            'title': f'Integrated Temperature Advection - {date}',
            'levels': contour_levels,
            'units': 'K/day',
            'filename': f'integrated_temp_advection.png'
        }
        plot_variable(temp_advection_integrated, track_id.values, output_dir, **map_attrs)

def main():
    """
    Main function to execute the plotting for all phases.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for phase in ['incipient', 'mature']:
        plot_study_cases(phase)

if __name__ == '__main__':
    main()
