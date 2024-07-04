# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_study_cases.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 19:56:13 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/04 11:37:14 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script plots maps with the advection of temperature, Geopotential height contours, and wind vectors.
"""

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

def plot_map(ax, data, u, v, hgt, **kwargs):
    """
    Plot data with Geopotential height contours and wind vectors.
    """
    transform = ccrs.PlateCarree()
    cmap, levels, title, units = kwargs['cmap'], kwargs['levels'], kwargs['title'], kwargs['units']
    latitudes, longitudes = data.latitude, data.longitude

    # Create the contour plot for the data
    levels_min, levels_max = np.min(levels), np.max(levels)
    norm = colors.TwoSlopeNorm(vmin=levels_min, vcenter=0, vmax=levels_max) if levels_min < 0 and levels_max > 0 else colors.Normalize(vmin=levels_min, vmax=levels_max)
    cf = ax.contourf(longitudes, latitudes, data, cmap=cmap, norm=norm, levels=levels, transform=transform, extend='both')

    # Add colorbar
    try:
        colorbar = plt.colorbar(cf, ax=ax, pad=0.1, orientation='horizontal', shrink=0.5, label=units)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        colorbar.ax.xaxis.set_major_formatter(formatter)
        colorbar.update_ticks()
    except ValueError:
        pass

    # Add coastlines, country borders, and state borders
    ax.coastlines(linewidth=1, color='gray')
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='gray')
    ax.add_feature(cfeature.STATES, linestyle='-', linewidth=1, edgecolor='gray')

    # Add Geopotential height contours
    ax.contour(longitudes, latitudes, hgt, colors='k', linestyles='-', linewidths=2, transform=transform)

    # Plot wind vectors
    wsp = np.sqrt(u**2 + v**2)
    wsp_mean, wsp_max = int(np.mean(wsp)), round(int(np.max(wsp)), -1)
    skip_n = 8
    scale_factor = wsp_mean * 60 if wsp_max <= 30 else wsp_mean * 30
    label = wsp_max if wsp_max <= 30 else wsp_max + 10
    qu = ax.quiver(longitudes[::skip_n], latitudes[::skip_n], u[::skip_n, ::skip_n], v[::skip_n, ::skip_n], transform=transform, zorder=1,width=0.005, headwidth=2, headlength=2, headaxislength=2, scale=scale_factor)
    ax.quiverkey(qu, X=0.5, Y=-0.1, U=label, label=f'{label} m/s', labelpos='E', coordinates='axes')

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

def plot_variable(data, u, v, hgt, track_id, output_dir, **map_attrs):
    """
    Plot a single variable and save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': CRS})
    plot_map(ax, data, u, v, hgt, **map_attrs)
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

        # Get variables
        T = id_data['t']  # Assuming T (temperature) is a variable in the dataset
        u, v, hgt = id_data['u'], id_data['v'], id_data['hgt']
        temp_advection = id_data['temp_advection'] * units('K/s')
        temp_advection = temp_advection.metpy.convert_units('K/day')

        # Compute u * T^2 and v * T^2
        u_T2 = u * (T ** 2)
        v_T2 = v * (T ** 2)

        # Plot temperature advection, u * T^2, and v * T^2 at each level
        for level in ds.level:
            # Calculate the maximum absolute value for symmetric levels
            min_val = temp_advection.sel(level=level).min(skipna=True).metpy.dequantify().item()
            max_val = temp_advection.sel(level=level).max(skipna=True).metpy.dequantify().item()
            abs_max = max(abs(min_val), abs(max_val))

            min_val_u_T2 = u_T2.sel(level=level).min(skipna=True).metpy.dequantify().item()
            max_val_u_T2 = u_T2.sel(level=level).max(skipna=True).metpy.dequantify().item()
            abs_max_u_T2 = max(abs(min_val_u_T2), abs(max_val_u_T2))

            min_val_v_T2 = v_T2.sel(level=level).min(skipna=True).metpy.dequantify().item()
            max_val_v_T2 = v_T2.sel(level=level).max(skipna=True).metpy.dequantify().item()
            abs_max_v_T2 = max(abs(min_val_v_T2), abs(max_val_v_T2))

            # Create symmetric contour levels
            contour_levels = np.linspace(-abs_max, abs_max, 12)
            contour_levels_u_T2 = np.linspace(-abs_max_u_T2, abs_max_u_T2, 12)
            contour_levels_v_T2 = np.linspace(-abs_max_v_T2, abs_max_v_T2, 12)

            level_str = str(int(level))

            # Plot temperature advection
            map_attrs_advection = {
                'cmap': 'RdBu_r',
                'title': f'Temperature Advection @ {level_str} hPa - {date}',
                'levels': contour_levels,
                'units': 'K/day',
                'filename': f'composite_temp_advection_{level_str}hpa.png'
            }
            plot_variable(temp_advection.sel(level=level), u.sel(level=level), v.sel(level=level), hgt.sel(level=level), track_id.values, output_dir, **map_attrs_advection)

            # Plot u * T^2
            map_attrs_u_T2 = {
                'cmap': 'RdBu_r',
                'title': f'u * T^2 @ {level_str} hPa - {date}',
                'levels': contour_levels_u_T2,
                'units': 'm^2/s^2',
                'filename': f'composite_u_T2_{level_str}hpa.png'
            }
            plot_variable(u_T2.sel(level=level), u.sel(level=level), v.sel(level=level), hgt.sel(level=level), track_id.values, output_dir, **map_attrs_u_T2)

            # Plot v * T^2
            map_attrs_v_T2 = {
                'cmap': 'RdBu_r',
                'title': f'v * T^2 @ {level_str} hPa - {date}',
                'levels': contour_levels_v_T2,
                'units': 'm^2/s^2',
                'filename': f'composite_v_T2_{level_str}hpa.png'
            }
            plot_variable(v_T2.sel(level=level), u.sel(level=level), v.sel(level=level), hgt.sel(level=level), track_id.values, output_dir, **map_attrs_v_T2)

def main():
    """
    Main function to execute the plotting for all phases.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for phase in ['incipient', 'mature']:
        plot_study_cases(phase)

if __name__ == '__main__':
    main()
