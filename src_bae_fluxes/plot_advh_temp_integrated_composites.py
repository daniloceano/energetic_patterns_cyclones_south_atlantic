# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_advh_temp_integrated_composites.py            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 19:56:13 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/03 13:28:41 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script will plot maps with the integrated advection of temperature, Geopotential height contours, and wind vectors.
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from metpy.units import units

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
GRID_LABEL_SIZE = 10
FIGURES_DIR = '../figures_bae_fluxes/composites'
CRS = ccrs.PlateCarree()

NC_PATH = '../results_nc_files/composites_bae/'

def plot_map(ax, temp_advection_integrated,**kwargs):
    """Plot integrated temperature advection with Geopotential height contours and wind vectors."""
    transform = ccrs.PlateCarree()
    cmap, levels, title, units = kwargs.get('cmap'), kwargs.get('levels'), kwargs.get('title'), kwargs.get('units')

    # Create the contour plot for temperature advection
    levels_min, levels_max = np.min(levels), np.max(levels)
    if levels_min < 0 and levels_max > 0:
        norm = colors.TwoSlopeNorm(vmin=np.min(levels), vcenter=0, vmax=np.max(levels))
    else:
        norm = colors.Normalize(vmin=np.min(levels), vmax=np.max(levels))
    cf = ax.contourf(temp_advection_integrated.x, temp_advection_integrated.y, temp_advection_integrated, cmap=cmap, norm=norm, transform=transform, levels=levels, extend='both')
    
    try:
        colorbar = plt.colorbar(cf, ax=ax, pad=0.1, orientation='horizontal', shrink=0.5, label=units)
        # Setup the colorbar to use scientific notation conditionally
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))  # Adjust these limits based on your specific needs
        colorbar.ax.xaxis.set_major_formatter(formatter)

    except ValueError:
        pass

    # Set up grid lines
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.5, color='k')

    # Draw the 15x15 square centered on the domain
    lon_center, lat_center = 0, 0
    square_half_size = 7.5 / 0.25
    square_lon = [lon_center - square_half_size, lon_center + square_half_size, lon_center + square_half_size, lon_center - square_half_size, lon_center - square_half_size]
    square_lat = [lat_center - square_half_size, lat_center - square_half_size, lat_center + square_half_size, lat_center + square_half_size, lat_center - square_half_size]
    ax.plot(square_lon, square_lat, transform=transform, color='r', linewidth=2)

    # Customize the ticks on x and y axes
    ax.xaxis.set_major_locator(ticker.AutoLocator())  # Automatically determine the location of ticks
    ax.yaxis.set_major_locator(ticker.AutoLocator())

    # Label formatting to show just numbers
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))

    # Set specific tick values if needed
    ax.set_xticks(np.arange(-50, 50, 5))
    ax.set_yticks(np.arange(-50, 50, 5))

    # Adjusting font size for axis tick labels
    ax.tick_params(axis='both', which='major', labelsize=GRID_LABEL_SIZE)

    ax.set_title(title, fontsize=TITLE_SIZE)  # You can adjust the fontsize as necessary

def determine_norm_bounds(data, factor=1.0):
    """Determines symmetric normalization bounds for plotting centered around zero."""
    data_min, data_max = data.min().values, data.max().values
    max_abs_value = max(abs(data_min), abs(data_max)) * factor
    return -max_abs_value, max_abs_value

def plot_variable(temp_advection_integrated, output_dir, **map_attrs):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, temp_advection_integrated, **map_attrs)
    plt.tight_layout()
    filename = map_attrs['filename']
    file_path = os.path.join(output_dir, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

def plot_composites(netcdf_dir, phase):

    # Open the netCDF file
    filepath = os.path.join(netcdf_dir, f'bae_composite_{phase}_mean.nc')
    ds = xr.open_dataset(filepath)

    # Set up output directory
    output_dir = os.path.join(FIGURES_DIR, phase)
    os.makedirs(output_dir, exist_ok=True)

    # Integrate temperature advection over all vertical levels
    temp_advection = ds['temp_advection'] * units('K/s')
    temp_advection_integrated = temp_advection.integrate('level')
    temp_advection_integrated = temp_advection_integrated.metpy.convert_units('K/day')
    
    # Calculate the maximum absolute value for symmetric levels
    min_val = temp_advection_integrated.min(skipna=True).metpy.dequantify().item()
    max_val = temp_advection_integrated.max(skipna=True).metpy.dequantify().item()
    abs_max = max(abs(min_val), abs(max_val))

    # Create symmetric contour levels
    contour_levels = np.linspace(-abs_max, abs_max, 12)

    # Plot integrated Temperature Advection
    map_attrs = {
            'cmap': 'RdBu_r',
            'title': f'Integrated Temperature Advection - {phase.capitalize()} Phase',
            'levels': contour_levels,
            'units': 'K/day',
            'filename': 'integrated_temp_advection.png'
        }
    plot_variable(temp_advection_integrated, output_dir, **map_attrs)
    
def main():
    # create output directory if it doesn't exist
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for phase in ['incipient', 'mature']:
        plot_composites(NC_PATH, phase)


if __name__ == '__main__':
    main()
