# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_composites.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 19:56:13 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/02 09:14:00 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script will plot maps with the advection of temperature.
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import metpy.calc as mpcalc
from metpy.units import units

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
GRID_LABEL_SIZE = 10
FIGURES_DIR = '../figures_bae_fluxes'
CRS = ccrs.PlateCarree()

filepath = '../results_nc_files/composites_fluxes_downstream/bae_composite_mean.nc'

def plot_map(ax, data, **kwargs):
    """Plot temperature advection using dynamic normalization based on data values."""
    transform = ccrs.PlateCarree()
    cmap, levels, title, units = kwargs.get('cmap'), kwargs.get('levels'), kwargs.get('title'), kwargs.get('units')

    # Create the contour plot
    levels_min, levels_max = np.min(levels), np.max(levels)
    if levels_min < 0 and levels_max > 0:
        norm = colors.TwoSlopeNorm(vmin=np.min(levels), vcenter=0, vmax=np.max(levels))
    else:
        norm = colors.Normalize(vmin=np.min(levels), vmax=np.max(levels))
    cf = ax.contourf(data.x, data.y, data, cmap=cmap, norm=norm, transform=transform, levels=levels, extend='both')
    
    try:
        colorbar = plt.colorbar(cf, ax=ax, pad=0.1, orientation='horizontal', shrink=0.5, label=units)
        # Setup the colorbar to use scientific notation conditionally
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))  # Adjust these limits based on your specific needs
        colorbar.ax.xaxis.set_major_formatter(formatter)

        # Calculate ticks: Skip every 2 ticks
        current_ticks = colorbar.get_ticks()
        new_ticks = current_ticks[::2]  # Take every second tick
        colorbar.set_ticks(new_ticks)   # Set the modified ticks
        colorbar.update_ticks()

    except ValueError:
        pass

    # Set up grid lines
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.5, color='k')

    # Customize the ticks on x and y axes
    ax.xaxis.set_major_locator(ticker.AutoLocator())  # Automatically determine the location of ticks
    ax.yaxis.set_major_locator(ticker.AutoLocator())

    # Label formatting to show just numbers
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))

    # Set specific tick values if needed
    ax.set_xticks(np.arange(-30, 30, 5))
    ax.set_yticks(np.arange(-30, 30, 5))

    # Adjusting font size for axis tick labels
    ax.tick_params(axis='both', which='major', labelsize=GRID_LABEL_SIZE)

    ax.set_title(title, fontsize=TITLE_SIZE)  # You can adjust the fontsize as necessary

def determine_norm_bounds(data, factor=1.0):
    """Determines symmetric normalization bounds for plotting centered around zero."""
    data_min, data_max = data.min().values, data.max().values
    max_abs_value = max(abs(data_min), abs(data_max)) * factor
    return -max_abs_value, max_abs_value

def plot_variable(temp_advection, **map_attrs):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, temp_advection, **map_attrs)
    plt.tight_layout()
    filename = map_attrs['filename']
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')
    
def main():
    # Composites file

    # Open variables
    ds = xr.open_dataset(filepath)
    temperature = ds['t'] * units.kelvin
    u = ds['u'] * units('m/s')
    v = ds['v'] * units('m/s')

    temp_advection = mpcalc.advection(temperature, u, v)
    temp_advection.name = 'temp_advection'

    contour_levels = {}
    # Create levels for plot each variable
    for var in ds.data_vars:
        contour_levels[var] = {}
        for level in ds.level:
            level_str = str(int(level))
            min_val = float(min(
                ds[var].sel(level=level).min(skipna=True),
                ds[var].sel(level=level).min(skipna=True)))
            max_val = float(max(
                ds[var].sel(level=level).max(skipna=True),
                ds[var].sel(level=level).max(skipna=True)))
            contour_levels[var][str(level_str)] = np.linspace(min_val, max_val, 11)
        
    contour_levels['temp_advection'] = {}
    for level in ds.level:
        level_str = str(int(level))
        temp_adv_level = temp_advection.sel(level=level)
        contour_levels['temp_advection'][level_str] = np.linspace(
            temp_adv_level.min(skipna=True), temp_adv_level.max(skipna=True), 11)
    
    for level in ds.level:

        temp_advection_level = temp_advection.sel(level=level)

        level_str = str(int(level))

        # Plot Temperature Advection
        map_attrs = {
                'cmap': 'RdBu_r',
                'title': r'Temperature Advection @ ' + f'{level_str} hPa',
                'levels': contour_levels['temp_advection'][level_str],
                'units': 'K/s',
                'filename': f'composite_temp_advection_{level_str}hpa.png'
            }
        plot_variable(temp_advection_level, **map_attrs)

if __name__ == '__main__':
    main()
