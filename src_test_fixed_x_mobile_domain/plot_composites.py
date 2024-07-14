# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_composites.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 19:56:13 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/08 00:30:59 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script will plot maps with the derivative of PV in y, as to display if barotropic instability is occuring.
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
import cmocean.cm as cmo
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

TITLE_SIZE = 16
TICK_LABEL_SIZE = 12
LABEL_SIZE = 12
GRID_LABEL_SIZE = 12
FIGURES_DIR = '../figures_test_fixed_framework/composites/'
CRS = ccrs.PlateCarree()
COMPOSITE_DIR = '../results_nc_files/composites_test_fixed_x_mobile/'


def plot_map(data, u, v, hgt, **kwargs):
    """Plot potential vorticity using dynamic normalization based on data values."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=CRS)

    transform = ccrs.PlateCarree()
    cmap, levels, title, units = kwargs.get('cmap'), kwargs.get('levels'), kwargs.get('title'), kwargs.get('units')

    # Create the contour plot
    levels_min, levels_max = np.min(levels), np.max(levels)
    if levels_min < 0 and levels_max > 0:
        norm = colors.TwoSlopeNorm(vmin=np.min(levels), vcenter=0, vmax=np.max(levels))
    else:
        norm = colors.Normalize(vmin=np.min(levels), vmax=np.max(levels))
    cf = ax.contourf(data.longitude, data.latitude, data, cmap=cmap, norm=norm, transform=transform, levels=levels, extend='both')
    
    # Add a colorbar
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

    # Setting gridlines and labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False   # Disable labels at the top
    gl.right_labels = False # Disable labels on the right
    gl.xlabel_style = {'size': GRID_LABEL_SIZE, 'color': 'gray', 'weight': 'bold'}  # Style for x-axis labels
    gl.ylabel_style = {'size': GRID_LABEL_SIZE, 'color': 'gray', 'weight': 'bold'}  # Style for y-axis labels
    
    # Optionally set the x and y locators to control the locations of the grid lines
    gl.xlocator = ticker.MaxNLocator(nbins=5)  # Adjust the number of bins as needed
    gl.ylocator = ticker.MaxNLocator(nbins=5)

    # Add coastlines
    ax.coastlines()

    ax.set_title(title, fontsize=TITLE_SIZE)  # You can adjust the fontsize as necessary

    # Save the figure
    filename = kwargs.get('filename')
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

def plot_lon_mean(lon_mean, **plot_attrs):
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.axvline(0, color='#c1121f', linestyle='--', linewidth=0.5)
    ax.plot(lon_mean, lon_mean.latitude,
                color='#003049', linewidth=3)
    ax.set_title(plot_attrs['title'], fontsize=TITLE_SIZE)
    ax.set_ylabel('Latitude', fontsize=LABEL_SIZE)
    ax.set_xlabel(plot_attrs['units'], fontsize=LABEL_SIZE)
    plt.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    filename = plot_attrs['filename']
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

def determine_norm_bounds(data, factor=1.0):
    """Determines symmetric normalization bounds for plotting centered around zero."""
    data_min, data_max = data.min().values, data.max().values
    max_abs_value = max(abs(data_min), abs(data_max)) * factor
    return -max_abs_value, max_abs_value

def main():

    filepath = f'{COMPOSITE_DIR}/pv_egr_mean_composite.nc'
    os.makedirs(FIGURES_DIR, exist_ok=True)

    ds = xr.open_dataset(filepath)
    ds['pv_baroclinic'] = ds['pv_baroclinic'] * 1e6
    pv_baroclinic = ds['pv_baroclinic']
    absolute_vorticity = ds['absolute_vorticity']
    egr = ds['EGR']
    u = ds['u']
    v = ds['v']
    hgt = ds['hgt']

    # Calculate derivatives
    pv_baroclinic_derivative = pv_baroclinic.diff('latitude')
    absolute_vorticity_derivative = absolute_vorticity.diff('latitude')

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
        
    contour_levels['pv_baroclinic_derivative'] = {}
    contour_levels['absolute_vorticity_derivative'] = {}
    for level in ds.level:
        level_str = str(int(level))
        pvd_level = pv_baroclinic_derivative.sel(level=level)
        avd_level = absolute_vorticity_derivative.sel(level=level)
        contour_levels['pv_baroclinic_derivative'][level_str] = np.linspace(
            pvd_level.min(skipna=True), pvd_level.max(skipna=True), 11)
        contour_levels['absolute_vorticity_derivative'][level_str] = np.linspace(
            avd_level.min(skipna=True), avd_level.max(skipna=True), 11)
    for level in pv_baroclinic.level.values:
        # Convert level to string
        level_str = str(int(level))

        # Select data for the current level
        pv_baroclinic_level = pv_baroclinic.sel(level=level)
        absolute_vorticity_level = absolute_vorticity.sel(level=level)
        egr_level = egr.sel(level=level)

        pv_baroclinic_derivative_level = pv_baroclinic_derivative.sel(level=level)
        absolute_vorticity_derivative_level = absolute_vorticity_derivative.sel(level=level)

        pv_baroclinic_derivative_lon_mean_level = pv_baroclinic_derivative_level.sel(longitude=slice(-180, 180)).mean('longitude')
        absolute_vorticity_derivative_lon_mean_level = absolute_vorticity_derivative_level.sel(longitude=slice(-180, 180)).mean('longitude')

        u_level = u.sel(level=level)
        v_level = v.sel(level=level)
        hgt_level = hgt.sel(level=level)
        
        # Baroclinic PV
        map_attrs = {
            'cmap': 'Blues_r',
            'title': r'$PV$' + f' @ {level_str} hPa',
            'levels': contour_levels['pv_baroclinic'][level_str],
            'units': 'PVU',
            'filename': f'composite_fixed_pv_baroclinic_{level_str}.png',
        }
        plot_map(pv_baroclinic_level, u_level, v_level, hgt_level, **map_attrs)    

        # Baroclinic PV derivative
        map_attrs = {
            'cmap': 'RdBu_r',
            'title': r'$\frac{\partial PV}{\partial y}$' + f' @ {level_str} hPa',
            'levels': contour_levels['pv_baroclinic_derivative'][level_str],
            'units': 'PVU',
            'filename': f'composite_fixed_pv_baroclinic_derivative_{level_str}.png',
        }
        plot_map(pv_baroclinic_derivative_level, u_level[:-1], v_level[:-1], hgt_level[:-1], **map_attrs)

        # Baroclinic PV derivative lon mean
        plot_attrs = {
            'title': r'$\frac{\partial PV}{\partial y}$' + f' @ {level_str} hPa',
            'units': 'PVU',
            'filename': f'composite_fixed_pv_baroclinic_derivative_lon_mean_{level_str}.png',
        }
        plot_lon_mean(pv_baroclinic_derivative_lon_mean_level, **plot_attrs)

        # Absolute Vorticity 
        map_attrs = {
            'cmap': 'Blues_r',
            'title': r'$\eta$' + f' @ {level_str} hPa',
            'levels': contour_levels['absolute_vorticity'][level_str],
            'units': 's$^{-1}$',
            'filename': f'composite_fixed_absolute_vorticity_{level_str}.png',
        }
        plot_map(absolute_vorticity_level, u_level, v_level, hgt_level, **map_attrs)

        # Absolute Vorticity derivative
        map_attrs = {
            'cmap': cmo.curl,
            'title': r'$\frac{\partial \eta}{\partial y}$' + f' @ {level_str} hPa',
            'levels': contour_levels['absolute_vorticity_derivative'][level_str],
            'units': r'$s^{-1}$',
            'filename': f'composite_fixed_absolute_vorticity_derivative_{level_str}.png',
        }
        plot_map(absolute_vorticity_derivative_level, u_level[:-1], v_level[:-1], hgt_level[:-1], **map_attrs)

        # Absolute Vorticity derivative lon mean
        plot_attrs = {
            'title': r'$\frac{\partial \eta}{\partial y}$' + f' @ {level_str} hPa',
            'units': r'$s^{-1}$',
            'filename': f'composite_fixed_absolute_vorticity_derivative_lon_mean_{level_str}.png',
        }
        plot_lon_mean(absolute_vorticity_derivative_lon_mean_level, **plot_attrs)

        # EGR
        map_attrs = {
            'cmap': 'Spectral_r',
            'title': 'EGR ' + f' @ {level_str} hPa',
            'levels': contour_levels['EGR'][level_str],
            'units': 'd$^{-1}$',
            'filename': f'composite_fixed_egr_{level_str}.png',
        }
        plot_map(egr_level, u_level, v_level, hgt_level, **map_attrs)
    

if __name__ == '__main__':
    main()
