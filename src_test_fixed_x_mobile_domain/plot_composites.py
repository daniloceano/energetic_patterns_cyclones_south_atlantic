# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_composites.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 19:56:13 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/11 15:06:31 by daniloceano      ###   ########.fr        #
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
FIGURES_DIR = '../figures_test_fixed_framework'
CRS = ccrs.PlateCarree()
COMPOSITE_DIR = '../results_nc_files/composites_test_fixed_x_mobile/'


def plot_map(ax, data, u, v, hgt, **kwargs):
    """Plot potential vorticity using dynamic normalization based on data values."""
    transform = ccrs.PlateCarree()
    cmap, levels, title, units = kwargs.get('cmap'), kwargs.get('levels'), kwargs.get('title'), kwargs.get('units')

    # Create the contour plot
    levels_min, levels_max = np.min(levels), np.max(levels)
    if levels_min < 0 and levels_max > 0:
        norm = colors.TwoSlopeNorm(vmin=np.min(levels), vcenter=0, vmax=np.max(levels))
    else:
        norm = colors.Normalize(vmin=np.min(levels), vmax=np.max(levels))
    cf = ax.contourf(data.longitude, data.latitude, data, cmap=cmap, norm=norm, transform=transform, levels=levels, extend='both')

    # Add hgt 
    ax.contour(data.longitude, data.latitude, hgt, colors='gray', linestyles='dashed', linewidths=2, transform=transform)

    # Add quiver
    min_u = np.min(u)
    scale_factor = 100 if min_u < 10 else 400  # Adjust these values to tune the arrow
    skip = (slice(None, None, 15), slice(None, None, 15))
    qu = ax.quiver(data.longitude[skip[0]], data.latitude[skip[0]], u[skip], v[skip], transform=transform, zorder=1,
              width=0.008, headwidth=2, headlength=2, headaxislength=2,  scale=scale_factor)
    
    # Quiver key
    label = 10 if min_u < 10 else 20
    ax.quiverkey(qu, X=0.9, Y=1.05, U=label, label=f'{label} m/s', labelpos='E', coordinates='axes')

    # Add a colorbar
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

    ax.set_title(title, fontsize=TITLE_SIZE)  # You can adjust the fontsize as necessary
def determine_norm_bounds(data, factor=1.0):
    """Determines symmetric normalization bounds for plotting centered around zero."""
    data_min, data_max = data.min().values, data.max().values
    max_abs_value = max(abs(data_min), abs(data_max)) * factor
    return -max_abs_value, max_abs_value

def main():

    filepath = f'{COMPOSITE_DIR}/pv_egr_mean_composite.nc'

    ds = xr.open_dataset(filepath)
    ds['pv_baroclinic'] = ds['pv_baroclinic'] * 1e6
    pv_baroclinic = ds['pv_baroclinic']
    absolute_vorticity = ds['absolute_vorticity']
    egr = ds['EGR']
    u_1000, v_1000, hgt_1000 = ds['u_1000'], ds['v_1000'], ds['hgt_1000']
    u_250, v_250, hgt_250 = ds['u_250'], ds['v_250'], ds['hgt_250']

    # Calculate derivatives
    pv_baroclinic_derivative = pv_baroclinic.diff('latitude')
    absolute_vorticity_derivative = absolute_vorticity.diff('latitude')

    levels = {}
    # Create levels for plot each variable
    for var in ds.data_vars:
        min_val = float(min(ds[var].min(), ds[var].min()))
        max_val = float(max(ds[var].max(), ds[var].max()))
        levels[var] = np.linspace(min_val, max_val, 11)
    levels['pv_baroclinic_derivative'] = np.linspace(np.min(pv_baroclinic_derivative), np.max(pv_baroclinic_derivative), 11)
    levels['absolute_vorticity_derivative'] = np.linspace(np.min(absolute_vorticity_derivative), np.max(absolute_vorticity_derivative), 11)
    
    # Baroclinic PV
    map_attrs = {
        'cmap': 'Blues_r',
        'title': r'$PV$' + ' @ 1000 hPa',
        'levels': levels['pv_baroclinic'],
        'units': 'PVU',
    }
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, pv_baroclinic, u_1000, v_1000, hgt_1000, **map_attrs)
    filename = 'pv_baroclinic_composite.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Baroclinic PV derivative
    map_attrs = {
        'cmap': 'RdBu_r',
        'title': r'$\frac{\partial PV}{\partial y}$' + ' @ 1000 hPa',
        'levels': levels['pv_baroclinic_derivative'],
        'units': 'PVU',
    }
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, pv_baroclinic_derivative, u_1000[:-1], v_1000[:-1], hgt_1000[:-1], **map_attrs)
    filename = 'pv_baroclinic_composite_derivative_fixed.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Baroclinic PV derivative lon mean
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.axvline(0, color='#c1121f', linestyle='--', linewidth=0.5)
    ax.plot(pv_baroclinic_derivative.mean('longitude'), pv_baroclinic_derivative.mean('longitude').latitude,
                 color='#003049', linewidth=3)
    ax.set_title(r'$\frac{\partial PV}{\partial y}$' + ' @ 1000 hPa', fontsize=TITLE_SIZE)
    ax.set_ylabel('Latitude', fontsize=LABEL_SIZE)
    ax.set_xlabel('PVU', fontsize=LABEL_SIZE)
    plt.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    filename = 'pv_baroclinic_composite_derivative_lon_mean_fixed.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Absolute Vorticity 
    map_attrs = {
        'cmap': 'Blues_r',
        'title': r'$\eta$' + ' @ 250 hPa',
        'levels': levels['absolute_vorticity'],
        'units': 's$^{-1}$',
    }
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, absolute_vorticity, u_250, v_250, hgt_250, **map_attrs)
    filename = 'absolute_vorticity_composite_fixed.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Absolute Vorticity derivative
    map_attrs = {
        'cmap': cmo.curl,
        'title': r'$\frac{\partial \eta}{\partial y}$' + ' @ 250 hPa',
        'levels': levels['absolute_vorticity_derivative'],
        'units': r'$s^{-1}$',
    }
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, absolute_vorticity_derivative, u_250[:-1], v_250[:-1], hgt_250[:-1], **map_attrs)
    filename = 'absolute_vorticity_composite_derivative_fixed.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Absolute Vorticity derivative lon mean
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.axvline(0, color='#c1121f', linestyle='--', linewidth=0.5)
    ax.plot(absolute_vorticity_derivative.mean('longitude'), absolute_vorticity_derivative.mean('longitude').latitude,
                 color='#003049', linewidth=3)
    ax.set_title(r'$\frac{\partial \eta}{\partial y}$' + ' @ 250 hPa', fontsize=TITLE_SIZE)
    ax.set_ylabel('Latitude', fontsize=LABEL_SIZE)
    ax.set_xlabel(r's$^{-1}$', fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    filename = 'absolute_vorticity_composite_derivative_lon_mean_fixed.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(file_path)
    print(f'Saved {filename}')

    # EGR 
    map_attrs = {
        'cmap': 'Spectral_r',
        'title': 'EGR @ 1000 hPa',
        'levels': levels['EGR'],
        'units': 'd$^{-1}$',
    }
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, egr, u_1000, v_1000, hgt_1000, **map_attrs)
    filename = 'EGR_composite_fixed.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')
    

if __name__ == '__main__':
    main()
