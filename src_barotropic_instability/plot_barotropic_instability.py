# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_barotropic_instability.py                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 19:56:13 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/11 01:05:50 by daniloceano      ###   ########.fr        #
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
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
GRID_LABEL_SIZE = 10
FIGURES_DIR = '../figures_barotropic_baroclinic_instability'
CRS = ccrs.PlateCarree()

def plot_map(ax, data, cmap, title, levels, units, transform=ccrs.PlateCarree()):
    """Plot potential vorticity using dynamic normalization based on data values."""
    levels_min, levels_max = np.min(levels), np.max(levels)
    if levels_min < 0 and levels_max > 0:
        norm = colors.TwoSlopeNorm(vmin=np.min(levels), vcenter=0, vmax=np.max(levels))
    else:
        norm = colors.Normalize(vmin=np.min(levels), vmax=np.max(levels))
    cf = ax.contourf(data.x, data.y, data, cmap=cmap, norm=norm, transform=transform, levels=levels, extend='both')
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

    # Set up grid lines
    ax.grid(True, linestyle='--', alpha=0.5)

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

    colorbar.update_ticks()
    ax.set_title(title, fontsize=TITLE_SIZE)  # You can adjust the fontsize as necessary

def determine_norm_bounds(data, factor=1.0):
    """Determines symmetric normalization bounds for plotting centered around zero."""
    data_min, data_max = data.min().values, data.max().values
    max_abs_value = max(abs(data_min), abs(data_max)) * factor
    return -max_abs_value, max_abs_value

def main(filepath='../results_nc_files/composites_barotropic_baroclinic/pv_egr_composite_mean.nc'):

    ds = xr.open_dataset(filepath)
    ds['pv_baroclinic'] = ds['pv_baroclinic'] * 1e6
    pv_baroclinic = ds['pv_baroclinic']
    absolute_vorticity = ds['absolute_vorticity']
    egr = ds['EGR']

    # Calculate derivatives
    pv_baroclinic_derivative = pv_baroclinic.diff('y')
    absolute_vorticity_derivative = absolute_vorticity.diff('y')
    pv_baroclinic_derivative.name = 'pv_baroclinic_derivative'
    absolute_vorticity_derivative.name = 'absolute_vorticity_derivative'

    levels = {}
    # Create levels for plot each variable
    for var in ds.data_vars:
        min_val = float(min(ds[var].min(), ds[var].min()))
        max_val = float(max(ds[var].max(), ds[var].max()))
        levels[var] = np.linspace(min_val, max_val, 11)
    levels['pv_baroclinic_derivative'] = np.linspace(np.min(pv_baroclinic_derivative), np.max(pv_baroclinic_derivative), 11)
    levels['absolute_vorticity_derivative'] = np.linspace(np.min(absolute_vorticity_derivative), np.max(absolute_vorticity_derivative), 11)
    
    # Baroclinic PV
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, pv_baroclinic, 'Blues_r', r'$PV$' + ' @ 1000 hPa', levels['pv_baroclinic'], 'PVU')
    plt.tight_layout()
    filename = 'pv_baroclinic_composite.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Barotropic PV derivative
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, pv_baroclinic_derivative, cmo.curl, r'$\frac{\partial PV}{\partial y}$' + ' @ 1000 hPa', levels['pv_baroclinic_derivative'], 'PVU')
    plt.tight_layout()
    filename = 'pv_baroclinic_composite_derivative.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Baroclinic PV derivative lon mean
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.axvline(0, color='#c1121f', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='#c1121f', linestyle='--', linewidth=0.5)
    ax.plot(pv_baroclinic_derivative.mean('x'), pv_baroclinic_derivative.mean('x').y,
                 color='#003049', linewidth=3)
    plt.xlabel('PVU', fontsize=LABEL_SIZE)
    ax.set_title(r'$\frac{\partial PV}{\partial y}$' + ' @ 1000 hPa', fontsize=TITLE_SIZE)
    plt.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    filename = 'pv_baroclinic_composite_derivative_lon_mean.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Absolute Vorticity 
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, absolute_vorticity, 'Blues_r', r'$\eta$' + ' @ 250 hPa', levels['absolute_vorticity'], r's$^{-1}$')
    plt.tight_layout()
    filename = 'absolute_vorticity_composite.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Absolute Vorticity derivative
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, absolute_vorticity_derivative, cmo.curl, r'$\frac{\partial \eta}{\partial y}$' + ' @ 250 hPa', levels['absolute_vorticity_derivative'], r's$^{-1}$')
    plt.tight_layout()
    filename = 'absolute_vorticity_composite_derivative.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Absolute Vorticity derivative lon mean
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.axvline(0, color='#c1121f', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='#c1121f', linestyle='--', linewidth=0.5)
    ax.plot(absolute_vorticity_derivative.mean('x'), absolute_vorticity_derivative.mean('x').y,
                 color='#003049', linewidth=3)
    ax.set_title(r'$\frac{\partial \eta}{\partial y}$' + ' @ 250 hPa', fontsize=TITLE_SIZE)
    plt.xlabel(r's$^{-1}$', fontsize=LABEL_SIZE)
    plt.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    filename = 'absolute_vorticity_composite_derivative_lon_mean.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(file_path)
    print(f'Saved {filename}')

    # EGR
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, egr, 'Spectral_r', 'EGR @ 1000 hPa', levels['EGR'], r'd$^{-1}$')
    plt.tight_layout()
    filename = 'EGR_composite.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')
    

if __name__ == '__main__':
    main()
