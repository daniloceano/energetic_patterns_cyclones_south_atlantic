# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_barotropic_instability.py                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 19:56:13 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/10 20:43:55 by daniloceano      ###   ########.fr        #
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
TICK_LABEL_SIZE = 12
FIGURES_DIR = '../figures_barotropic_baroclinic_instability'
CRS = ccrs.PlateCarree()

def plot_map(ax, data, cmap, title, levels, transform=ccrs.PlateCarree()):
    """Plot potential vorticity using dynamic normalization based on data values."""
    levels_min, levels_max = np.min(levels), np.max(levels)
    if levels_min < 0 and levels_max > 0:
        norm = colors.TwoSlopeNorm(vmin=np.min(levels), vcenter=0, vmax=np.max(levels))
    else:
        norm = colors.Normalize(vmin=np.min(levels), vmax=np.max(levels))
    cf = ax.contourf(data.x, data.y, data, cmap=cmap, norm=norm, transform=transform, levels=levels, extend='both')
    colorbar = plt.colorbar(cf, ax=ax, pad=0.1, orientation='horizontal', shrink=0.5)
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
    ax.set_title(title, fontsize=12)  # You can adjust the fontsize as necessary

def determine_norm_bounds(data, factor=1.0):
    """Determines symmetric normalization bounds for plotting centered around zero."""
    data_min, data_max = data.min().values, data.max().values
    max_abs_value = max(abs(data_min), abs(data_max)) * factor
    return -max_abs_value, max_abs_value

def main(filepath='../results_nc_files/composites_barotropic_baroclinic/pv_egr_composite_mean.nc'):

    ds = xr.open_dataset(filepath)
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
    plot_map(ax, pv_baroclinic, 'Blues_r', r'$PV$', levels['pv_baroclinic'])
    plt.tight_layout()
    filename = 'pv_baroclinic_composite.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Barotropic PV derivative
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, pv_baroclinic_derivative, cmo.curl, r'$\frac{\partial PV}{\partial y}$', levels['pv_baroclinic_derivative'])
    plt.tight_layout()
    filename = 'pv_baroclinic_composite_derivative.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Baroclinic PV derivative lon mean
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.plot(pv_baroclinic_derivative.mean('x'), np.arange(len(pv_baroclinic_derivative.mean('x'))),
                 color='#003049', linewidth=3)
    ax.axvline(0, color='#c1121f', linestyle='--', linewidth=0.5)
    ax.set_title(r'$\frac{\partial PV}{\partial y}$' + ' lon mean', fontsize=TITLE_SIZE)
    ax.set_yticks([])
    plt.tick_params(axis='x', labelsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    filename = 'pv_baroclinic_composite_derivative_lon_mean.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Absolute Vorticity 
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, absolute_vorticity, 'Blues_r', r'$\eta$', levels['absolute_vorticity'])
    plt.tight_layout()
    filename = 'absolute_vorticity_composite.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Absolute Vorticity derivative
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, absolute_vorticity_derivative, cmo.curl, r'$\frac{\partial \eta}{\partial y}$', levels['absolute_vorticity_derivative'])
    plt.tight_layout()
    filename = 'absolute_vorticity_composite_derivative.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Absolute Vorticity derivative lon mean
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.plot(absolute_vorticity_derivative.mean('x'), np.arange(len(absolute_vorticity_derivative.mean('x'))),
                 color='#003049', linewidth=3)
    ax.axvline(0, color='#c1121f', linestyle='--', linewidth=0.5)
    ax.set_title(r'$\frac{\partial \eta}{\partial y}$' + ' lon mean', fontsize=TITLE_SIZE)
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    filename = 'absolute_vorticity_composite_derivative_lon_mean.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(file_path)
    print(f'Saved {filename}')

    # Absolute Vorticity 
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, egr, 'rainbow', 'EGR', levels['EGR'])
    plt.tight_layout()
    filename = 'EGR_composite.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')
    

if __name__ == '__main__':
    main()
