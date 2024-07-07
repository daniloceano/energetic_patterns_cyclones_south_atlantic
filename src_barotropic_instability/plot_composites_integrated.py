# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_composites_integrated.py                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 19:56:13 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/06 16:29:40 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script will plot maps with the integrated derivative of PV in y, as to display if barotropic instability is occurring.
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

def plot_map(ax, data, **kwargs):
    """Plot potential vorticity using dynamic normalization based on data values."""
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

def plot_variable(pv_baroclinic, **map_attrs):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, pv_baroclinic, **map_attrs)
    plt.tight_layout()
    filename = map_attrs['filename']
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

def plot_derivative(pv_baroclinic_derivative, **map_attrs):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, pv_baroclinic_derivative, **map_attrs)
    plt.tight_layout()
    filename = map_attrs['filename']
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

def plot_lon_mean(lon_mean, **map_attrs):
    fig = plt.figure(figsize=(4, 5))
    ax = plt.gca()
    ax.axvline(0, color='#c1121f', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='#c1121f', linestyle='--', linewidth=0.5)
    ax.plot(lon_mean, lon_mean.y, color='#003049', linewidth=3)
    plt.xlabel(map_attrs['units'], fontsize=LABEL_SIZE)
    ax.set_title(map_attrs['title'], fontsize=TITLE_SIZE)
    plt.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    filename = map_attrs['filename']
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

def main():
    # Composites file
    filepath = '../results_nc_files/composites_barotropic_baroclinic/pv_egr_composite_mean.nc'

    # Open variables
    ds = xr.open_dataset(filepath)
    ds['pv_baroclinic'] = ds['pv_baroclinic'] * 1e6
    pv_baroclinic = ds['pv_baroclinic']
    absolute_vorticity = ds['absolute_vorticity']
    egr = ds['EGR']
    u = ds['u']
    v = ds['v']
    hgt = ds['hgt']

    # Calculate derivatives
    pv_baroclinic_derivative = pv_baroclinic.diff('y')
    absolute_vorticity_derivative = absolute_vorticity.diff('y')
    pv_baroclinic_derivative.name = 'pv_baroclinic_derivative'
    absolute_vorticity_derivative.name = 'absolute_vorticity_derivative'

    # Integrate variables in the vertical
    pv_baroclinic_integrated = pv_baroclinic.integrate('level')
    pv_baroclinic_derivative_integrated = pv_baroclinic_derivative.integrate('level')
    absolute_vorticity_integrated = absolute_vorticity.integrate('level')
    absolute_vorticity_derivative_integrated = absolute_vorticity_derivative.integrate('level')
    egr_integrated = egr.dropna(dim='level').integrate('level')

    contour_levels = {}
    # Create levels for plot each variable
    contour_levels['pv_baroclinic'] = np.linspace(
        pv_baroclinic_integrated.min(skipna=True), pv_baroclinic_integrated.max(skipna=True), 11)
    contour_levels['absolute_vorticity'] = np.linspace(
        absolute_vorticity_integrated.min(skipna=True), absolute_vorticity_integrated.max(skipna=True), 11)
    contour_levels['EGR'] = np.linspace(
        egr_integrated.min(skipna=True), egr_integrated.max(skipna=True), 11)
    contour_levels['pv_baroclinic_derivative_integrated'] = np.linspace(
        pv_baroclinic_derivative_integrated.min(skipna=True), pv_baroclinic_derivative_integrated.max(skipna=True), 11)
    contour_levels['absolute_vorticity_derivative_integrated'] = np.linspace(
        absolute_vorticity_derivative_integrated.min(skipna=True), absolute_vorticity_derivative_integrated.max(skipna=True), 11)

    # Plot Baroclinic PV integrated
    map_attrs = {
            'cmap': 'Blues_r',
            'title': r'$PV$' + ' Integrated',
            'levels': contour_levels['pv_baroclinic'],
            'units': 'PVU',
            'filename': 'composite_semi-lagrangian_pv_baroclinic_integrated.png'
        }
    plot_variable(pv_baroclinic_integrated, **map_attrs)

    # Baroclinic PV derivative integrated
    map_attrs = {
            'cmap': cmo.curl,
            'title': r'$\frac{\partial PV}{\partial y}$' + ' Integrated',
            'levels': contour_levels['pv_baroclinic_derivative_integrated'],
            'units': 'PVU',
            'filename': 'composite_semi-lagrangian_pv_baroclinic_derivative_integrated.png'
        }
    plot_derivative(pv_baroclinic_derivative_integrated, **map_attrs)

    # Baroclinic PV derivative lon mean integrated
    map_attrs = {
            'title': r'$\frac{\partial PV}{\partial y}$' + ' Integrated',
            'units': 'PVU',
            'filename': 'composite_semi-lagrangian_pv_baroclinic_derivative_lon_mean_integrated.png'
        }
    plot_lon_mean(pv_baroclinic_derivative_integrated.mean('x'), **map_attrs)

    # Absolute Vorticity integrated
    map_attrs = {
        'cmap': 'Blues_r',
        'title': r'$\eta$' + ' Integrated',
        'levels': contour_levels['absolute_vorticity'],
        'units': r'$s^{-1}$',
        'filename': 'composite_semi-lagrangian_absolute_vorticity_integrated.png'
    }
    plot_variable(absolute_vorticity_integrated, **map_attrs)

    # Absolute Vorticity derivative integrated
    map_attrs = {
        'cmap': cmo.curl,
        'title': r'$\frac{\partial \eta}{\partial y}$' + ' Integrated',
        'levels': contour_levels['absolute_vorticity_derivative_integrated'],
        'units': r'$s^{-1}$',
        'filename': 'composite_semi-lagrangian_absolute_vorticity_derivative_integrated.png'
    }
    plot_derivative(absolute_vorticity_derivative_integrated, **map_attrs)

    # Absolute Vorticity derivative lon mean integrated
    map_attrs = {
            'title': r'$\frac{\partial \eta}{\partial y}$' + ' Integrated',
            'units': r's$^{-1}$',
            'filename': 'composite_semi-lagrangian_absolute_vorticity_derivative_lon_mean_integrated.png'
        }
    plot_lon_mean(absolute_vorticity_derivative_integrated.mean('x'), **map_attrs)

    # EGR integrated
    map_attrs = {
        'cmap': 'Spectral_r',
        'title': 'EGR Integrated',
        'levels': contour_levels['EGR'],
        'units': r'$d^{-1}$',
        'filename': 'composite_semi-lagrangian_EGR_integrated.png'
    }
    plot_variable(egr_integrated, **map_attrs)
        

if __name__ == '__main__':
    main()
