# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_barotropic_instability.py                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 19:56:13 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/02 18:28:31 by daniloceano      ###   ########.fr        #
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
FIGURES_DIR = '../figures_barotropic_baroclinic_instability'
CRS = ccrs.PlateCarree()

def load_and_prepare_data(filepath):
    """Load dataset and prepare data for plotting."""
    ds = xr.open_dataset(filepath)
    pv_baroclinic = ds.pv_baroclinic
    pv_barotropic = ds.pv_barotropic
    pv_baroclinic_derivative = pv_baroclinic.diff('y')
    pv_barotropic_derivative = pv_barotropic.diff('y')

    return pv_baroclinic, pv_barotropic, pv_baroclinic_derivative, pv_barotropic_derivative

    return fig, axes

def plot_map(ax, data, cmap, title, transform=ccrs.PlateCarree()):
    """Plot potential vorticity using dynamic normalization based on data values."""
    vmin, vmax = determine_norm_bounds(data)
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cf = ax.contourf(data.x, data.y, data, cmap=cmap, norm=norm, transform=transform)
    colorbar = plt.colorbar(cf, ax=ax, pad=0.1, orientation='horizontal', shrink=0.5)
    colorbar.ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    colorbar.ax.xaxis.get_major_formatter().set_scientific(True)
    colorbar.ax.xaxis.get_major_formatter().set_powerlimits((0, 0))
    colorbar.ax.set_xticklabels(colorbar.ax.get_xticklabels(), rotation=45, fontsize=TICK_LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)

def determine_norm_bounds(data, factor=1.0):
    """Determines symmetric normalization bounds for plotting centered around zero."""
    data_min, data_max = data.min().values, data.max().values
    max_abs_value = max(abs(data_min), abs(data_max)) * factor
    return -max_abs_value, max_abs_value

def main(filepath='pv_composite_mean.nc'):
    pv_baroclinic, pv_barotropic, pv_baroclinic_derivative, pv_barotropic_derivative = load_and_prepare_data(filepath)
    
    # Baroclinic PV
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, pv_baroclinic, cmo.balance, r'$PV$')
    filename = 'pv_baroclinic_composite.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Barotropic PV derivative
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, pv_baroclinic_derivative, cmo.curl, r'$\frac{\partial PV}{\partial y}$')
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
    filename = 'pv_baroclinic_composite_derivative_lon_mean.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Absolute Vorticity 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, pv_barotropic, cmo.balance, r'$\eta$')
    filename = 'absolute_vorticity_composite.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Absolute Vorticity derivative
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, pv_barotropic_derivative, cmo.curl, r'$\frac{\partial \eta}{\partial y}$')
    filename = 'absolute_vorticity_composite_derivative.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

    # Absolute Vorticity derivative lon mean
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.plot(pv_barotropic_derivative.mean('x'), np.arange(len(pv_barotropic_derivative.mean('x'))),
                 color='#003049', linewidth=3)
    ax.axvline(0, color='#c1121f', linestyle='--', linewidth=0.5)
    ax.set_title(r'$\frac{\partial \eta}{\partial y}$' + ' lon mean', fontsize=TITLE_SIZE)
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=TICK_LABEL_SIZE)
    filename = 'absolute_vorticity_composite_derivative_lon_mean.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(file_path)
    print(f'Saved {filename}')
    

if __name__ == '__main__':
    main()
