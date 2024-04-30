# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_barotropic_instability.py                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 19:56:13 by daniloceano       #+#    #+#              #
#    Updated: 2024/04/30 20:34:56 by daniloceano      ###   ########.fr        #
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
FIGURES_DIR = '../figures_barotropic_instability'

def load_and_prepare_data(filepath):
    """Load dataset and prepare data for plotting."""
    ds = xr.open_dataset(filepath)
    pv_baroclinic = ds.pv_baroclinic
    pv_barotropic = ds.pv_barotropic
    pv_baroclinic_derivative = pv_baroclinic.diff('y')
    pv_barotropic_derivative = pv_barotropic.diff('y')
    return pv_baroclinic, pv_barotropic, pv_baroclinic_derivative, pv_barotropic_derivative

def configure_plot():
    """Set up the plot configuration and return axes."""
    crs_longlat = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16, 12))  # Adjust overall figure size to fit your needs
    # Setup GridSpec with different width ratios for each column
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.5])  # Reduce the width of the third column

    axes = []
    for i in range(6):
        if i % 3 != 2:  # Apply geographic projection to the first two columns
            axes.append(fig.add_subplot(gs[i], projection=crs_longlat))
        else:  # The third column in each row without projection and narrower
            axes.append(fig.add_subplot(gs[i]))

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
    
    # Set up plot
    fig, axes = configure_plot()
    
    # First row for Baroclinic PV and its derivative
    plot_map(axes[0], pv_baroclinic, cmo.balance, r'$PV_{BC}$')
    plot_map(axes[1], pv_baroclinic_derivative, cmo.curl, r'$\frac{\partial PV_{BC}}{\partial y}$')
    axes[2].plot(pv_baroclinic_derivative.mean('x'), np.arange(len(pv_baroclinic_derivative.mean('x'))),
                 color='#003049', linewidth=3)
    axes[2].axvline(0, color='#c1121f', linestyle='--', linewidth=0.5)
    axes[2].set_title(r'$\frac{\partial PV_{BC}}{\partial y}$' + ' lon mean', fontsize=TITLE_SIZE)
    axes[2].set_yticks([])
    axes[2].tick_params(axis='x', labelsize=TICK_LABEL_SIZE)

    # Second row for Barotropic PV and its derivative
    plot_map(axes[3], pv_barotropic, cmo.balance, r'$PV_{BT}$')
    plot_map(axes[4], pv_barotropic_derivative, cmo.curl, r'$\frac{\partial PV_{BT}}{\partial y}$')
    axes[5].plot(pv_barotropic_derivative.mean('x'), np.arange(len(pv_barotropic_derivative.mean('x'))),
                 color='#003049', linewidth=3)
    axes[5].axvline(0, color='#c1121f', linestyle='--', linewidth=0.5)
    axes[5].set_title(r'$\frac{\partial PV_{BT}}{\partial y}$' + ' lon mean', fontsize=TITLE_SIZE)
    axes[5].set_yticks([])
    axes[5].tick_params(axis='x', labelsize=TICK_LABEL_SIZE)

    plt.tight_layout()

    filename = 'pv_composite.png'
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

if __name__ == '__main__':
    main()
