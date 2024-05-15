# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_composites.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 19:56:13 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/15 16:43:13 by daniloceano      ###   ########.fr        #
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
    cf = ax.contourf(data.x, data.y, data, cmap=cmap, norm=norm, transform=transform, levels=levels, extend='both')

    # # Add hgt 
    # ax.contour(data.x, data.y, hgt, colors='gray', linestyles='dashed', linewidths=2, transform=transform)

    # # Add quiver
    # min_u = np.min(u)
    # scale_factor = 100 if min_u < 10 else 400  # Adjust these values to tune the arrow
    # skip = (slice(None, None, 5), slice(None, None, 5))
    # qu = ax.quiver(data.x[skip[0]], data.y[skip[0]], u[skip], v[skip], transform=transform, zorder=1,
    #           width=0.008, headwidth=2, headlength=2, headaxislength=2,  scale=scale_factor)
    
    # # Quiver key
    # label = 10 if min_u < 10 else 20
    # ax.quiverkey(qu, X=0.9, Y=1.05, U=label, label=f'{label} m/s', labelpos='E', coordinates='axes')
    
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

    colorbar.update_ticks()
    ax.set_title(title, fontsize=TITLE_SIZE)  # You can adjust the fontsize as necessary

def determine_norm_bounds(data, factor=1.0):
    """Determines symmetric normalization bounds for plotting centered around zero."""
    data_min, data_max = data.min().values, data.max().values
    max_abs_value = max(abs(data_min), abs(data_max)) * factor
    return -max_abs_value, max_abs_value

def plot_variable(pv_baroclinic, u, v, hgt, **map_attrs):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, pv_baroclinic, u, v, hgt, **map_attrs)
    plt.tight_layout()
    filename = map_attrs['filename']
    file_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

def plot_derivative(pv_baroclinic_derivative, u, v, hgt, **map_attrs):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, pv_baroclinic_derivative, u[:-1], v[:-1], hgt[:-1], **map_attrs)
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
        ax.set_title(r'$\frac{\partial PV}{\partial y}$' + ' @ 1000 hPa', fontsize=TITLE_SIZE)
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

    contour_levels = {}
    # Create levels for plot each variable
    for var in ds.data_vars:
        min_val = float(min(ds[var].min(), ds[var].min()))
        max_val = float(max(ds[var].max(), ds[var].max()))
        contour_levels[var] = np.linspace(min_val, max_val, 11)
    contour_levels['pv_baroclinic_derivative'] = np.linspace(np.min(pv_baroclinic_derivative), np.max(pv_baroclinic_derivative), 11)
    contour_levels['absolute_vorticity_derivative'] = np.linspace(np.min(absolute_vorticity_derivative), np.max(absolute_vorticity_derivative), 11)
    
    for level in ds.level:

        u_level = u.sel(level=level)
        v_level = v.sel(level=level)
        hgt_level = hgt.sel(level=level)

        pv_baroclinic_level = pv_baroclinic.sel(level=level)
        pv_baroclinic_derivative_level = pv_baroclinic_derivative.sel(level=level)
        pv_baroclinic_derivative_lon_mean = pv_baroclinic_derivative_level.mean('x')

        absolute_vorticity_level = absolute_vorticity.sel(level=level)
        absolute_vorticity_derivative_level = absolute_vorticity_derivative.sel(level=level)
        absolute_vorticity_derivative_lon_mean = absolute_vorticity_derivative_level.mean('x')

        egr_level = egr.sel(level=level)

        level_str = str(int(level))

        
        # Plot Baroclinic PV
        map_attrs = {
                'cmap': 'Blues_r',
                'title': r'$PV$' + f' @ {level_str} hPa',
                'levels': contour_levels['pv_baroclinic'],
                'units': 'PVU',
                'filename': f'composite_semi-lagrangian_pv_baroclinic_{level_str}hpa.png'
            }
        plot_variable(pv_baroclinic_level, u_level, v_level, hgt_level, **map_attrs)

        # Baroclinic PV derivative
        map_attrs = {
                'cmap': cmo.curl,
                'title': r'$\frac{\partial PV}{\partial y}$' + f' @ {level_str} hPa',
                'levels': contour_levels['pv_baroclinic_derivative'],
                'units': 'PVU',
                'filename': f'composite_semi-lagrangian_pv_baroclinic_derivative_{level_str}hpa.png'
            }
        plot_derivative(pv_baroclinic_derivative_level, u_level[:-1], v_level[:-1], hgt_level[:-1], **map_attrs)

        # Baroclinic PV derivative lon mean
        map_attrs = {
                'title': r'$\frac{\partial PV}{\partial y}$' + f' @ {level_str} hPa',
                'units': 'PVU',
                'filename': f'composite_semi-lagrangian_pv_baroclinic_derivative_lon_mean_{level_str}hpa.png'
            }
        plot_lon_mean(pv_baroclinic_derivative_lon_mean, **map_attrs)

        # Absolute Vorticity 
        map_attrs = {
            'cmap': 'Blues_r',
            'title': r'$\eta$' + f' @ {level_str} hPa',
            'levels': contour_levels['absolute_vorticity'],
            'units': r'$s^{-1}$',
            'filename': f'composite_semi-lagrangian_absolute_vorticity_{level_str}hpa.png'
        }
        plot_variable(absolute_vorticity_level, u_level, v_level, hgt_level, **map_attrs)

        # Absolute Vorticity derivative
        map_attrs = {
            'cmap': cmo.curl,
            'title': r'$\frac{\partial \eta}{\partial y}$' + f' @ {level_str} hPa',
            'levels': contour_levels['absolute_vorticity_derivative'],
            'units': r'$s^{-1}$',
            'filename': f'composite_semi-lagrangian_absolute_vorticity_derivative_{level_str}hpa.png'
        }
        plot_derivative(absolute_vorticity_derivative_level, u_level[:-1], v_level[:-1], hgt_level[:-1], **map_attrs)

        # Absolute Vorticity derivative lon mean
        map_attrs = {
                'title': r'$\frac{\partial \eta}{\partial y}$' + f' @ {level_str} hPa',
                'units': r's$^{-1}$',
                'filename': f'composite_semi-lagrangian_absolute_vorticitye_derivative_lon_mean_{level_str}hpa.png'
            }
        plot_lon_mean(absolute_vorticity_derivative_lon_mean, **map_attrs)

        # EGR
        map_attrs = {
            'cmap': 'Spectral_r',
            'title': 'EGR @ 1000 hPa',
            'levels': contour_levels['EGR'],
            'units': r'$d^{-1}$',
            'filename': f'composite_semi-lagrangian_EGR_{level_str}hpa.png'
        }
        plot_variable(egr_level, u_level, v_level, hgt_level, **map_attrs)
        

if __name__ == '__main__':
    main()
