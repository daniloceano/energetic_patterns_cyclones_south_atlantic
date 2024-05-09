# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_study_case.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/08 13:15:01 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/09 00:41:10 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


"""
This script will plot maps with the derivative of PV in y, as to display if barotropic instability is occuring.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
import cmocean.cm as cmo
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from shapely.geometry.polygon import Polygon

TITLE_SIZE = 16
TICK_LABEL_SIZE = 12
FIGURES_DIR = '../figures_test_fixed_framework/study_case/'
CRS = ccrs.PlateCarree()


def plot_map(ax, data, cmap, levels, title, transform=ccrs.PlateCarree()):
    """Plot potential vorticity using dynamic normalization based on data values."""
    levels_min, levels_max = np.amin(levels), np.amax(levels)
    if levels_min < 0 and levels_max > 0:
        norm = colors.TwoSlopeNorm(vmin=levels_min, vcenter=0, vmax=levels_max)
    else:
        norm = colors.Normalize(vmin=levels_min, vmax=levels_max)
    
    cf = ax.contourf(data['longitude'], data['latitude'], data, cmap=cmap, norm=norm, transform=transform, levels=levels, extend='both')
    colorbar = plt.colorbar(cf, ax=ax, pad=0.1, orientation='horizontal', shrink=0.5)
    colorbar.ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    colorbar.ax.xaxis.get_major_formatter().set_scientific(True)
    colorbar.ax.xaxis.get_major_formatter().set_powerlimits((0, 0))

    # Automatically adjust the number of ticks on the colorbar
    colorbar.locator = MaxNLocator(nbins=7)  # Adjust 'nbins' to the desired number of ticks
    colorbar.update_ticks()

    colorbar.ax.set_xticklabels(colorbar.ax.get_xticklabels(), rotation=45, fontsize=TICK_LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)

def plot_box(ax, min_lon, min_lat, max_lon, max_lat, var):
    mean_pgon = Polygon([(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat),
                         (max_lon, min_lat), (min_lon, min_lat)]) 
    if var == 'pv_baroclinic' or var == 'absolute_vorticity':
        edgecolor = 'k'
    else:
        edgecolor = 'red'
    ax.add_geometries([mean_pgon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor=edgecolor, linewidth=1, alpha=0.8, zorder=3)

def main(filepath='19931164_results_study_case.nc'):
    # Create figures directory if it doesn't exist
    os.makedirs(FIGURES_DIR, exist_ok=True)

    tracks_with_periods = pd.read_csv('../tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv')
    ds_original = xr.open_dataset(filepath)

    # Choose study case and extract track
    track_id_study_case = int(ds_original.track_id.values)
    track_study_case = tracks_with_periods[tracks_with_periods['track_id'] == track_id_study_case].copy()
    track_study_case['date'] = pd.to_datetime(track_study_case['date'])

    # Extract central point
    date_study_case = pd.to_datetime(ds_original['time'].values)
    central_lon = float(track_study_case[track_study_case['date'] == date_study_case]['lon vor'])
    central_lat = float(track_study_case[track_study_case['date'] == date_study_case]['lat vor'])

    # Slicing data around central point
    min_lon, max_lon = central_lon - 7.5, central_lon + 7.5
    min_lat, max_lat = central_lat - 7.5, central_lat + 7.5
    ds_sliced = ds_original.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))

    # List of variables to plot
    var_names = ['pv_baroclinic', 'absolute_vorticity', 'EGR']
    var_labels = {
        'pv_baroclinic': r'$PV$',
        'absolute_vorticity': r'$\eta$',
        'EGR': 'EGR',
        'pv_baroclinic_derivative': r'$\frac{\partial PV}{\partial t}$',
        'absolute_vorticity_derivative': r'$\frac{\partial \eta}{\partial t}$',
    }

    # Calculate derivatives
    for var in var_names:
        ds_original[f'{var}_derivative'] = ds_original[var].diff('latitude')
        ds_original[f'{var}_derivative_lon_mean'] = ds_original[f'{var}_derivative'].mean('longitude')
        ds_sliced[f'{var}_derivative'] = ds_sliced[var].diff('latitude')
        ds_sliced[f'{var}_derivative_lon_mean'] = ds_sliced[f'{var}_derivative'].mean('longitude')

    levels = {}
    # Create levels for plot each variable
    for var in ds_original.data_vars:
        min_val = float(min(ds_original[var].min(), ds_sliced[var].min()))
        max_val = float(max(ds_original[var].max(), ds_sliced[var].max()))
        if var == 'EGR':
            max_val *= 0.7
        levels[var] = np.linspace(min_val, max_val, 51)

    # Plotting
    for ds, method in zip([ds_original, ds_sliced], ['fixed', 'semi-lagrangian']):
        for var in var_names:
            cmap = cmo.balance if var != 'EGR' else cmo.thermal
            method_label = 'F' if method == 'fixed' else 'SL'
        
            # Create map
            fig, ax = plt.subplots(subplot_kw={'projection': CRS})
            if var != 'EGR':
                title = fr'{var_labels[var]}' + f' ({method_label})'
            else:
                title = fr'{var_labels[var]}' + f' ({method_label}) ({ds[var].mean().values:.2e})'
            plot_map(ax, ds[var], cmap, levels[var], title)
            plot_box(ax, min_lon, min_lat, max_lon, max_lat, var)
            filename = f'{var}_{method}.png'
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, filename))
            print(f'Saved {filename}')

            if var != 'EGR':
                derivative_var = f'{var}_derivative'
                fig, ax = plt.subplots(subplot_kw={'projection': CRS})
                plot_map(ax, ds[derivative_var], cmo.curl, levels[var], fr'{var_labels[derivative_var]}' + f' ({method_label})')
                plot_box(ax, min_lon, min_lat, max_lon, max_lat, var)
                plt.tight_layout()
                filename = f'{derivative_var}_{method}.png'
                plt.savefig(os.path.join(FIGURES_DIR, filename))
                print(f'Saved {filename}')

                derivative_lon_mean_var = f'{var}_derivative_lon_mean'
                fig = plt.figure(figsize=(5, 5))
                ax = plt.gca()
                ax.plot(ds[derivative_lon_mean_var], ds[derivative_lon_mean_var].latitude, color='#003049', linewidth=3)
                ax.axvline(0, color='#c1121f', linestyle='--', linewidth=0.5)
                ax.set_title(fr'{var_labels[derivative_var]}' + f' lon_mean ({method_label})', fontsize=TITLE_SIZE)
                plt.xlim(levels[derivative_lon_mean_var].min(), levels[derivative_lon_mean_var].max())
                ax.set_yticks([])
                plt.tick_params(axis='x', labelsize=TICK_LABEL_SIZE)
                plt.tight_layout()
                filename = f'{derivative_lon_mean_var}_{method}.png'
                plt.savefig(os.path.join(FIGURES_DIR, filename))
                print(f'Saved {filename}')


if __name__ == '__main__':
    main()