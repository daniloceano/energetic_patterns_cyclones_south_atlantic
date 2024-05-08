# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_study_case.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/08 13:15:01 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/08 15:40:43 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


"""
This script will plot maps with the derivative of PV in y, as to display if barotropic instability is occuring.
"""

import os
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
import cmocean.cm as cmo
import matplotlib.ticker as ticker

TITLE_SIZE = 16
TICK_LABEL_SIZE = 12
FIGURES_DIR = '../figures_test_fixed_framework/study_case/'
CRS = ccrs.PlateCarree()


def plot_map(ax, data, cmap, title, vmin, vmax, transform=ccrs.PlateCarree()):
    """Plot potential vorticity using dynamic normalization based on data values."""
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cf = ax.contourf(data['longitude'], data['latitude'], data, cmap=cmap, norm=norm, transform=transform)
    colorbar = plt.colorbar(cf, ax=ax, pad=0.1, orientation='horizontal', shrink=0.5)
    colorbar.ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    colorbar.ax.xaxis.get_major_formatter().set_scientific(True)
    colorbar.ax.xaxis.get_major_formatter().set_powerlimits((0, 0))
    colorbar.ax.set_xticklabels(colorbar.ax.get_xticklabels(), rotation=45, fontsize=TICK_LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)

def determine_global_bounds(datasets, var_names):
    """Determines global normalization bounds for plotting, centered around zero."""
    bounds = {}
    for var in var_names:
        data_min = min([float(ds[var].min()) for ds in datasets])
        data_max = max([float(ds[var].max()) for ds in datasets])
        max_abs_value = max(abs(data_min), abs(data_max))
        bounds[var] = (-max_abs_value, max_abs_value)
    return bounds

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
    ds_sliced = ds_original.sel(longitude=slice(central_lon-7.5, central_lon+7.5), latitude=slice(central_lat+7.5, central_lat-7.5))

    # List of variables to plot
    var_names = ['pv_baroclinic', 'absolute_vorticity', 'EGR']
    derivatives = ['diff', 'diff']

    # Calculate derivatives
    for var in var_names:
        ds_original[f'{var}_derivative'] = ds_original[var].diff('latitude')
        ds_sliced[f'{var}_derivative'] = ds_sliced[var].diff('latitude')

    # Determine global bounds for all plots
    bounds = determine_global_bounds([ds_original, ds_sliced], var_names + [f'{var}_derivative' for var in var_names])

    # Plotting
    for ds, label in zip([ds_original, ds_sliced], ['original', 'sliced']):
        for var in var_names:
            fig, ax = plt.subplots(subplot_kw={'projection': CRS})
            plot_map(ax, ds[var], cmo.balance, f'{var} {label}', *bounds[var])
            filename = f'{var}_{label}.png'
            plt.savefig(os.path.join(FIGURES_DIR, filename))
            print(f'Saved {filename}')

            derivative_var = f'{var}_derivative'
            fig, ax = plt.subplots(subplot_kw={'projection': CRS})
            plot_map(ax, ds[derivative_var], cmo.curl, f'{derivative_var} {label}', *bounds[derivative_var])
            filename = f'{derivative_var}_{label}.png'
            plt.savefig(os.path.join(FIGURES_DIR, filename))
            print(f'Saved {filename}')

if __name__ == '__main__':
    main()