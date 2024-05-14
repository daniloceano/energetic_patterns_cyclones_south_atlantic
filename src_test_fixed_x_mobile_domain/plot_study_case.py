# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_study_case.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/08 13:15:01 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/14 16:34:48 by daniloceano      ###   ########.fr        #
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
import cartopy
import cartopy.crs as ccrs
import matplotlib.colors as colors
import cmocean.cm as cmo
import matplotlib.ticker as ticker
from glob import glob
from shapely.geometry.polygon import Polygon
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature, COASTLINE
from cartopy.feature import BORDERS
from concurrent.futures import ProcessPoolExecutor

TITLE_SIZE = 16
TICK_LABEL_SIZE = 12
LABEL_SIZE = 12
GRID_LABEL_SIZE = 12
FIGURES_DIR = '../figures_test_fixed_framework/'
CRS = ccrs.PlateCarree()
COLORS = ['#3B95BF', '#87BF4B', '#BFAB37', '#BF3D3B', '#873e23', '#A13BF0']

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
    
    # Plot the colour filled contours
    cf = ax.contourf(data.longitude, data.latitude, data, cmap=cmap, norm=norm, transform=transform, levels=levels, extend='both')

    # Add hgt as a contour
    ax.contour(data.longitude, data.latitude, hgt, colors='gray', linestyles='dashed', linewidths=2, transform=transform)

    # Add quiver
    min_u = np.min(u)
    scale_factor = 300 if '1000' in title else 800  # Adjust these values to tune the arrow
    skip_n = 10 if '1000' in title else 15
    skip = (slice(None, None, skip_n), slice(None, None, skip_n))
    qu = ax.quiver(data.longitude[skip[0]], data.latitude[skip[0]], u[skip], v[skip], transform=transform, zorder=1,
              width=0.008, headwidth=2, headlength=2, headaxislength=2,  scale=scale_factor)
    
    # Quiver key
    label = 10 if '1000' in title else 30
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

    # Add coastlines
    ax.coastlines(linewidth=1, color='k')

    # Add states and borders
    ax.add_feature(cartopy.feature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m', edgecolor='black', facecolor='none'), linestyle='-')
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', edgecolor='black')

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

def plot_box(ax, min_lon, min_lat, max_lon, max_lat):
    mean_pgon = Polygon([(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat),
                         (max_lon, min_lat), (min_lon, min_lat)]) 
    edgecolor = 'red'
    ax.add_geometries([mean_pgon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor=edgecolor, linewidth=1, alpha=0.8, zorder=3)

def plot_study_case(filepath, tracks_with_periods, figures_dir):
    ds_original = xr.open_dataset(filepath)

    # Choose study case and extract track
    track_id_study_case = int(ds_original.track_id.values)
    track_study_case = tracks_with_periods[tracks_with_periods['track_id'] == track_id_study_case].copy()
    track_study_case['date'] = pd.to_datetime(track_study_case['date'])

    # Extract central point
    date_study_case = pd.to_datetime(ds_original['time'].values)
    date_study_case = pd.to_datetime(date_study_case.values[0])
    central_lon = float(track_study_case[track_study_case['date'] == date_study_case]['lon vor'].iloc[0])
    central_lat = float(track_study_case[track_study_case['date'] == date_study_case]['lat vor'].iloc[0])

    # Make data 2D
    ds_original = ds_original.sel(time=date_study_case)

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
        'pv_baroclinic_derivative': r'$\frac{\partial PV}{\partial y}$',
        'absolute_vorticity_derivative': r'$\frac{\partial \eta}{\partial y}$',
    }

    # Calculate derivatives
    for var in var_names:
        ds_original[f'{var}_derivative'] = ds_original[var].diff('latitude')
        ds_original[f'{var}_derivative_lon_mean'] = ds_original[f'{var}_derivative'].mean('longitude')
        ds_sliced[f'{var}_derivative'] = ds_sliced[var].diff('latitude')
        ds_sliced[f'{var}_derivative_lon_mean'] = ds_sliced[f'{var}_derivative'].mean('longitude')

    # Units dictionary
    units = {
        'pv_baroclinic': 'PVU',
        'absolute_vorticity': r'$s^{-1}$',
        'EGR': r'$d^{-1}$'
        }
    
    # Vertical levels to plot
    vertical_levels = {
        'pv_baroclinic': '1000 hPa',
        'absolute_vorticity': '250 hPa',
        'EGR': '1000 hPa'
    }

    levels = {}
    # Create levels for plot each variable
    for var in ds_original.data_vars:
        min_val = float(min(ds_original[var].min(), ds_sliced[var].min()))
        max_val = float(max(ds_original[var].max(), ds_sliced[var].max()))
        levels[var] = np.linspace(min_val, max_val, 101)

    # Plotting
    for ds, method in zip([ds_original, ds_sliced], ['fixed', 'semi-lagrangian']):
        for var in var_names:
            # Choose method
            method_label = 'F' if method == 'fixed' else 'SL'
        
            # Create map
            fig, ax = plt.subplots(subplot_kw={'projection': CRS})
            if var != 'EGR':
                title = fr'{var_labels[var]}' + f' @ {vertical_levels[var]} ({method_label})'
                cmap = cmo.balance
            else:
                title = fr'{var_labels[var]}' + f' @ {vertical_levels[var]} ({method_label})\n({ds[var].mean().values:.2f})'
                cmap = 'Spectral_r'

            # Create plot
            plot_attrs = {
                'cmap': cmap,
                'levels': levels[var],
                'title': title,
                'units': units[var]
            }

            # Choose variable
            if 'absolute_vorticity' in var:
                u, v, hgt = ds['u_250'], ds['v_250'], ds['hgt_250']
            else:
                u, v, hgt = ds['u_1000'], ds['v_1000'], ds['hgt_1000']

            # Plot the data
            plot_map(ax, ds[var], u, v, hgt, **plot_attrs)
            plot_box(ax, min_lon, min_lat, max_lon, max_lat)
            filename = f'{var}_{method}.png'
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, filename))
            print(f'Saved {filename}')

            # Plot the derivative
            if var != 'EGR':
                derivative_var = f'{var}_derivative'
                plot_attrs = {
                    'cmap': cmo.curl,
                    'levels': levels[derivative_var],
                    'title': fr'{var_labels[derivative_var]} @ {vertical_levels[var]} ({method_label})',
                    'units': units[var]
                }
                fig, ax = plt.subplots(subplot_kw={'projection': CRS})
                plot_map(ax, ds[derivative_var], u, v, hgt, **plot_attrs)
                plot_box(ax, min_lon, min_lat, max_lon, max_lat)
                plt.tight_layout()
                filename = f'{derivative_var}_{method}.png'
                plt.savefig(os.path.join(figures_dir, filename))
                print(f'Saved {filename}')

            # Plot the derivative lon mean
            if var != 'EGR':
                derivative_lon_mean_var = f'{var}_derivative_lon_mean'
                fig = plt.figure(figsize=(5, 5))
                ax = plt.gca()
                ax.plot(ds[derivative_lon_mean_var], ds[derivative_lon_mean_var].latitude, color='#003049', linewidth=3)
                ax.axvline(0, color='#c1121f', linestyle='--', linewidth=0.5)
                title = fr'{var_labels[derivative_var]}' + f' @ {vertical_levels[var]} ({method_label})'
                ax.set_title(title, fontsize=TITLE_SIZE)
                ax.set_ylabel('Latitude', fontsize=LABEL_SIZE)
                ax.set_xlabel(r's$^{-1}$', fontsize=LABEL_SIZE)
                ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
                plt.tight_layout()
                filename = f'{derivative_lon_mean_var}_{method}.png'
                plt.savefig(os.path.join(figures_dir, filename))
                print(f'Saved {filename}')

def setup_map(ax):
    TEXT_COLOR = '#383838'

    # Add land feature (no facecolor)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='none'))

    # Add state borders
    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces_lines')
    ax.add_feature(states, edgecolor='#283618', linewidth=1)

    # Add populated places (cities)
    cities = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='populated_places')
    ax.add_feature(cities, edgecolor='#283618', linewidth=1)

    # Add country borders
    countries = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_0_countries')
    ax.add_feature(countries, edgecolor='black', linewidth=1)

    # Add coastlines
    ax.coastlines(zorder=1)

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle='-', alpha=0.8, color=TEXT_COLOR, linewidth=0.25)
    gl.xlabel_style = {'size': 14, 'color': TEXT_COLOR}
    gl.ylabel_style = {'size': 14, 'color': TEXT_COLOR}
    gl.bottom_labels = None
    gl.right_labels = None


def plot_domain_box(track_id, tracks_with_periods, filepath, figures_dir):
    plt.close('all')
    # Get track data
    track_data = tracks_with_periods[tracks_with_periods['track_id'] == int(track_id)]

    # Get mininum and maximum latitude and longitude and create a 15x15 degree bounding box
    min_lat = track_data['lat vor'].min()
    max_lat = track_data['lat vor'].max()
    min_lon = track_data['lon vor'].min()
    max_lon = track_data['lon vor'].max()
    bbox_lat_min = np.floor(min_lat - 7.5)
    bbox_lat_max = np.ceil(max_lat + 7.5)
    bbox_lon_min = np.floor(min_lon - 7.5)
    bbox_lon_max = np.ceil(max_lon + 7.5)

    # Get position of system
    ds = xr.open_dataset(filepath)
    ds_time = pd.to_datetime(ds.time.values).item()
    track_data['date'].loc[:] = pd.to_datetime(track_data['date'].copy())
    track_time = track_data[track_data['date'] == ds_time]

    # Create a dataframe with the box limits
    df_boxes = pd.DataFrame([[track_id, bbox_lat_min, bbox_lat_max, bbox_lon_min, bbox_lon_max]], columns=['track_id', 'min_lat', 'max_lat', 'min_lon', 'max_lon'])

    # Plot the box limits
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set extent
    min_lon, min_lat = df_boxes['min_lon'].min(), df_boxes['min_lat'].min()
    max_lon, max_lat = df_boxes['max_lon'].max(), df_boxes['max_lat'].max()

    # Add 20 degrees on each side
    min_lon_buff = min_lon - 10 if min_lon - 10 > -180 else -180
    max_lon_buff = max_lon + 10 if max_lon + 10 < 180 else 180
    min_lat_buff = min_lat - 10 if min_lat - 10 > -80 else -80
    max_lat_buff = max_lat + 10 if max_lat + 10 < 0 else 0

    # Set extent
    ax.set_extent([min_lon_buff, max_lon_buff, min_lat_buff, max_lat_buff], crs=ccrs.PlateCarree())
    setup_map(ax)

    # Plot polygons
    for _, row in df_boxes.iterrows():
        min_lon, max_lon, min_lat, max_lat = row['min_lon'], row['max_lon'], row['min_lat'], row['max_lat']
        pgon = Polygon([(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat), (min_lon, min_lat)])
        ax.add_geometries([pgon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor=COLORS[3], linewidth=2, alpha=1, zorder=3)

    # Plot position of system
    ax.scatter(track_time['lon vor'].values, track_time['lat vor'].values, s=100, edgecolor='k', facecolor='none')
    ax.scatter(track_time['lon vor'].values, track_time['lat vor'].values, s=100, edgecolor='none', facecolor='k', alpha=0.3)

    # Save figure
    plt.tight_layout()
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(os.path.join(figures_dir, f'box_limits_{track_id}.png'))
    print('Figure saved in directory: {}'.format(figures_dir))

def plot_track(track_id, tracks_with_periods, filepath, figures_dir):
    plt.close('all')
    
    # Subset the track data
    track_data = tracks_with_periods[tracks_with_periods['track_id'] == int(track_id)]
    min_lon, min_lat = track_data['lon vor'].min(), track_data['lat vor'].min()
    max_lon, max_lat = track_data['lon vor'].max(), track_data['lat vor'].max()

    # Get mininum and maximum latitude and longitude and create a 15x15 degree bounding box
    min_lon_buff = min_lon - 20 if min_lon - 20 > -180 else -180
    max_lon_buff = max_lon + 20 if max_lon + 20 < 180 else 180
    min_lat_buff = min_lat - 20 if min_lat - 20 > -90 else -90
    max_lat_buff = max_lat + 20 if max_lat + 20 < 90 else 90

    # Get position of system
    ds = xr.open_dataset(filepath)
    ds_time = pd.to_datetime(ds.time.values).item()
    track_data.loc[:, 'date'] = pd.to_datetime(track_data['date'])
    track_time = track_data[track_data['date'] == ds_time]

    # Create 15x15 degree bounding box around the system
    central_lat, central_lon = track_time['lat vor'].item(), track_time['lon vor'].item()
    bbox_lon_min = central_lon - 7.5
    bbox_lon_max = central_lon + 7.5
    bbox_lat_min = central_lat - 7.5
    bbox_lat_max = central_lat + 7.5
    df_boxes = pd.DataFrame([[track_id, bbox_lat_min, bbox_lat_max, bbox_lon_min, bbox_lon_max]], columns=['track_id', 'min_lat', 'max_lat', 'min_lon', 'max_lon'])

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([min_lon_buff, max_lon_buff, min_lat_buff, max_lat_buff], crs=ccrs.PlateCarree())
    setup_map(ax)

    # Plot track
    plt.plot(track_data['lon vor'], track_data['lat vor'], color=COLORS[3], linewidth=5, alpha=0.75)

    # Plot position of system
    ax.scatter(track_time['lon vor'].values, track_time['lat vor'].values, s=100, edgecolor='k', facecolor='none', zorder=100)
    ax.scatter(track_time['lon vor'].values, track_time['lat vor'].values, s=100, edgecolor='none', facecolor='k', alpha=0.3, zorder=100)

    # Plot polygons
    for _, row in df_boxes.iterrows():
        min_lon, max_lon, min_lat, max_lat = row['min_lon'], row['max_lon'], row['min_lat'], row['max_lat']
        pgon = Polygon([(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat), (min_lon, min_lat)])
        ax.add_geometries([pgon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor=COLORS[3], linewidth=2, alpha=1, zorder=3)

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, f'track_{track_id}.png'))
    print('Track figure saved in directory: {}'.format(figures_dir))

def process_file(filepath, tracks_with_periods):
    # Get system id
    print(f'Plotting {filepath}')
    system_id = os.path.basename(filepath).split('_')[0]

    # Create figures directory
    figures_dir = os.path.join(FIGURES_DIR, f'{system_id}_study_case')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot study case variables
    plot_study_case(filepath, tracks_with_periods, figures_dir)

    # Plot domain box
    plot_domain_box(system_id, tracks_with_periods, filepath, figures_dir)

    # Plot track
    plot_track(system_id, tracks_with_periods, filepath, figures_dir)

def main():
    # Get list of files
    files = sorted(glob('../results_nc_files/composites_test_fixed_x_mobile/*study_case.nc'))

    # Get tracks
    tracks_with_periods = pd.read_csv('../tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv')

    # Use ProcessPoolExecutor to parallelize the processing
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        futures = [executor.submit(process_file, filepath, tracks_with_periods) for filepath in files]
        for future in futures:
            future.result()

if __name__ == '__main__':
    main()
