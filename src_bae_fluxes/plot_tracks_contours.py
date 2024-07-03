# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_tracks_contours.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/07/03 13:09:34 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/03 13:09:36 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter

# Configuration constants
datacrs = ccrs.PlateCarree()
proj = ccrs.AlbersEqualArea(central_longitude=-30, central_latitude=-35, standard_parallels=(-20.0, -60.0))
CSV_PATH = '../tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv'
OUTPUT_DIRECTORY = '../figures_bae_fluxes/tracks'
LINE_STYLES = {'default': 'solid'}
COLOR_PHASES = {'incipient': 'blue', 'mature': 'green', 'decay': 'red'}
PHASES = ['incipient', 'mature']

def gridlines(ax):
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle='dashed', alpha=0.6, color='#383838', linewidth=0.5)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10))
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    gl.xlabel_style = {'rotation': 0, 'ha': 'center', 'fontsize': 12}
    gl.ylabel_style = {'rotation': 0, 'ha': 'center', 'fontsize': 12}
    gl.bottom_labels = False
    gl.right_labels = False
    gl.top_labels = False
    gl.left_labels = False

    gl2 = ax.gridlines(draw_labels=True, zorder=2, linestyle='dashed', alpha=0, color='#383838')
    gl2.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20))
    gl2.ylocator = mticker.FixedLocator(np.arange(-90, 91, 20))
    gl2.xformatter = LongitudeFormatter()
    gl2.yformatter = LatitudeFormatter()
    gl2.xlabel_style = {'size': 14, 'color': '#383838'}
    gl2.ylabel_style = {'size': 14, 'color': '#383838'}
    gl2.xlabel_style = {'rotation': 0, 'ha': 'center', 'fontsize': 12}
    gl2.ylabel_style = {'rotation': 0, 'ha': 'center', 'fontsize': 12}
    gl2.bottom_labels = False
    gl2.right_labels = False
    gl2.top_labels = True
    gl2.left_labels = True

def plot_density_contours(ax, latitudes, longitudes, color):
    """
    Plot density contours for the given latitude and longitude data.
    """
    # Define the range based on the extent of the map
    range_lon = [-80, 50]
    range_lat = [-90, -15]

    # Create a 2D histogram to estimate density
    density, xedges, yedges = np.histogram2d(longitudes, latitudes, bins=[100, 100], range=[range_lon, range_lat])
    
    # Apply Gaussian filter for smoothing
    density = gaussian_filter(density, sigma=4)

    # Normalize the density
    density = density.T
    density = np.ma.masked_where(density == 0, density)  # Mask zero values
    norm_density = (density - density.min()) / (density.max() - density.min())

    # Plot density contours
    contour = ax.contour(xedges[:-1], yedges[:-1], norm_density, levels=np.linspace(0, 1, 3), 
                          colors=[color], transform=datacrs, linewidths=4)

def plot_complete_track(df, track_id, ax):
    """
    Plot the complete track for a given track_id.
    """
    track_data = df[df['track_id'] == track_id]
    latitudes = track_data['lat vor'].values
    longitudes = track_data['lon vor'].values

    ax.plot(longitudes, latitudes, linestyle='-', linewidth=2, transform=datacrs, alpha=0.8)
    gridlines(ax)

def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(CSV_PATH)

    # Read the list of systems to be analyzed
    selected_systems = pd.read_csv('systems_to_be_analysed.txt', header=None)[0].tolist()

    # Select only the selected systems
    df = df[df['track_id'].isin(selected_systems)]

    # Convert longitude to -180 to 180
    df['lon vor'] = np.where(df['lon vor'] > 180, df['lon vor'] - 360, df['lon vor'])

    # Plot density contours for each phase
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': proj})
    ax.set_extent([-70, -10, -20, -60], crs=datacrs)

    for phase in PHASES:
        phase_data = df[df['period'] == phase]
        if phase_data.empty:
            continue
        
        latitudes = phase_data['lat vor'].values
        longitudes = phase_data['lon vor'].values
        
        plot_density_contours(ax, latitudes, longitudes, COLOR_PHASES[phase])
    
    ax.coastlines(linewidth=1.2, color='k', alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1.2, edgecolor='k', alpha=0.8)
    ax.add_feature(cfeature.STATES, linestyle='-', linewidth=1.2, edgecolor='k', alpha=0.8)
    ax.add_feature(cfeature.LAND, color='gray', alpha=0.3)
    gridlines(ax)

    unique_handles = [mpatches.Patch(color=COLOR_PHASES[phase], label=phase) for phase in PHASES]
    ax.legend(handles=unique_handles, loc='upper right')

    fname = os.path.join(OUTPUT_DIRECTORY, f'track_plot_with_density.png')
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f'Track plot with density contours saved in {fname}')

    # Plot complete tracks for each track_id
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': proj})
    ax.set_extent([-70, -10, -20, -60], crs=datacrs)
    for track_id in selected_systems:
        plot_complete_track(df, track_id, ax)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1.2, edgecolor='k', alpha=0.8)
    ax.add_feature(cfeature.STATES, linestyle='-', linewidth=1.2, edgecolor='k', alpha=0.8)
    ax.add_feature(cfeature.LAND, color='gray', alpha=0.3)
    fname = os.path.join(OUTPUT_DIRECTORY, f'complete_track_plot.png')
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f'Complete track plot for track_id {track_id} saved in {fname}')

if __name__ == '__main__':
    main()