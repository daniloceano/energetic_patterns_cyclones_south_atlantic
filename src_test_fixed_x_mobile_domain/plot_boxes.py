# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_boxes.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/06 15:55:30 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/06 15:55:31 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob

import cartopy
import cartopy.crs as ccrs
from shapely.geometry.polygon import Polygon
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature, COASTLINE
from cartopy.feature import BORDERS

FILTERED_TRACKS = os.path.abspath('../tracks_SAt_filtered/tracks_SAt_filtered.csv') # Path to filtered tracks

COLORS = ['#3B95BF', '#87BF4B', '#BFAB37', '#BF3D3B', '#873e23', '#A13BF0']
MARKERS = ['s', 'o', '^', 'v', '<', '>']
MARKER_COLORS = ['#59c0f0', '#b0fa61', '#f0d643', '#f75452', '#f07243', '#bc6ff7']
LINESTYLE = '-'
LINEWIDTH = 3
TEXT_COLOR = '#383838'
MARKER_EDGE_COLOR = 'grey'
LEGEND_FONT_SIZE = 10
AXIS_LABEL_FONT_SIZE = 12
TITLE_FONT_SIZE = 18

def setup_gridlines(ax):
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle='-', alpha=0.8, color=TEXT_COLOR, linewidth=0.25)
    gl.xlabel_style = {'size': 14, 'color': TEXT_COLOR}
    gl.ylabel_style = {'size': 14, 'color': TEXT_COLOR}
    gl.bottom_labels = None
    gl.right_labels = None

def setup_map(ax):
    ax.coastlines(zorder=1)
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN, facecolor="lightblue")

def map_borders(ax):
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

def plot_box_limits(df_boxes, figures_directory, plot_mean=True):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    min_lon, min_lat = df_boxes['min_lon'].min(), df_boxes['min_lat'].min()
    max_lon, max_lat = df_boxes['max_lon'].max(), df_boxes['max_lat'].max()

    ax.set_extent([min_lon-20, max_lon+20, max_lat+20, min_lat-20], crs=ccrs.PlateCarree())
    map_borders(ax)
    setup_map(ax)
    setup_gridlines(ax)

    if plot_mean:
        # Plot mean polygon
        mean_pgon = Polygon([(df_boxes['min_lon'].mean(), df_boxes['min_lat'].mean()), 
                             (df_boxes['min_lon'].mean(), df_boxes['max_lat'].mean()), 
                             (df_boxes['max_lon'].mean(), df_boxes['max_lat'].mean()), 
                             (df_boxes['max_lon'].mean(), df_boxes['min_lat'].mean()), 
                             (df_boxes['min_lon'].mean(), df_boxes['min_lat'].mean())])
        ax.add_geometries([mean_pgon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor=COLORS[3], linewidth=3, alpha=1, zorder=3)
    else:
        # Plot individual polygons
        for _, row in df_boxes.iterrows():
            min_lon, max_lon, min_lat, max_lat = row['min_lon'], row['max_lon'], row['min_lat'], row['max_lat']
            pgon = Polygon([(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat), (min_lon, min_lat)])
            ax.add_geometries([pgon], crs=ccrs.PlateCarree(), facecolor=COLORS[3], edgecolor='none', linewidth=1, alpha=0.1, zorder=3)

    plt.tight_layout()

    os.makedirs(figures_directory, exist_ok=True)
    plot_type = 'mean' if plot_mean else 'individual'
    fig.savefig(os.path.join(figures_directory, f'box_limits_{plot_type}.png'))
    print('Figure saved in directory: {}'.format(figures_directory))
    

def main():
    figures_directory = '../figures_test_fixed_framework'
    # Get ids of systems that will be analysed
    results_directories = sorted(glob('../../LEC_Results_fixed_framework_test/*'))
    system_ids = [int(os.path.basename(directory).split('_')[0]) for directory in results_directories]

    tracks = pd.read_csv(FILTERED_TRACKS)
    tracks_filtered_ids = tracks[tracks['track_id'].isin(system_ids)]

    df_boxes = pd.DataFrame(columns=['track_id', 'min_lat', 'max_lat', 'min_lon', 'max_lon'])

    for track_id in system_ids:
        track_data = tracks_filtered_ids[tracks_filtered_ids['track_id'] == track_id]

        # Get mininum and maximum latitude and longitude and create a 15x15 degree bounding box
        min_lat = track_data['lat vor'].min()
        max_lat = track_data['lat vor'].max()
        min_lon = track_data['lon vor'].min()
        max_lon = track_data['lon vor'].max()
        bbox_lat_min = np.floor(min_lat - 7.5)
        bbox_lat_max = np.ceil(max_lat + 7.5)
        bbox_lon_min = np.floor(min_lon - 7.5)
        bbox_lon_max = np.ceil(max_lon + 7.5)

        tmp = pd.DataFrame([[track_id, bbox_lat_min, bbox_lat_max, bbox_lon_min, bbox_lon_max]], columns=['track_id', 'min_lat', 'max_lat', 'min_lon', 'max_lon'])
        
        if df_boxes.empty:
            df_boxes = tmp
        else:
            df_boxes = pd.concat([df_boxes, tmp], ignore_index=True)

    df_boxes_mean = df_boxes.groupby(['min_lat', 'max_lat', 'min_lon', 'max_lon']).mean().reset_index()

    plot_box_limits(df_boxes, figures_directory, plot_mean=False)
    plot_box_limits(df_boxes_mean, figures_directory, plot_mean=True)

if __name__ == '__main__':
    main()