# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_tracks.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/06 15:56:01 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/06 16:13:39 by daniloceano      ###   ########.fr        #
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

FILTERED_TRACKS = os.path.abspath('../tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv') # Path to filtered tracks
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
    # ax.add_feature(cartopy.feature.LAND)
    # ax.add_feature(cartopy.feature.OCEAN, facecolor="lightblue")

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

def plot_tracks(tracks_filtered_ids, figures_directory):
    plt.close('all')
    
    min_lon, min_lat = tracks_filtered_ids['lon vor'].min(), tracks_filtered_ids['lat vor'].min()
    max_lon, max_lat = tracks_filtered_ids['lon vor'].max(), tracks_filtered_ids['lat vor'].max()

    min_lon_buff = min_lon - 20 if min_lon - 20 > -180 else -180
    max_lon_buff = max_lon + 20 if max_lon + 20 < 180 else 180
    min_lat_buff = min_lat - 20 if min_lat - 20 > -90 else -90
    max_lat_buff = max_lat + 20 if max_lat + 20 < 90 else 90

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([min_lon_buff, max_lon_buff, min_lat_buff, max_lat_buff], crs=ccrs.PlateCarree())
    map_borders(ax)
    setup_map(ax)
    setup_gridlines(ax)

    for i, track_id in enumerate(tracks_filtered_ids['track_id'].unique()):
        track = tracks_filtered_ids[tracks_filtered_ids['track_id'] == track_id]
        region = track['region'].unique()[0]
        if region == 'ARG':
            color = COLORS[3]
        elif region == 'LA-PLATA':
            color = COLORS[1]
        elif region == 'SE-BR':
            color = COLORS[0]
        plt.plot(track['lon vor'], track['lat vor'], color=color, linewidth=2, alpha=0.5)

    plt.tight_layout()
    
    os.makedirs(figures_directory, exist_ok=True)
    fig.savefig(os.path.join(figures_directory, f'tracks.png'))
    print('Figure saved in directory: {}'.format(figures_directory))
    

def main():
    # Get ids of systems that will be analysed
    system_ids = pd.read_csv('systems_to_be_analysed.txt', header=None).values.flatten().tolist()

    tracks = pd.read_csv(FILTERED_TRACKS)
    tracks_filtered_ids = tracks[tracks['track_id'].isin(system_ids)]

    plot_tracks(tracks_filtered_ids, '../figures_barotropic_baroclinic_instability/')


if __name__ == '__main__':
    main()