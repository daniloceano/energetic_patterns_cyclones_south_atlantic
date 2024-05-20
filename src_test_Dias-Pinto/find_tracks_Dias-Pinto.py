# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    find_tracks_Dias-Pinto.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/19 20:10:53 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/20 09:44:51 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from glob import glob

# Path to the directory containing the new track files
TRACKS_DIR = '../tracks_SAt/'

# Colors for plotting
COLORS = ['#3B95BF', '#87BF4B', '#BFAB37', '#BF3D3B', '#873e23', '#A13BF0', '#59c0f0', '#b0fa61', '#f0d643', '#f75452', '#f07243', '#bc6ff7']

def setup_gridlines(ax):
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle='-', alpha=0.8, color='grey', linewidth=0.25)
    gl.xlabel_style = {'size': 14, 'color': 'grey'}
    gl.ylabel_style = {'size': 14, 'color': 'grey'}
    gl.bottom_labels = None
    gl.right_labels = None

def setup_map(ax):
    ax.coastlines(zorder=1)

def map_borders(ax):
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='none'))
    states = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_1_states_provinces_lines')
    ax.add_feature(states, edgecolor='#283618', linewidth=1)
    cities = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='populated_places')
    ax.add_feature(cities, edgecolor='#283618', linewidth=1)
    countries = NaturalEarthFeature(category='cultural', scale='50m', facecolor='none', name='admin_0_countries')
    ax.add_feature(countries, edgecolor='black', linewidth=1)

def read_all_tracks(tracks_dir):
    all_files = glob(os.path.join(tracks_dir, '*.csv'))
    df_list = []
    for file in all_files:
        df = pd.read_csv(file, header=None, names=['track_id', 'date', 'lon vor', 'lat vor', 'vor42'])
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

def find_matching_systems(tracks_df, systems):
    matching_systems = []
    for idx, system in enumerate(systems):
        start_date = pd.to_datetime(system['start_date'])
        end_date = pd.to_datetime(system['end_date'])
        mask = (tracks_df['date'] >= start_date) & (tracks_df['date'] <= end_date)
        matching_tracks = tracks_df[mask]

        if not matching_tracks.empty:
            system_id = matching_tracks['track_id'].unique()
            for sid in system_id:
                system_tracks = matching_tracks[matching_tracks['track_id'] == sid]
                first_position = system_tracks.iloc[0]
                
                if idx == 0:
                    lon_min, lon_max, lat_min, lat_max = -60, -50, -30, -20
                elif idx == 1:
                    lon_min, lon_max, lat_min, lat_max = -80, -60, -30, -20
                elif idx == 2:
                    lon_min, lon_max, lat_min, lat_max = -80, -70, -45, -40
                
                if (first_position['lon vor'] >= lon_min) and (first_position['lon vor'] <= lon_max) and \
                   (first_position['lat vor'] >= lat_min) and (first_position['lat vor'] <= lat_max):
                    matching_systems.append({
                        'start_date': system['start_date'],
                        'end_date': system['end_date'],
                        'matching_track_ids': [sid],
                        'matching_tracks': system_tracks
                    })
    return matching_systems

def plot_all_tracks(matching_systems, figures_directory):
    plt.close('all')
    
    all_tracks = pd.concat([system['matching_tracks'] for system in matching_systems])
    min_lon, min_lat = all_tracks['lon vor'].min(), all_tracks['lat vor'].min()
    max_lon, max_lat = all_tracks['lon vor'].max(), all_tracks['lat vor'].max()

    min_lon_buff = min_lon - 20 if min_lon - 20 > -180 else -180
    max_lon_buff = max_lon + 20 if max_lon + 20 < 180 else 180
    min_lat_buff = min_lat - 20 if min_lat - 20 > -90 else -90
    max_lat_buff = max_lat + 20 if max_lat + 20 < 90 else 90

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([min_lon_buff, max_lon_buff, min_lat_buff, max_lat_buff], crs=ccrs.PlateCarree())
    map_borders(ax)
    setup_map(ax)
    setup_gridlines(ax)

    color_idx = 0
    for system in matching_systems:
        for track_id in system['matching_track_ids']:
            track = system['matching_tracks'][system['matching_tracks']['track_id'] == track_id]
            ax.plot(track['lon vor'], track['lat vor'], color=COLORS[color_idx % len(COLORS)], linewidth=2, label=f'Track ID {track_id}')
            color_idx += 1

    ax.legend(loc='upper right', fontsize=14)
    plt.tight_layout()
    
    os.makedirs(figures_directory, exist_ok=True)
    fig.savefig(os.path.join(figures_directory, f'tracks.png'))
    print('Figure saved in directory: {}'.format(figures_directory))

def export_matching_systems(matching_systems, results_directory):
    os.makedirs(results_directory, exist_ok=True)
    for system in matching_systems:
        for track_id in system['matching_track_ids']:
            track = system['matching_tracks'][system['matching_tracks']['track_id'] == track_id]
            output_file = os.path.join(results_directory, f'track_{track_id}.csv')
            track.to_csv(output_file, index=False)
            print(f"Track {track_id} saved to {output_file}")

def main():
    systems = [
        {'start_date': '2005-08-08 12:00:00', 'end_date': '2005-08-14 00:00:00'},
        {'start_date': '2007-06-22 00:00:00', 'end_date': '2007-06-26 00:00:00'},
        {'start_date': '2008-06-29 12:00:00', 'end_date': '2008-07-03 00:00:00'}
    ]

    tracks = read_all_tracks(TRACKS_DIR)
    tracks['date'] = pd.to_datetime(tracks['date'])
    matching_systems = find_matching_systems(tracks, systems)

    for system in matching_systems:
        print(f"System from {system['start_date']} to {system['end_date']} matches track IDs: {system['matching_track_ids']}")
    
    plot_all_tracks(matching_systems, '../figures_test_Dias-Pinto/')
    export_matching_systems(matching_systems, '../results_test_Dias_Pinto/')

if __name__ == '__main__':
    main()
