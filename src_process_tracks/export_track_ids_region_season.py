# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_track_ids_region_season.py                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/25 19:29:52 by daniloceano       #+#    #+#              #
#    Updated: 2024/04/26 00:08:10 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script exports unique track IDs from a dataset based on meteorological tracking data. 
It supports exporting based on all regions and seasons, specific regions, or specific meteorological seasons (DJF, MAM, JJA, SON).

Author: daniloceano <danilo.oceano@gmail.com>
Created: 2024/04/25
Last Updated: 2024/04/25
"""

import pandas as pd
import os
from datetime import datetime

def get_season(date):
    """
    Determine the meteorological season from a given date.

    Parameters:
        date (str): The date string in the format 'YYYY-MM-DD HH:MM:SS'.

    Returns:
        str: The meteorological season ('DJF', 'MAM', 'JJA', 'SON').
    """
    month = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').month
    if month in (12, 1, 2):
        return 'DJF'
    elif month in (3, 4, 5):
        return 'MAM'
    elif month in (6, 7, 8):
        return 'JJA'
    elif month in (9, 10, 11):
        return 'SON'

def export_all_tracks(tracks_filtered, csv_directory):
    """
    Export all unique track IDs from the dataset to a CSV file.

    Parameters:
        tracks_filtered (DataFrame): The pandas DataFrame containing the track data.
        csv_directory (str): Directory path to save the CSV file.
    """
    unique_tracks = tracks_filtered['track_id'].unique()
    csv_path = os.path.join(csv_directory, 'all_track_ids.csv')
    pd.DataFrame(unique_tracks, columns=['track_id']).to_csv(csv_path, index=False)
    print("All track IDs exported successfully.")

def export_tracks_by_region(tracks_filtered, csv_directory):
    """
    Export track IDs for each unique region to separate CSV files.

    Parameters:
        tracks_filtered (DataFrame): The pandas DataFrame containing the track data.
        csv_directory (str): Directory path to save the CSV files.
    """
    for region in tracks_filtered['region'].unique():
        tracks_region = tracks_filtered[tracks_filtered['region'] == region]
        unique_tracks_region = tracks_region['track_id'].unique()
        csv_path = os.path.join(csv_directory, f'track_ids_{region}.csv')
        pd.DataFrame(unique_tracks_region, columns=['track_id']).to_csv(csv_path, index=False)
        print(f"Track IDs for region {region} exported successfully.")

def export_tracks_by_season(tracks_filtered, csv_directory):
    """
    Export track IDs for each unique season to separate CSV files.

    Parameters:
        tracks_filtered (DataFrame): The pandas DataFrame containing the track data.
        csv_directory (str): Directory path to save the CSV files.
    """
    tracks_filtered['season'] = tracks_filtered['date'].apply(get_season)
    for region in tracks_filtered['region'].unique():
        for season in tracks_filtered['season'].unique():
            tracks_season_region = tracks_filtered[(tracks_filtered['region'] == region) & (tracks_filtered['season'] == season)]
            unique_tracks_season_region = tracks_season_region['track_id'].unique()
            csv_path = os.path.join(csv_directory, f'track_ids_{region}_{season}.csv')
            pd.DataFrame(unique_tracks_season_region, columns=['track_id']).to_csv(csv_path, index=False)
            print(f"Track IDs for season {season} in region {region} exported successfully to {csv_path}.")

if __name__ == '__main__':
    csv_directory = '../csv_track_ids_by_region_season/'
    os.makedirs(csv_directory, exist_ok=True)

    tracks_filtered = pd.read_csv('../tracks_SAt_filtered/tracks_SAt_filtered.csv')
    export_all_tracks(tracks_filtered, csv_directory)
    export_tracks_by_region(tracks_filtered, csv_directory)
    export_tracks_by_season(tracks_filtered, csv_directory)