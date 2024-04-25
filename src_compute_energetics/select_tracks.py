# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    select_tracks.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/01/19 16:22:17 by daniloceano       #+#    #+#              #
#    Updated: 2024/04/25 19:13:14 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


"""
This script processes track data from the SAt dataset, focusing on weather systems with genesis in specific regions. 
It filters out tracks that do not originate within predefined geographic boundaries and those spending a significant portion of their lifecycle over the continent. 
The script utilizes parallel processing for efficiency and outputs the filtered data for further analysis.
"""


import pandas as pd
import numpy as np
from glob import glob
from multiprocessing import Pool
import concurrent.futures
import multiprocessing
from tqdm import tqdm
import geopandas as gpd
import os 
import logging


# Constants defining the geographic boundaries of regions of interest.
REGIONS = {
    "SE-BR": [(-52, -38, -37, -23)],
    "LA-PLATA": [(-69, -38, -52, -23)],
    "ARG": [(-70, -55, -50, -39)],
    "SE-SAO": [(-15, -55, 30, -37)],
    "SA-NAM": [(8, -33, 20, -21)],
    "AT-PEN": [(-65, -69, -44, -58)],
    "WEDDELL": [(-65, -85, -10, -72)]
}

PATH_TO_RAW_DATA = '../raw_data/SAt'

def read_csv_file(filepath):
    """
    Reads a CSV file and returns its content as a pandas DataFrame.

    Parameters:
    filepath (str): Path to the CSV file.

    Returns:
    DataFrame: The content of the CSV file.
    """
    return pd.read_csv(filepath, header=None)

def get_tracks(logger):
    """
    Reads and merges track data from CSV files located in the "../tracks_SAt" directory.
    Adjusts longitude values and filters tracks based on the first time step being within predefined regions.
    
    Parameters:
    logger (logging.Logger): Logger for logging information.

    Returns:
    pd.DataFrame: The merged and filtered track data with columns for track ID, date, longitude, latitude, and vor42.
    """
    logger.info("Reading raw track files...")
    file_list = glob("../tracks_SAt/*.csv")
    with Pool() as pool:
        dfs = pool.map(read_csv_file, tqdm(file_list))
    logger.info("Merging tracks...")
    tracks = pd.concat(dfs, ignore_index=True)
    tracks.columns = ['track_id', 'date', 'lon vor', 'lat vor', 'vor42']
    tracks['lon vor'] = np.where(tracks['lon vor'] > 180, tracks['lon vor'] - 360, tracks['lon vor'])
    logger.info("Done.")
    return tracks

def configure_logging():
    """
    Configures logging for the script.
    """
    # Configure logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('./log.select_tracks.txt', mode='w')

    # Create formatters and add them to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def is_first_step_in_region(track_group, region_bounds):
    """
    Determines if the system have genesis within the specified region.
    Handles empty track groups by returning False.

    Parameters:
    track_group (DataFrame): Grouped data for a single system.
    region_bounds (tuple): A tuple of (lon_min, lat_min, lon_max, lat_max) defining the region.

    Returns:
    bool: True if the first time step of the track is in the region, False otherwise.
    """
    if track_group.empty:
        return False
    first_step = track_group.iloc[0]
    lon_min, lat_min, lon_max, lat_max = region_bounds
    lon = first_step['lon vor']
    lat = first_step['lat vor']
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max

def filter_tracks_for_single_region(grouped_tracks, region_name, bounds):
    """
    Filters tracks for a single region.

    Parameters:
    grouped_tracks (DataFrameGroupBy): Grouped track data.
    region_name (str): Name of the region.
    bounds (tuple): Boundary coordinates of the region.

    Returns:
    DataFrame: Filtered tracks for the specified region.
    """
    matching_tracks = grouped_tracks.filter(lambda x: is_first_step_in_region(x, bounds)).index
    result = pd.Series(region_name, index=matching_tracks)
    return result

def filter_tracks_by_region(tracks, logger):
    """
    Filter tracks, selecting only systems with genesis in the defined regions.
    Adds a 'region' column to the DataFrame indicating the region of genesis.
    Uses parallel processing to improve performance.

    Parameters:
    tracks (DataFrame): The input tracks data.
    logger (logging.Logger): Logger for logging information.

    Returns:
    DataFrame: The input tracks data with an additional 'region' column.
    """
    logger.info("Filtering tracks by region...")
    grouped_tracks = tracks.groupby('track_id')
    tracks['region'] = None  # Initialize the 'region' column
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(filter_tracks_for_single_region, grouped_tracks, region_name, bounds[0]) 
                   for region_name, bounds in REGIONS.items()]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(REGIONS)):
            result = future.result()
            tracks.loc[result.index, 'region'] = result

    logger.info("Done.")
    return tracks

def prepare_tracks(tracks, continent_gdf):
    """
    Prepare the tracks DataFrame for efficient continent checks.
    """
    # Convert all track points to a GeoSeries
    points = gpd.points_from_xy(tracks['lon vor'], tracks['lat vor'])
    tracks_geo = gpd.GeoDataFrame(tracks, geometry=points, crs=continent_gdf.crs)
    return tracks_geo

def check_tracks_on_continent(tracks, track_ids, continent_gdf, threshold_percentage):
    """
    Check a batch of tracks for the percentage of time spent on the continent.
    """
    results = []
    for cyclone_id in tqdm(track_ids, desc="Processing Batch"):
        cyclone_track = tracks[tracks['track_id'] == cyclone_id]
        on_continent_count = cyclone_track['geometry'].within(continent_gdf.unary_union).sum()
        total_time_steps = len(cyclone_track)
        percentage_on_continent = (on_continent_count / total_time_steps) * 100
        results.append((cyclone_id, percentage_on_continent < threshold_percentage))
    return results

def filter_tracks_by_continent(tracks, continent_gdf, threshold_percentage=80):
    """
    Filter tracks based on the percentage of time spent over the continent.
    """
    # Prepare the tracks DataFrame
    tracks = prepare_tracks(tracks, continent_gdf)

    # Create batches of track IDs
    unique_cyclone_ids = tracks['track_id'].unique()
    num_cores = multiprocessing.cpu_count() or 1  # Ensure num_cores is at least 1
    id_batches = np.array_split(unique_cyclone_ids, num_cores)

    # Process in parallel
    with multiprocessing.Pool() as pool:
        results = pool.starmap(
            check_tracks_on_continent,
            [(tracks, batch, continent_gdf, threshold_percentage) for batch in id_batches]
        )

    # Flatten results and filter tracks
    valid_track_ids = [cyclone_id for batch in results for cyclone_id, is_valid in batch if is_valid]
    return tracks[tracks['track_id'].isin(valid_track_ids)]

def verify_track_numbers(tracks, logger):
    for region_name in REGIONS:
            num_tracks = len(tracks[tracks['region'] == region_name].groupby('track_id'))
            logger.info(f"Number of tracks in {region_name}: {num_tracks}")
    num_unmatched_tracks = len(tracks[tracks['region'].isnull()].groupby('track_id'))
    logger.info(f"Number of unmatched tracks: {num_unmatched_tracks}")

if __name__ == "__main__":
    logger = configure_logging()

    # Get the tracks
    logger.info("Starting track processing")
    tracks = get_tracks(logger)

    # Filter the tracks by region
    logger.info("Filtering tracks by region")
    tracks = filter_tracks_by_region(tracks, logger)

    # Here we will work only with tracks that have genesis in one of the regions
    # over South American coast: ARG, LA-PLATA and SE-BR.
    filtered_tracks = tracks[tracks['region'].isin(['ARG', 'LA-PLATA', 'SE-BR'])]

    # Filter the tracks, excluding systems that spend 80% of their time in the over the continent
    continent_shapefile = '../natural_earth_continents/ne_50m_land.shp'
    continent_gdf = gpd.read_file(continent_shapefile)
    logger.info("Filtering tracks by continent")
    filtered_tracks_no_continental = filter_tracks_by_continent(filtered_tracks, continent_gdf)

    verify_track_numbers(tracks, logger)

    os.makedirs('../tracks_SAt_filtered', exist_ok=True)
    output_file = '../tracks_SAt_filtered/tracks_SAt_filtered.csv'
    filtered_tracks_no_continental.to_csv(output_file, index=False)
    logger.info(f"Filtered tracks saved to {output_file}")
    logger.info("Track processing completed.")
    