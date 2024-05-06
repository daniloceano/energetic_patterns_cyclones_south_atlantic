# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    check-results_tracks.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/06 09:41:14 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/06 09:44:12 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import glob
import os
import pandas as pd

# Directory pattern for periods.csv files
periods_csv_pattern = '../../LEC_Results_energetic-patterns/*/periods.csv'
# Directory pattern for *results.csv files
results_csv_pattern = '../../LEC_Results_energetic-patterns/*/*results.csv'

# Function to count directories with files matching a pattern
def count_directories_with_files(pattern):
    # Find all files matching the pattern
    files = glob.glob(pattern)
    # Extract unique directory paths
    directories = {os.path.dirname(file) for file in files}
    return len(directories)

# Count the total number of directories
total_directories = len(glob.glob('../../LEC_Results_energetic-patterns/*'))

# Count the number of directories with *results.csv files
directories_with_results = count_directories_with_files(results_csv_pattern)

# Count the number of directories with periods.csv files
directories_with_periods = count_directories_with_files(periods_csv_pattern)

# Calculate percentages
percentage_with_results = (directories_with_results / total_directories) * 100
percentage_with_periods = (directories_with_periods / total_directories) * 100

# Print the results
print("Total number of directories:", total_directories)
print("Number of directories with *results.csv files:", directories_with_results)
print("Percentage of directories with *results.csv files: {:.2f}%".format(percentage_with_results))
print("Number of directories with periods.csv files:", directories_with_periods)
print("Percentage of directories with periods.csv files: {:.2f}%".format(percentage_with_periods))

# Directories
results_directory = '../../LEC_Results_energetic-patterns'
tracks_directory = '../tracks_SAt_filtered'

# Find track IDs from results directory
results = glob.glob(f'{results_directory}/*_ERA5_track')
result_track_ids = [os.path.basename(result).split('_')[0] for result in results]

# Read track IDs from tracks filtered CSV
tracks_filtered = pd.read_csv(f'{tracks_directory}/tracks_SAt_filtered.csv')
tracks_filtered_track_ids = tracks_filtered['track_id'].unique()

# Count of unique track IDs in each source
num_unique_track_ids_results = len(result_track_ids)
num_unique_track_ids_tracks_filtered = len(tracks_filtered_track_ids)

# Correspondence of track IDs between results and tracks_filtered
common_track_ids = set(result_track_ids).intersection(set(tracks_filtered_track_ids))

# Number of track IDs that results have but tracks_filtered don't
missing_in_tracks_filtered = set(result_track_ids) - set(tracks_filtered_track_ids)

# Number of track IDs that tracks_filtered have but results don't
missing_in_results = set(tracks_filtered_track_ids) - set(result_track_ids)

print("Number of unique track IDs in results directory:", num_unique_track_ids_results)
print("Number of unique track IDs in tracks_filtered CSV:", num_unique_track_ids_tracks_filtered)
print("Number of common track IDs between results and tracks_filtered:", len(common_track_ids))

print("Number of track IDs in results but not in tracks_filtered:", len(missing_in_tracks_filtered))
print("Number of track IDs in tracks_filtered but not in results:", len(missing_in_results))
