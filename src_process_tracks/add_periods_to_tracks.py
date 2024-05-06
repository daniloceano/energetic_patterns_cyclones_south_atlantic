# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    add_periods_to_tracks.py                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/05 19:19:34 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/05 21:02:23 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from glob import glob
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, freeze_support

tracks_directory = '../tracks_SAt_filtered'
results_directory = '../../LEC_Results_energetic-patterns'

# Read tracks from the filtered dataset
tracks_filtered = pd.read_csv(f'{tracks_directory}/tracks_SAt_filtered.csv')

# Function to process each track in parallel
def process_track(track_id):
    track = tracks_filtered[tracks_filtered['track_id'] == track_id].copy()

    # Check if there are any corresponding results
    corresponding_results = glob(f'{results_directory}/{track_id}_ERA5_track')
    if len(corresponding_results) > 0:
        corresponding_results = corresponding_results[0]

        # Read periods from the LEC results
        try:
            periods = pd.read_csv(f"{corresponding_results}/periods.csv", index_col=0)
        except:
            return None

        # Add periods to the track
        for period in periods.index:
            period_start, period_end = periods.loc[period, 'start'], periods.loc[period, 'end']
            track.loc[(track['date'] >= period_start) & (track['date'] <= period_end), 'period'] = period

        # Drop rows with missing periods
        track = track.dropna(subset=['period'])

        return track
    else:
        return None

if __name__ == '__main__':
    # Ensure proper forking on Windows
    freeze_support()

    # Define the number of processes to use
    num_processes = cpu_count()

    # Parallelize processing tracks
    with Pool(num_processes) as pool:
        tracks_with_periods_list = list(tqdm(pool.imap(process_track, tracks_filtered['track_id'].unique()), total=len(tracks_filtered)))

    # Concatenate the tracks with periods
    tracks_with_periods = pd.concat([track for track in tracks_with_periods_list if track is not None])

    # Save the tracks with periods
    tracks_with_periods.to_csv(f'{tracks_directory}/tracks_SAt_filtered_with_periods.csv', index=False)
