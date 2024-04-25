# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    determine_export_genesis_region.py                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/25 19:20:33 by daniloceano       #+#    #+#              #
#    Updated: 2024/04/25 19:21:33 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Exports csv with the number of systems in each genesis region and 
the number of systems that were successfully processed. 
"""

import os
import pandas as pd
from glob import glob

output_path = "../csv_life_cycle_analysis/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

tracks_database_path = "../tracks_SAt_filtered/"
tracks = pd.read_csv(f"{tracks_database_path}/tracks_SAt_filtered.csv")

processed_systems_path = "../csv_database_energy_by_periods/"
processed_systems_files = glob(f"{processed_systems_path}/*_averages.csv")

# Extract system IDs from the processed files' names
processed_systems_ids = [os.path.basename(f).split('_')[0] for f in processed_systems_files]

genesis_regions = tracks['region'].unique()
region_summary = []

for region in genesis_regions:
    filtered_tracks = tracks[tracks['region'] == region]
    systems = list(filtered_tracks['track_id'].astype(str).unique())

    # Count how many systems in this region have been processed successfully
    successful_cases = len([system for system in systems if system in processed_systems_ids])
    
    region_summary.append([region, len(systems), successful_cases])

# Convert summary list to DataFrame
df_summary = pd.DataFrame(region_summary, columns=['Region', 'Total Systems', 'Successful Cases'])

# Save to CSV
summary_csv_path = os.path.join(output_path, "genesis_region_summary.csv")
df_summary.to_csv(summary_csv_path, index=False)
print(f"Summary saved to {summary_csv_path}")
