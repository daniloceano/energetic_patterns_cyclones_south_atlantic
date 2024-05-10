# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    find_periods.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 17:10:09 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/10 13:52:10 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This scritps will find in which periods of the life cycle the barotropic instability term is more intense.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_event_directory(event_dir, track_with_periods):
    ck_levels_path = glob(f'{event_dir}/Ck_level*.csv')
    ca_levels_path = glob(f'{event_dir}/Ca_level*.csv')

    if not ck_levels_path or not ca_levels_path:
        print(f"No files found for {event_dir}")
        return pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames if files are not found

    ck_levels_df = pd.read_csv(ck_levels_path[0], index_col='time', parse_dates=True)
    ca_levels_df = - pd.read_csv(ca_levels_path[0], index_col='time', parse_dates=True)

    # Get periods data
    system_id = int(os.path.basename(event_dir).split('_')[0])
    track_system = track_with_periods[track_with_periods['track_id'] == system_id].dropna(subset=['period'])
    
    # Assigning phases
    for phase in track_system['period'].unique():
        phase_data = track_system[track_system['period'] == phase]
        phase_start = phase_data['date'].min()
        phase_end = phase_data['date'].max()
        ck_levels_df.loc[phase_start:phase_end, 'period'] = phase
        ca_levels_df.loc[phase_start:phase_end, 'period'] = phase

    # Removing nan and NaN values
    ck_levels_df = ck_levels_df[~ck_levels_df['period'].isin(['nan', 'NaN', ''])].dropna()
    ca_levels_df = ca_levels_df[~ca_levels_df['period'].isin(['nan', 'NaN', ''])].dropna()

    melted_df_ck = ck_levels_df.melt(id_vars='period', var_name='level', value_name='value')
    melted_df_ca = ca_levels_df.melt(id_vars='period', var_name='level', value_name='value')
    
    return melted_df_ck, melted_df_ca


def plot_boxplots_all_phases(data, term, phases_order):
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='period', y='value', data=data, order=phases_order, showfliers=False, hue='period', palette='deep')
    plt.axhline(y=0, color='r', linestyle='-', linewidth=1.25)
    plt.ylabel(f'{term.capitalize()} (W/m²)')
    plt.xlabel('Phase')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/boxplot_{term}_all_phases_all_systems.png')
    plt.close()

def plot_boxplots_each_phase(data, term, phases_order):
    fig, axes = plt.subplots(nrows=len(phases_order), figsize=(12, 2 * len(phases_order)), sharex=True)

    for i, phase in enumerate(phases_order):
        if data[data['period'] == phase].empty:
            continue
        sns.boxplot(x='level', y='value', data=data[data['period'] == phase], ax=axes[i], showfliers=False, hue='level', palette='deep')
        axes[i].axhline(y=0, color='r', linestyle='-', linewidth=1.25)
        axes[i].set_title(f'{phase}')
        axes[i].set_ylabel(f'{term.capitalize()} (W/m²)')
        axes[i].set_xlabel('Vertical Level (hPa)')

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/boxplot_{term}_each_phase.png')
    plt.close()

# Setup directories for figures
figures_dir = '../figures_barotropic_baroclinic_instability'
os.makedirs(figures_dir, exist_ok=True)

# Define the order and set of all possible phases
phases_order = ['incipient', 'intensification', 'mature', 'decay']

# Get all directories containing event data
directories_paths = glob('../../LEC_Results_fixed_framework_test/*')

# Get tracks with periods
track_with_periods = pd.read_csv('../tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv')

# Select track to process
selected_systems = pd.read_csv('systems_to_be_analysed.txt', header=None)[0].tolist()
selected_systems_str = [str(system) for system in selected_systems]
filtered_directories = [directory for directory in directories_paths if any(system_id in directory for system_id in selected_systems_str)]

# Prepare an empty DataFrame to collect all Ck data
all_ck_data = pd.DataFrame()
all_ca_data = pd.DataFrame()

# Use ThreadPoolExecutor to process directories in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    # Prepare futures and a dict to track them
    futures = [executor.submit(process_event_directory, dir_path, track_with_periods) for dir_path in filtered_directories]
    # Wrap futures with tqdm for a progress bar
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Events"):
        ck_data, ca_data = future.result()
        all_ck_data = pd.concat([all_ck_data, ck_data], ignore_index=True)
        all_ca_data = pd.concat([all_ca_data, ca_data], ignore_index=True)

# Now proceed with data cleaning and plotting as before
all_ck_data = all_ck_data.dropna(subset=['value'])
all_ca_data = all_ca_data.dropna(subset=['value'])

plot_boxplots_all_phases(all_ck_data, 'ck', phases_order)
plot_boxplots_all_phases(all_ca_data, 'ca', phases_order)
plot_boxplots_each_phase(all_ck_data, 'ck', phases_order)
plot_boxplots_each_phase(all_ca_data, 'ca', phases_order)