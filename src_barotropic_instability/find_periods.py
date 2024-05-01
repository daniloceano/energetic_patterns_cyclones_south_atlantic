# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    find_periods.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 17:10:09 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/01 17:23:27 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This scritps will find in which periods of the life cycle the barotropic instability term is more intense.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm

def plot_boxplots(data, term, phases_order):
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='phase', y='value', data=data, order=phases_order, whis=(0, 100))
    plt.ylabel(f'{term.capitalize()} {term} (W/m²)')
    plt.xlabel('Phase')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/boxplot_{term}_all_phases_all_systems.png')
    plt.show()
    plt.close()

def plot_violin_plots(data, term, phases_order):
    fig, axes = plt.subplots(nrows=len(phases_order), figsize=(12, 2 * len(phases_order)), sharex=True)

    for i, phase in enumerate(phases_order):
        if data[data['phase'] == phase].empty:
            continue
        sns.boxplot(x='level', y='value', data=data[data['phase'] == phase], ax=axes[i], whis=(0, 100))
        axes[i].set_title(f'{phase}')
        axes[i].set_ylabel(f'{term.capitalize()} {term} (W/m²)')
        axes[i].set_xlabel('Vertical Level (hPa)')

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/boxplot_{term}_each_phase.png')
    plt.close()

# Setup directories for figures
figures_dir = '../figures_barotropic_instability'
os.makedirs(figures_dir, exist_ok=True)

# Define the order and set of all possible phases
phases_order = ['incipient', 'intensification', 'mature', 'decay', 
                'intensification 2', 'mature 2', 'decay 2']

# Get all directories containing event data
directories_paths = glob('../../LEC_Results_energetic-patterns/*')

# Prepare an empty DataFrame to collect all Ck data
all_ck_data = pd.DataFrame()
all_ca_data = pd.DataFrame()

# Loop through each event directory to collect data, wrapped with tqdm for a progress bar
for event_dir in tqdm(directories_paths, desc="Processing Events"):
    ck_levels_path = glob(f'{event_dir}/Ck_level*.csv')
    ca_levels_path = glob(f'{event_dir}/Ca_level*.csv')
    periods_path = glob(f'{event_dir}/periods.csv')

    if not ck_levels_path or not ca_levels_path or not periods_path:
        continue  # Skip if necessary files are not found

    ck_levels_df = pd.read_csv(ck_levels_path[0], index_col='time', parse_dates=True)
    ca_levels_df = pd.read_csv(ca_levels_path[0], index_col='time', parse_dates=True)
    periods_df = pd.read_csv(periods_path[0], parse_dates=['start', 'end'], index_col=0)

    # Levels to hPa
    ck_levels_df.columns = [float(col) / 100 for col in ck_levels_df.columns]
    ca_levels_df.columns = [float(col) / 100 for col in ca_levels_df.columns]

    for phase in periods_df.index:
        phase_start, phase_end = periods_df.loc[phase, ['start', 'end']]

        ck_levels_df.sort_index(inplace=True)
        ck_levels_df.loc[phase_start:phase_end, 'phase'] = phase

        ca_levels_df.sort_index(inplace=True)
        ca_levels_df.loc[phase_start:phase_end, 'phase'] = phase

    # Melt the DataFrame for easier plotting and append to the collective DataFrame
    melted_df_ck = ck_levels_df.melt(id_vars='phase', var_name='level', value_name='value')
    all_ck_data = pd.concat([all_ck_data, melted_df_ck], ignore_index=True)

    melted_df_ca = ca_levels_df.melt(id_vars='phase', var_name='level', value_name='value')
    all_ca_data = pd.concat([all_ca_data, melted_df_ca], ignore_index=True)

# Drop rows with None values to avoid plotting issues
all_ck_data = all_ck_data.dropna(subset=['value'])
all_ca_data = all_ca_data.dropna(subset=['value'])

# 1. Boxplot plot for each phase with data from all systems
plot_boxplots(all_ck_data, 'ck', phases_order)
plot_boxplots(all_ca_data, 'ca', phases_order)

# 2. Panel with multiple plots, each displaying a different phase with violin plots for distinct vertical levels
plot_violin_plots(all_ck_data, 'ck', phases_order)
plot_violin_plots(all_ca_data, 'ca', phases_order)