# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    find_periods.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 17:10:09 by daniloceano       #+#    #+#              #
#    Updated: 2024/04/23 20:18:04 by daniloceano      ###   ########.fr        #
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

# Loop through each event directory to collect data
for event_dir in directories_paths:
    ck_levels_path = glob(f'{event_dir}/Ck_level*.csv')
    periods_path = glob(f'{event_dir}/periods.csv')

    if not ck_levels_path or not periods_path:
        continue  # Skip if necessary files are not found

    ck_levels_df = pd.read_csv(ck_levels_path[0], index_col='time', parse_dates=True)
    periods_df = pd.read_csv(periods_path[0], parse_dates=['start', 'end'], index_col=0 )

    for phase in periods_df.index:
        phase_start, phase_end = periods_df.loc[phase, ['start', 'end']]
        ck_levels_df.loc[phase_start:phase_end, 'phase'] = phase

    # Melt the DataFrame for easier plotting and append to the collective DataFrame
    melted_df = ck_levels_df.melt(id_vars='phase', var_name='level', value_name='value')
    all_ck_data = pd.concat([all_ck_data, melted_df], ignore_index=True)

# Drop rows with None values to avoid plotting issues
all_ck_data = all_ck_data.dropna(subset=['value'])

# 1. Violin plot for each phase with data from all systems
plt.figure(figsize=(14, 8))
sns.violinplot(x='phase', y='value', data=all_ck_data, order=phases_order)
plt.title('Distribution of Ck Values Across All Phases and Systems')
plt.ylabel('Ck (W/m²)')
plt.xlabel('Phase')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{figures_dir}/all_phases_all_systems_violin_plot.png')
plt.show()
plt.close()

# 2. Panel with multiple plots, each displaying a different phase with violin plots for distinct vertical levels
fig, axes = plt.subplots(nrows=len(phases_order), figsize=(12, 2 * len(phases_order)), sharex=True)

for i, phase in enumerate(phases_order):
    if all_ck_data[all_ck_data['phase'] == phase].empty:
        continue
    sns.violinplot(x='level', y='value', data=all_ck_data[all_ck_data['phase'] == phase], ax=axes[i])
    axes[i].set_title(f'Distribution of Ck Values for {phase}')
    axes[i].set_ylabel('Ck (W/m²)')
    axes[i].set_xlabel('Vertical Level (hPa)')

plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f'{figures_dir}/ck_each_phase_violin_plots.png')
plt.show()
plt.close()