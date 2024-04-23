# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    find_periods.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 17:10:09 by daniloceano       #+#    #+#              #
#    Updated: 2024/04/23 17:27:33 by daniloceano      ###   ########.fr        #
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

# Directory for saving figures
figures_dir = '../figures_barotropic_instability'
os.makedirs(figures_dir, exist_ok=True)

# Read your data (dummy paths assumed to be replaced with actual paths)
directories_paths = glob('../../LEC_Results_energetic-patterns/*')

# Assuming one of the directories is used as a dummy example
dummy = directories_paths[1]

ck_levels_path = glob(f'{dummy}/Ck_level*.csv')
periods_path = glob(f'{dummy}/periods.csv')

ck_levels_df = pd.read_csv(ck_levels_path[0], index_col='time', parse_dates=True)
periods_df = pd.read_csv(periods_path[0], index_col=0)

# Convert period date columns to datetime
periods_df['start'] = pd.to_datetime(periods_df['start'])
periods_df['end'] = pd.to_datetime(periods_df['end'])

# Tag each row in ck_levels_df with its respective phase
def assign_phase(row):
    for phase, period in periods_df.iterrows():
        if period['start'] <= row.name <= period['end']:
            return phase
    return 'unknown'

ck_levels_df['phase'] = ck_levels_df.apply(assign_phase, axis=1)

# 1. Violin plots without discriminating vertical levels
plt.figure(figsize=(10, 6))
sns.violinplot(x='phase', y='value', data=ck_levels_df.melt(id_vars='phase', var_name='level', value_name='value'))
plt.title('Distribution of Ck Values Across Different Phases')
plt.ylabel('Ck (W/m²)')
plt.xlabel('Phase')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{figures_dir}/ck_phases_violin_plot.png')
plt.show()

# 2. Multiple violin plots for each phase, detailing vertical levels
unique_phases = ck_levels_df['phase'].unique()
for phase in unique_phases:
    if phase != 'unknown':
        plt.figure(figsize=(12, 8))
        phase_data = ck_levels_df[ck_levels_df['phase'] == phase]
        sns.violinplot(x='level', y='value', data=phase_data.melt(id_vars='phase', var_name='level', value_name='value'))
        plt.title(f'Distribution of Ck Values for {phase}')
        plt.ylabel('Ck (W/m²)')
        plt.xlabel('Vertical Level (hPa)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'{figures_dir}/ck_{phase}_violin_plot.png')
        plt.show()
