# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    find_levels.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 15:00:18 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/07 12:29:15 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script analyzes LEC analysis results to detect which vertical levels have dominant barotropic instability terms.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns
from tqdm import tqdm

def read_and_process(file_path):
    df = pd.read_csv(file_path, index_col=0)
    if 'Ca' in file_path:
        df = -df
    if 'Ge' not in file_path:
        df.columns = [float(col) / 100 for col in df.columns]
    df['Phase'] = os.path.basename(os.path.dirname(file_path))
    return df

def plot_combined_boxplots(df_ck, df_ca, df_ge, figures_dir, phase):
    # Ensure 'Vertical Level' is numeric and remove 'time' rows
    df_ck = df_ck[df_ck['Vertical Level'] != 'time']
    df_ca = df_ca[df_ca['Vertical Level'] != 'time']
    df_ge = df_ge[df_ge['Vertical Level'] != 'time']
    
    df_ck['Vertical Level'] = pd.to_numeric(df_ck['Vertical Level']) * 100
    df_ca['Vertical Level'] = pd.to_numeric(df_ca['Vertical Level']) * 100
    df_ge['Vertical Level'] = pd.to_numeric(df_ge['Vertical Level']) * 100

    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    fig.suptitle(f'Boxplots by Vertical Levels - {phase}', fontsize=16)

    sns.boxplot(ax=axs[0], x='Vertical Level', y='Ck Value', data=df_ck, whis=(0, 100))
    axs[0].set_title('Ck')
    axs[0].axhline(y=0, color='r', linestyle='--', zorder=0)
    axs[0].set_xlabel('Vertical Levels [hPa]')
    axs[0].set_ylabel(r'($W \times m^{-2}$)')
    axs[0].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    axs[0].tick_params(axis='both', labelsize=12)

    sns.boxplot(ax=axs[1], x='Vertical Level', y='Ca Value', data=df_ca, whis=(0, 100))
    axs[1].set_title('Ca')
    axs[1].axhline(y=0, color='r', linestyle='--', zorder=0)
    axs[1].set_xlabel('Vertical Levels [hPa]')
    axs[1].set_ylabel(r'($W \times m^{-2}$)')
    axs[1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    axs[1].tick_params(axis='both', labelsize=12)

    sns.boxplot(ax=axs[2], x='Vertical Level', y='Ge Value', data=df_ge, whis=(0, 100))
    axs[2].set_title('Ge')
    axs[2].axhline(y=0, color='r', linestyle='--', zorder=0)
    axs[2].set_xlabel('Vertical Levels [hPa]')
    axs[2].set_ylabel(r'($W \times m^{-2}$)')
    axs[2].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    axs[2].tick_params(axis='both', labelsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(figures_dir, f'boxplots_{phase}_by_levels.png'))

# Directory to save figures
figures_dir = '../figures_barotropic_baroclinic_instability'
os.makedirs(figures_dir, exist_ok=True)

# Select track to process
directories_paths = glob('../../LEC_Results_fixed_framework_test/*')
selected_systems = pd.read_csv('systems_to_be_analysed.txt', header=None)[0].tolist()
selected_systems_str = [str(system) for system in selected_systems]
filtered_directories = [directory for directory in directories_paths if any(system_id in directory for system_id in selected_systems_str)]

# Step 1: Get all file paths
ck_paths = [os.path.join(directory, 'Ck_level.csv') for directory in filtered_directories]
ca_paths = [os.path.join(directory, 'Ca_level.csv') for directory in filtered_directories]
ge_paths = [os.path.join(directory, 'Ge_level.csv') for directory in filtered_directories]

# Step 2: Use ProcessPoolExecutor to read files in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    results_ck = list(tqdm(executor.map(read_and_process, ck_paths), total=len(ck_paths), desc="Processing Ck Files"))
    results_ca = list(tqdm(executor.map(read_and_process, ca_paths), total=len(ca_paths), desc="Processing Ca Files"))
    results_ge = list(tqdm(executor.map(read_and_process, ge_paths), total=len(ge_paths), desc="Processing Ge Files"))

# Step 3: Combine the data into single DataFrames
combined_df_ck = pd.concat(results_ck).reset_index()
combined_df_ca = pd.concat(results_ca).reset_index()
combined_df_ge = pd.concat(results_ge).reset_index()

# Step 4: Melt the DataFrame to make it compatible with seaborn's boxplot plot function
melted_df_ck = combined_df_ck.melt(id_vars='Phase', var_name='Vertical Level', value_name='Ck Value')
melted_df_ca = combined_df_ca.melt(id_vars='Phase', var_name='Vertical Level', value_name='Ca Value')
melted_df_ge = combined_df_ge.melt(id_vars='Phase', var_name='Vertical Level', value_name='Ge Value')

# Step 5: Create boxplot plots for each phase
for phase in melted_df_ck['Phase'].unique():
    phase_df_ck = melted_df_ck[melted_df_ck['Phase'] == phase]
    phase_df_ca = melted_df_ca[melted_df_ca['Phase'] == phase]
    phase_df_ge = melted_df_ge[melted_df_ge['Phase'] == phase]

    plot_combined_boxplots(phase_df_ck, phase_df_ca, phase_df_ge, figures_dir, phase)
