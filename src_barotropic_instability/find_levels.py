# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    find_levels.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 15:00:18 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/17 18:00:10 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script will analyse all the results from LEC analysis and detect in which vertical levels the
barotropic instability term is dominant.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns
from tqdm import tqdm  # Import tqdm for the progress bar

def read_and_process(file_path):
    df = pd.read_csv(file_path, index_col=0)
    if 'Ca' in file_path:
        df = - df
    if 'Ge' not in file_path:
        df.columns = [float(col) / 100 for col in df.columns]
    return df

def plot_violin_plot(df, term):
    plt.figure(figsize=(10, 10))  # Adjust size as needed
    ax = sns.boxplot(x='Vertical Level', y=f'{term} Value', data=df, whis=(0, 100))
    plt.xticks(rotation=90)  # Rotate labels for better visibility
    plt.axhline(y=0, color='r', linestyle='--', zorder=0)
    plt.xlabel('Vertical Levels [hPa]', fontsize=14)
    plt.ylabel(f'{term} ' + r'($W \times m^{-2}$)', fontsize=14)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'boxplot_{term.lower()}_by_levels.png'))

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
    results_ck = list(tqdm(executor.map(read_and_process, ck_paths), total=len(ck_paths), desc="Processing Files"))

with ProcessPoolExecutor(max_workers=4) as executor:
    results_ca = list(tqdm(executor.map(read_and_process, ca_paths), total=len(ca_paths), desc="Processing Files"))

with ProcessPoolExecutor(max_workers=4) as executor:
    results_ge = list(tqdm(executor.map(read_and_process, ge_paths), total=len(ge_paths), desc="Processing Files"))

# Step 3: Combine the data into a single DataFrame
combined_df_ck = pd.concat(results_ck)
combined_df_ca = pd.concat(results_ca)
combined_df_ge = pd.concat(results_ge)

# Step 4: Melt the DataFrame to make it compatible with seaborn's violin plot function
melted_df_ck = combined_df_ck.melt(var_name='Vertical Level', value_name='Ck Value')
melted_df_ca = combined_df_ca.melt(var_name='Vertical Level', value_name='Ca Value')
melted_df_ge = combined_df_ge.melt(var_name='Vertical Level', value_name='Ge Value')

# Step 5: Create violin plots
plot_violin_plot(melted_df_ck, 'Ck')
plot_violin_plot(melted_df_ca, 'Ca')
plot_violin_plot(melted_df_ge, 'Ge')