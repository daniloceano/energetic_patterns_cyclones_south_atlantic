# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    find_levels.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 15:00:18 by daniloceano       #+#    #+#              #
#    Updated: 2024/04/24 13:37:03 by daniloceano      ###   ########.fr        #
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
    df.columns = [float(col) / 100 for col in df.columns]
    return df

# Directory to save figures
figures_dir = '../figures_barotropic_instability'
os.makedirs(figures_dir, exist_ok=True)

# Step 1: Get all file paths
file_paths = glob('../../LEC_Results_energetic-patterns/*/Ck_level.csv')

# Step 2: Use ProcessPoolExecutor to read files in parallel
with ProcessPoolExecutor() as executor:
    # Wrap file_paths with tqdm to create a progress bar
    results = list(tqdm(executor.map(read_and_process, file_paths), total=len(file_paths), desc="Processing Files"))

# Step 3: Combine the data into a single DataFrame
combined_df = pd.concat(results)

# Step 4: Melt the DataFrame to make it compatible with seaborn's violin plot function
melted_df = combined_df.melt(var_name='Vertical Level', value_name='Ck Value')

# Step 5: Create violin plots
plt.figure(figsize=(10, 10))  # Adjust size as needed
ax = sns.boxplot(x='Vertical Level', y='Ck Value', data=melted_df, whis=(0, 100))
plt.xticks(rotation=90)  # Rotate labels for better visibility
plt.axhline(y=0, color='r', linestyle='--', zorder=0)
plt.xlabel('Vertical Levels [hPa]', fontsize=14)
plt.ylabel(r'Ck ($W \times m^{-2}$)', fontsize=14)
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(figures_dir, 'boxplot_ck_levels.png'))