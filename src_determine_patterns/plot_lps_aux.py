# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_lps_aux.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/02/27 17:45:49 by daniloceano       #+#    #+#              #
#    Updated: 2024/04/25 09:51:17 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from lorenz_phase_space.phase_diagrams import Visualizer

"""
Auxiliary functions for plotting LPS.
"""

def read_life_cycles(base_path, region=False):
    """
    Reads all CSV files in the specified directory and collects DataFrame for each system.
    """
    systems_energetics = {}

    if region:
        try:
            ids_region_file = f"../csv_life_cycle_analysis/track_ids_{region}.txt"
            ids_by_region = pd.read_csv(ids_region_file)
        except Exception as e:
            print(f"Error processing {ids_region_file}: {e}")
    
    for filename in tqdm(os.listdir(base_path), desc="Reading CSV files"):
        if filename.endswith('.csv'):
            file_path = os.path.join(base_path, filename)
            system_id = filename.split('_')[0]

            # Filter systems by region if specified
            if region:
                if system_id not in ids_by_region.values:
                    continue

            # Read the CSV file
            try:
                df = pd.read_csv(file_path)
                systems_energetics[system_id] = df
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return systems_energetics

def read_patterns(patterns_by_life_cycle_paths):
    """
    Reads all CSV files in the specified directory and collects DataFrame for each system.
    """
    patterns_energetics = {}
    for directory in patterns_by_life_cycle_paths:
        life_cycle_type = os.path.basename(directory)
        patterns = glob(f'{directory}/*')
        for pattern in patterns:
            df = pd.read_csv(pattern)
            cluster = os.path.basename(pattern).split('_')[1]
            patterns_energetics[f"{life_cycle_type}_{cluster}"] = df
    return patterns_energetics

def plot_system(lps, df):
    """
    Plots the Lorenz Phase Space diagram for a single system
    """
    # Generate the phase diagram
    lps.plot_data(
        x_axis=df['Ck'],
        y_axis=df['Ca'],
        marker_color=df['Ge'],
        marker_size=df['Ke']
    )

def determine_global_limits(systems_energetics):
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    color_min, color_max = float('inf'), float('-inf')
    size_min, size_max = float('inf'), float('-inf')

    for df in systems_energetics.values():
        x_min = min(x_min, df['Ck'].min())
        x_max = max(x_max, df['Ck'].max())
        y_min = min(y_min, df['Ca'].min())
        y_max = max(y_max, df['Ca'].max())
        color_min = min(color_min, df['Ge'].min())
        color_max = max(color_max, df['Ge'].max())
        size_min = min(size_min, df['Ke'].min())
        size_max = max(size_max, df['Ke'].max())

    return [x_min - 5, x_max + 5], [y_min - 5, y_max + 5], [color_min, color_max], [size_min, size_max]