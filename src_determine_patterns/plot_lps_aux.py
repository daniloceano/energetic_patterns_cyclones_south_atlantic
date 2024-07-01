# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_lps_aux.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/02/27 17:45:49 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/01 18:13:06 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

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

def read_patterns(results_path, PHASES, TERMS):
    """
    Read the energetics data for patterns from a JSON file.

    Args:
        results_path (str): The path to the directory containing the JSON file.
        PHASES (list): A list of strings representing the phases.
        TERMS (list): A list of strings representing the terms.

    Returns:
        list: A list of pandas DataFrames, each representing the energetics data for a pattern.
    """
     # Read the energetics data for patterns
    patterns_json = glob(f'{results_path}/kmeans_results.json')
    results = pd.read_json(patterns_json[0])

    clusters_center = results.loc['Cluster Center']

    patterns_energetics = []
    for i in range(len(clusters_center)):
        cluster_center = np.array(clusters_center.iloc[i])
        cluster_array = np.array(cluster_center).reshape(len(TERMS), len(PHASES))

        df = pd.DataFrame(cluster_array, columns=PHASES, index=TERMS).T
        patterns_energetics.append(df)
    return patterns_energetics, clusters_center, results

def plot_system(lps, df, lps_type):
    """
    Plots the Lorenz Phase Space diagram for a single system
    """
    if lps_type == 'mixed':
        x_axis = df['Ck']
        y_axis = df['Ca']
    elif lps_type == 'imports':
        x_axis = df['BAe']
        y_axis = df['BKe']
        
    # Generate the phase diagram
    lps.plot_data(
        x_axis=x_axis,
        y_axis=y_axis,
        marker_color=df['Ge'],
        marker_size=df['Ke']
    )

def determine_global_limits(systems_energetics, lps_type):
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    color_min, color_max = float('inf'), float('-inf')
    size_min, size_max = float('inf'), float('-inf')

    if lps_type == 'mixed':
        x_term = 'Ck'
        y_term = 'Ca'
    elif lps_type == 'imports':
        x_term = 'BAe'
        y_term = 'BKe'

    # Determine global limits
    if type(systems_energetics) == list:
        for df in systems_energetics:
            x_min = min(x_min, df[x_term].min())
            x_max = max(x_max, df[x_term].max())
            y_min = min(y_min, df[y_term].min())
            y_max = max(y_max, df[y_term].max())
            color_min = min(color_min, df['Ge'].min())
            color_max = max(color_max, df['Ge'].max())
            size_min = min(size_min, df['Ke'].min())
            size_max = max(size_max, df['Ke'].max())
    
    elif type(systems_energetics) == dict:
        for _, df in systems_energetics.items():
            x_min = min(x_min, df[x_term].min())
            x_max = max(x_max, df[x_term].max())
            y_min = min(y_min, df[y_term].min())
            y_max = max(y_max, df[y_term].max())
            color_min = min(color_min, df['Ge'].min())
            color_max = max(color_max, df['Ge'].max())
            size_min = min(size_min, df['Ke'].min())
            size_max = max(size_max, df['Ke'].max())

    return [x_min - 5, x_max + 5], [y_min - 5, y_max + 5], [color_min, color_max], [size_min, size_max]