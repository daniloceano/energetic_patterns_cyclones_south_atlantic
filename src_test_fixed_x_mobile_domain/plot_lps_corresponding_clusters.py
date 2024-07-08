# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_lps_corresponding_clusters.py                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/06 11:41:04 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/07 22:58:22 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
import json
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from lorenz_phase_space.phase_diagrams import Visualizer

def read_life_cycles(base_path):
    """
    Reads all CSV files in the specified directory and collects DataFrame for each system.
    """
    systems_energetics = {}

    for filename in tqdm(os.listdir(base_path), desc="Reading CSV files"):
        if filename.endswith('.csv'):
            file_path = os.path.join(base_path, filename)
            system_id = filename.split('_')[0]

            # Read the CSV file
            try:
                df = pd.read_csv(file_path)
                systems_energetics[system_id] = df
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return systems_energetics

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

def plot_all_systems_by_region_season(systems_energetics, id_list_directory, output_directory):

    # Get ids to plot
    json_file = f'{id_list_directory}/kmeans_results.json'
    json_data = pd.read_json(json_file)
    cluster_number = 3
    ids = json_data[f'Cluster {cluster_number}']['Cyclone IDs']

    # Initialize the Lorenz Phase Space plotter
    lps = Visualizer(LPS_type='mixed', zoom=True,
                    x_limits=[-100, 60],
                    y_limits=[-7, 20],
                    color_limits=[-20, 20])

    # Plot each system onto the Lorenz Phase Space diagram
    for system_id, df in tqdm(systems_energetics.items(), desc="Plotting systems"):
        if int(system_id) in ids:
            plot_system(lps, df)
    
    # Save the final plot
    plot_filename = f'lps_fixed_all_systems.png'

    plot_path = os.path.join(output_directory, plot_filename)
    lps.fig.savefig(plot_path)

    plt.close()
    print(f"Saved plot to {plot_path}")

def main():
    base_path = '../csv_fixed_framework_database_energy_by_periods'
    output_directory = '../figures_lps/test_fixed_framework/'
    clusters_directory = '../results_kmeans'

    os.makedirs(output_directory, exist_ok=True)

    # Read the energetics data for all systems
    systems_energetics = read_life_cycles(base_path)
    
    id_list_directory = os.path.join(clusters_directory, 'all_systems', 'IcItMD')
    plot_all_systems_by_region_season(systems_energetics, id_list_directory, output_directory)

if __name__ == "__main__":
    main()

