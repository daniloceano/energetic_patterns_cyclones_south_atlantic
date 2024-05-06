# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_lps_fixed_all_systems.py                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/05 18:56:28 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/06 11:37:10 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
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

def plot_all_systems_by_region_season(systems_energetics, output_directory):

    # Initialize the Lorenz Phase Space plotter
    lps = Visualizer(LPS_type='mixed', zoom=False)
    lps_zoom = Visualizer(LPS_type='mixed', zoom=True)

    # Plot each system onto the Lorenz Phase Space diagram
    for system_id, df in tqdm(systems_energetics.items(), desc="Plotting systems"):
        plot_system(lps, df)
        plot_system(lps_zoom, df)

    # Ajut limits for zoomed plot
    lps_zoom.ax.set_xlim(-100, 60)
    lps_zoom.ax.set_ylim(-7, 20)
    
    # Save the final plot
    plot_filename = f'lps_all_systems_fixed.png'
    plot_zoom_filename = f'lps_zoom_all_systems_fixed.png'

    plot_path = os.path.join(output_directory, plot_filename)
    plot_zoom_path = os.path.join(output_directory, plot_zoom_filename)

    lps.fig.savefig(plot_path)
    lps_zoom.fig.savefig(plot_zoom_path)

    plt.close()
    print(f"Saved plots to {plot_path}, {plot_zoom_path}")

def main():
    base_path = '../csv_fixed_framework_database_energy_by_periods'
    output_directory = '../figures_lps/test_fixed_framework/'

    os.makedirs(output_directory, exist_ok=True)

    # Read the energetics data for all systems
    systems_energetics = read_life_cycles(base_path)

    plot_all_systems_by_region_season(systems_energetics, output_directory)

    # clusters_to_use = ["ARG_DJF_cl_2", "ARG_JJA_cl_1",
    #                    "LA-PLATA_DJF_cl_2", "LA-PLATA_JJA_cl_2",
    #                    "SE-BR_DJF_cl_2", "SE-BR_JJA_cl_3"]
    
    # for cluster in clusters_to_use:
    #     plot_all_systems_by_region_season(systems_energetics, id_list_directory, cluster, output_directory)

if __name__ == "__main__":
    main()

