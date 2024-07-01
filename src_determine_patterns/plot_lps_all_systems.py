# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_lps_all_systems.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/25 09:39:10 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/01 17:26:46 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Plot the LPS for all systems.
"""

import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from lorenz_phase_space.phase_diagrams import Visualizer
from plot_lps_aux import read_life_cycles, plot_system, determine_global_limits

def plot_all_systems(systems_energetics, lps_type, output_directory):

    # Initialize the Lorenz Phase Space plotter
    lps = Visualizer(LPS_type=lps_type, zoom=False)

    # Plot each system onto the Lorenz Phase Space diagram
    for system_id, df in tqdm(systems_energetics.items(), desc="Plotting systems"):
        plot_system(lps, df, lps_type)
    
    # Save the final plot
    plot_filename = f'lps_{lps_type}_all_systems.png'
    plot_path = os.path.join(output_directory, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Final plot saved to {plot_path}")

    # Determine global limits
    x_limits, y_limits, color_limits, marker_limits = determine_global_limits(systems_energetics, lps_type)

    # Initialize Lorenz Phase Space with dynamic limits and zoom enabled
    lps = Visualizer(
        LPS_type=lps_type,
        zoom=True,
        x_limits=x_limits,
        y_limits=y_limits,
        color_limits=color_limits,
        marker_limits=marker_limits
    )

    # Plot each system onto the Lorenz Phase Space diagram
    for system_id, df in tqdm(systems_energetics.items(), desc="Plotting systems"):
        # Set terms to plot based on LPS type
        if lps_type == 'mixed':
            x_axis = list(df['Ck'])
            y_axis = list(df['Ca'])
        elif lps_type == 'imports':
            x_axis = list(df['BAe'])
            y_axis = list(df['BKe'])
        
        marker_color = list(df['Ge'])
        marker_size = list(df['Ke'])
        
        lps.plot_data(
            x_axis=x_axis,
            y_axis=y_axis,
            marker_color=marker_color,
            marker_size=marker_size
        )

    # Save the final plot
    plot_filename = f'lps_{lps_type}_all_systems_zoom.png'
    plot_path = os.path.join(output_directory, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Final plot saved to {plot_path}")

def main():
    base_path = '../csv_database_energy_by_periods'
    output_directory = '../figures_lps/all_systems/'
    os.makedirs(output_directory, exist_ok=True)

    # Read the energetics data for all systems
    systems_energetics = read_life_cycles(base_path)
    for lps_type in ['mixed', 'imports']:
        plot_all_systems(systems_energetics, lps_type, output_directory)

if __name__ == "__main__":
    main()