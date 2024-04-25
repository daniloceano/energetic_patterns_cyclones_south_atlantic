# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_lps_all_systems.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/25 09:39:10 by daniloceano       #+#    #+#              #
#    Updated: 2024/04/25 09:45:42 by daniloceano      ###   ########.fr        #
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

def plot_all_systems(systems_energetics, output_directory):

    # Initialize the Lorenz Phase Space plotter
    lps = Visualizer(LPS_type='mixed', zoom=False)

    # Plot each system onto the Lorenz Phase Space diagram
    for system_id, df in tqdm(systems_energetics.items(), desc="Plotting systems"):
        plot_system(lps, df)
    
    # Save the final plot
    plot_filename = 'lps_all_systems.png'
    plot_path = os.path.join(output_directory, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Final plot saved to {plot_path}")

    # Determine global limits
    x_limits, y_limits, color_limits, marker_limits = determine_global_limits(systems_energetics)

    # Initialize Lorenz Phase Space with dynamic limits and zoom enabled
    lps = Visualizer(
        LPS_type='mixed',
        zoom=True,
        x_limits=x_limits,
        y_limits=y_limits,
        color_limits=color_limits,
        marker_limits=marker_limits
    )

    # Plot each system onto the Lorenz Phase Space diagram
    for system_id, df in tqdm(systems_energetics.items(), desc="Plotting systems"):
        lps.plot_data(
            x_axis=df['Ck'],
            y_axis=df['Ca'],
            marker_color=df['Ge'],
            marker_size=df['Ke']
        )

    # Save the final plot
    plot_filename = 'lps_all_systems_zoom.png'
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
    plot_all_systems(systems_energetics, output_directory)

if __name__ == "__main__":
    main()