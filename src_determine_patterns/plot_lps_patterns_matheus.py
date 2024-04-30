# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_lps_patterns_matheus.py                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/25 08:16:45 by daniloceano       #+#    #+#              #
#    Updated: 2024/04/29 18:23:05 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from lorenz_phase_space.phase_diagrams import Visualizer
from plot_lps_aux import read_patterns, plot_system, determine_global_limits

def plot_all_patterns_all_clusters_one_figure(patterns_path, output_directory):
    # Read the energetics data for patterns
    patterns_by_life_cycle_paths = glob(f'{patterns_path}/*')

    patterns_energetics = read_patterns(patterns_by_life_cycle_paths)

    # Initialize the Lorenz Phase Space plotter
    lps = Visualizer(LPS_type='mixed', zoom=False)

    # Plot each system onto the Lorenz Phase Space diagram
    for system_id, df in tqdm(patterns_energetics.items(), desc="Plotting systems"):
        plot_system(lps, df)

    # Save the final plot
    plot_filename = 'lps_all_patterns_all_clusters.png'
    plot_path = os.path.join(output_directory, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Final plot saved to {plot_path}")

    # Determine global limits
    x_limits, y_limits, color_limits, marker_limits = determine_global_limits(patterns_energetics)

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
    for species, df in tqdm(patterns_energetics.items(), desc="Plotting systems"):
        print(f"Plotting {species}")
        lps.plot_data(
            x_axis=df['Ck'],
            y_axis=df['Ca'],
            marker_color=df['Ge'],
            marker_size=df['Ke']
        )

    # Save the final plot
    plot_filename = 'lps_all_patterns_all_clusters_zoom.png'
    plot_path = os.path.join(output_directory, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Final plot saved to {plot_path}")

def plot_clusters_one_figure(patterns_path, output_directory):
    patterns_by_life_cycle_paths = glob(f'{patterns_path}/*')

    for directory in patterns_by_life_cycle_paths:
        life_cycle_type = os.path.basename(directory)
        patterns = glob(f'{directory}/*')

        patterns_energetics = {}

        for pattern in patterns:
            df = pd.read_csv(pattern)
            cluster = os.path.basename(pattern).split('_')[1]
            patterns_energetics[cluster] = df

        # Determine global limits
        x_limits, y_limits, color_limits, marker_limits = determine_global_limits(patterns_energetics)

        # Initialize the Lorenz Phase Space plotter with dynamic limits and zoom enabled
        lps = Visualizer(
            LPS_type='mixed',
            zoom=True,
            x_limits=x_limits,
            y_limits=y_limits,
            color_limits=color_limits,
            marker_limits=marker_limits
        )

        # Plot each system onto the Lorenz Phase Space diagram
        for cluster, df in patterns_energetics.items():
            plot_system(lps, df)

        title = life_cycle_type.replace(
            "Ic", "Incipient").replace(
            "It", "Intensification").replace(
            "M", "Mature").replace(
            "D", "Decay").replace(
            "It2", "Intensification 2").replace(
            "M2", "Mature 2").replace(
            "D2", "Decay 2")
        plt.title(f"Life cycle type: {title}")

        # Save the final plot
        plot_filename = f'lps_{life_cycle_type}.png'
        plot_path = os.path.join(output_directory, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Final plot saved to {plot_path}")

def plot_clusters_each_pattern(patterns_path, output_directory):
    # Read the energetics data for patterns
    patterns_by_life_cycle_paths = glob(f'{patterns_path}/*')

    for directory in patterns_by_life_cycle_paths:
        life_cycle_type = os.path.basename(directory)
        patterns = glob(f'{directory}/*')

        patterns_energetics = {}

        for pattern in patterns:
            df = pd.read_csv(pattern)
            cluster = os.path.basename(pattern).split('_')[1]
            patterns_energetics[cluster] = df

            # Determine global limits
            x_limits, y_limits, color_limits, marker_limits = determine_global_limits(patterns_energetics)

            # Plot each system onto the Lorenz Phase Space diagram
            for cluster, df in patterns_energetics.items():

                # Initialize the Lorenz Phase Space plotter with dynamic limits and zoom enabled
                lps = Visualizer(
                    LPS_type='mixed',
                    zoom=True,
                    x_limits=x_limits,
                    y_limits=y_limits,
                    color_limits=color_limits,
                    marker_limits=marker_limits
                )
                plot_system(lps, df)

                title = life_cycle_type.replace(
                    "Ic", "Incipient").replace(
                    "It", "Intensification").replace(
                    "M", "Mature").replace(
                    "D", "Decay").replace(
                    "It2", "Intensification 2").replace(
                    "M2", "Mature 2").replace(
                    "D2", "Decay 2")
                plt.title(f"Life cycle type: {title}")

                # Save the final plot
                plot_filename = f'lps_{life_cycle_type}_{cluster}.png'
                plot_path = os.path.join(output_directory, plot_filename)
                plt.savefig(plot_path)
                plt.close()
                print(f"Final plot saved to {plot_path}")

def main():
    patterns_path = "../csv_patterns/"
    output_directory_all_patterns = '../figures_lps/clusters_all_systems/'
    output_directory_each_pattern = '../figures_lps/clusters_all_systems_each_pattern/'
    os.makedirs(output_directory_all_patterns, exist_ok=True)
    os.makedirs(output_directory_each_pattern, exist_ok=True)

    plot_all_patterns_all_clusters_one_figure(patterns_path, output_directory_all_patterns)
    plot_clusters_one_figure(patterns_path, output_directory_all_patterns)
    plot_clusters_each_pattern(patterns_path, output_directory_each_pattern)


if __name__ == "__main__":
    main()