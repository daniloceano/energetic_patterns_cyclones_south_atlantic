# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_lps_kmeans_all_systems.py                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/29 18:23:09 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/04 14:39:41 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Plot the LPS for all clusters, which were defined for all systems.
"""

import os
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from lorenz_phase_space.phase_diagrams import Visualizer
from plot_lps_aux import plot_system, determine_global_limits, read_patterns, get_cyclone_ids_by_cluster, read_life_cycles, reconstruct_phases_from_path

TERMS = ['Ck', 'Ca', 'Ke', 'Ge', 'BAe', 'BKe']

def plot_all_systems_all_clusters_one_figure(results_path, lps_type, patterns_energetics):
    # Initialize the Lorenz Phase Space plotter
    lps = Visualizer(LPS_type=lps_type, zoom=False)

    # Plot each system onto the Lorenz Phase Space diagram
    for df in patterns_energetics:
        plot_system(lps, df, lps_type)

    # Save the final plot
    plot_filename = f'lps_{lps_type}_all_systems_all_clusters.png'
    plot_path = os.path.join(results_path, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Final plot saved to {plot_path}")

    # Determine global limits
    x_limits, y_limits, color_limits, marker_limits = determine_global_limits(patterns_energetics, lps_type)

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
    for df in patterns_energetics:
        if lps_type == 'mixed':
            x_axis = df['Ck']
            y_axis = df['Ca']
        elif lps_type == 'imports':
            x_axis = df['BAe']
            y_axis = df['BKe']
        
        marker_color = df['Ge']
        marker_size = df['Ke']

        lps.plot_data(
            x_axis=x_axis,
            y_axis=y_axis,
            marker_color=marker_color,
            marker_size=marker_size
        )

    # Save the final plot
    plot_filename = f'lps_{lps_type}_all_systems_all_clusters_zoom.png'
    plot_path = os.path.join(results_path, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Final plot saved to {plot_path}")

def plot_clusters_each_pattern(results_path, phases, lps_type, patterns_energetics, clusters_center, results):

    # Determine global limits
    x_limits, y_limits, color_limits, marker_limits = determine_global_limits(patterns_energetics, lps_type)

    # Plot each system onto the Lorenz Phase Space diagram
    for i in range(len(clusters_center)):
        # Read the energetics data for patterns
        cluster_center = np.array(clusters_center.iloc[i])
        cluster_array = np.array(cluster_center).reshape(len(TERMS), len(phases))

        # Settings for the plot
        explained_variance = results.loc['Cluster Fraction'].iloc[i]
        cluster_number = clusters_center.index[i][-1]
        cluster_name = f"cl_{cluster_number}"
        title = f"Cluster {cluster_number} - Explained Variance: {explained_variance:.2f}"

        df = pd.DataFrame(cluster_array, columns=phases, index=TERMS).T

        # Initialize Lorenz Phase Space with dynamic limits and zoom enabled
        lps = Visualizer(
            LPS_type=lps_type,
            zoom=True,
            x_limits=x_limits,
            y_limits=y_limits,
            color_limits=color_limits,
            marker_limits=marker_limits
        )

        if lps_type == 'mixed':
            x_axis = df['Ck']
            y_axis = df['Ca']
        elif lps_type == 'imports':
            x_axis = df['BAe']
            y_axis = df['BKe']

        # Plot each system onto the Lorenz Phase Space diagram
        lps.plot_data(
            x_axis=x_axis,
            y_axis=y_axis,
            marker_color=df['Ge'],
            marker_size=df['Ke']
        )

        # Add title
        plt.title(title)

        # Save plot for each cluster
        plot_filename = f'lps_{lps_type}_all_systems_{cluster_name}_zoom.png'
        plot_path = os.path.join(results_path, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Final plot saved to {plot_path}")

def plot_systems_for_each_cluster(results_path, lps_type, systems_energetics, clusters_center, results):
    cluster_cyclones = get_cyclone_ids_by_cluster(results_path)

    x_limits, y_limits, color_limits, marker_limits = determine_global_limits(systems_energetics, lps_type)

    for cluster, cyclone_ids in cluster_cyclones.items():
        cluster_number = cluster.split()[-1]
        lps = Visualizer(
            LPS_type=lps_type,
            zoom=True,
            x_limits=x_limits,
            y_limits=y_limits,
            color_limits=color_limits,
            marker_limits=marker_limits
        )
        
        for cyclone_id in cyclone_ids:
            df = systems_energetics[str(cyclone_id)]
            if not df.empty:
                plot_system(lps, df, lps_type)

        # Settings for the title
        explained_variance = results.loc['Cluster Fraction'][cluster]
        cluster_name = f"cl_{cluster_number}"
        title = f"Cluster {cluster_number} - Explained Variance: {explained_variance:.2f}"
        plt.title(title)

        plot_filename = f'lps_{lps_type}_cluster_{cluster_number}_all_systems.png'
        plot_path = os.path.join(results_path, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Final plot saved to {plot_path}")

def main():
    results_path = "../results_kmeans/all_systems"
    base_path = '../csv_database_energy_by_periods'

    # Get all folders in the results path
    results_path_life_cycles = sorted(glob(f'{results_path}/*'))

    # Read the energetics data for all systems
    systems_energetics = read_life_cycles(base_path)

    # Loop for each life cycle
    for results_path_life_cycle in results_path_life_cycles:
        # Reconstruct the list of phases
        phases = reconstruct_phases_from_path(results_path_life_cycle)
        # Read the energetics data for patterns
        patterns_energetics, clusters_center, results = read_patterns(results_path_life_cycle, phases, TERMS)
        for lps_type in ['mixed', 'imports']:
            plot_all_systems_all_clusters_one_figure(results_path_life_cycle, lps_type, patterns_energetics)
            plot_clusters_each_pattern(results_path_life_cycle, phases, lps_type, patterns_energetics, clusters_center, results )
            plot_systems_for_each_cluster(results_path_life_cycle, lps_type, systems_energetics, clusters_center, results)


if __name__ == "__main__":
    main()