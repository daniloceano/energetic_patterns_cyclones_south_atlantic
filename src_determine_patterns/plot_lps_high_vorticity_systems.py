# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_lps_high_vorticity_systems.py                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/07/05 11:00:00 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/05 15:26:56 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Plot the LPS for all clusters, which were defined for high vorticity systems.
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
    """Plot LPS for all systems and clusters in one figure."""
    lps = Visualizer(LPS_type=lps_type, zoom=False)
    
    for df in patterns_energetics:
        plot_system(lps, df, lps_type)
    
    save_plot(results_path, f'lps_{lps_type}_all_systems_all_clusters.png')
    
    x_limits, y_limits, color_limits, marker_limits = determine_global_limits(patterns_energetics, lps_type)
    
    lps = Visualizer(
        LPS_type=lps_type, zoom=True,
        x_limits=x_limits, y_limits=y_limits,
        color_limits=color_limits, marker_limits=marker_limits
    )
    
    for df in patterns_energetics:
        plot_data_on_lps(lps, df, lps_type)
    
    save_plot(results_path, f'lps_{lps_type}_all_systems_all_clusters_zoom.png')


def plot_clusters_each_pattern(results_path, phases, lps_type, patterns_energetics, clusters_center, results):
    """Plot LPS for each cluster pattern."""
    x_limits, y_limits, color_limits, marker_limits = determine_global_limits(patterns_energetics, lps_type)
    
    for i in range(len(clusters_center)):
        cluster_center = np.array(clusters_center.iloc[i]).reshape(len(TERMS), len(phases))
        explained_variance = results.loc['Cluster Fraction'].iloc[i]
        cluster_number = clusters_center.index[i][-1]
        title = f"Cluster {cluster_number} - Explained Variance: {explained_variance:.2f}"
        
        df = pd.DataFrame(cluster_center, columns=phases, index=TERMS).T
        lps = Visualizer(
            LPS_type=lps_type, zoom=True,
            x_limits=x_limits, y_limits=y_limits,
            color_limits=color_limits, marker_limits=marker_limits
        )
        
        plot_data_on_lps(lps, df, lps_type, title)
        save_plot(results_path, f'lps_{lps_type}_all_systems_cl_{cluster_number}_zoom.png')


def plot_systems_for_each_cluster(results_path, lps_type, systems_energetics, clusters_center, results):
    """Plot LPS for each system in each cluster."""
    cluster_cyclones = get_cyclone_ids_by_cluster(results_path)
    x_limits, y_limits, color_limits, marker_limits = determine_global_limits(systems_energetics, lps_type)
    
    for cluster, cyclone_ids in cluster_cyclones.items():
        cluster_number = cluster.split()[-1]
        lps = Visualizer(
            LPS_type=lps_type, zoom=True,
            x_limits=x_limits, y_limits=y_limits,
            color_limits=color_limits, marker_limits=marker_limits
        )
        
        for cyclone_id in cyclone_ids:
            df = systems_energetics.get(str(cyclone_id), pd.DataFrame())
            if not df.empty:
                plot_system(lps, df, lps_type)
        
        explained_variance = results.loc['Cluster Fraction'][cluster]
        title = f"Cluster {cluster_number} - Explained Variance: {explained_variance:.2f}"
        save_plot(results_path, f'lps_{lps_type}_cluster_{cluster_number}_all_systems.png', title)


def save_plot(results_path, filename, title=None):
    """Save the plot with the given filename."""
    plot_path = os.path.join(results_path, filename)
    if title:
        plt.title(title, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(plt.gca().get_xlabel(), fontsize=16)
    plt.ylabel(plt.gca().get_ylabel(), fontsize=16)
    if plt.gca().get_legend() is not None:
        plt.setp(plt.gca().get_legend().get_texts(), fontsize=14)
    if plt.gca().collections:  # Check if a color bar exists
        colorbar = plt.gca().collections[-1].colorbar
        if colorbar:
            colorbar.ax.tick_params(labelsize=14)  # Update color bar tick labels
            colorbar.set_label(colorbar.ax.get_ylabel(), fontsize=16)
    plt.savefig(plot_path)
    plt.close()
    print(f"Final plot saved to {plot_path}")


def plot_data_on_lps(lps, df, lps_type, title=None):
    """Plot data on Lorenz Phase Space."""
    if lps_type == 'mixed':
        x_axis, y_axis = df['Ck'], df['Ca']
    elif lps_type == 'imports':
        x_axis, y_axis = df['BAe'], df['BKe']
    
    lps.plot_data(
        x_axis=x_axis, y_axis=y_axis,
        marker_color=df['Ge'], marker_size=df['Ke']
    )
    if title:
        plt.title(title, fontsize=14)

def main():
    results_path = "../results_kmeans/high_vorticity_systems"
    base_path = '../csv_database_energy_by_periods'
    
    results_path_life_cycles = sorted(glob(f'{results_path}/*'))
    systems_energetics = read_life_cycles(base_path)
    
    for results_path_life_cycle in results_path_life_cycles:
        phases = reconstruct_phases_from_path(results_path_life_cycle)
        patterns_energetics, clusters_center, results = read_patterns(results_path_life_cycle, phases, TERMS)
        
        for lps_type in ['mixed', 'imports']:
            plot_all_systems_all_clusters_one_figure(results_path_life_cycle, lps_type, patterns_energetics)
            plot_clusters_each_pattern(results_path_life_cycle, phases, lps_type, patterns_energetics, clusters_center, results)
            plot_systems_for_each_cluster(results_path_life_cycle, lps_type, systems_energetics, clusters_center, results)


if __name__ == "__main__":
    main()
