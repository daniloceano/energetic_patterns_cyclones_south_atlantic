# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_lps_kmeans_region_season.py                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/27 12:10:53 by daniloceano       #+#    #+#              #
#    Updated: 2024/04/30 09:12:00 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


"""
Plot the LPS for all clusters, which were separated by regions and seasons.
"""

import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from lorenz_phase_space.phase_diagrams import Visualizer
from plot_lps_aux import plot_system, read_patterns

PHASES = ['incipient', 'intensification', 'mature', 'decay']
TERMS = ['Ck', 'Ca', 'Ke', 'Ge']

def plot_all_clusters(results_path):
    
    # Initialize the Lorenz Phase Space plotter
    lps = Visualizer(LPS_type='mixed', zoom=True,
                     x_limits=[-40, 2],
                     y_limits=[-2, 8],
                     color_limits=[-10, 10])

    for season in ['DJF','JJA']:
        for region in ['SE-BR', 'LA-PLATA', 'ARG']:
            # Read the energetics data for patterns
            patterns_clusters_paths = f"{results_path}/{region}_{season}/IcItMD/"
            patterns_energetics, clusters_center, results = read_patterns(patterns_clusters_paths, PHASES, TERMS)

            # Plot each pattern onto the Lorenz Phase Space diagram
            for df in patterns_energetics:
                plot_system(lps, df)
            
    # Save the final plot
    plot_filename = f'lps_all_clusters_all_regions_seasons.png'
    plot_path = os.path.join("../figures_lps/", plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Final plot saved to {plot_path}")

def plot_all_clusters_one_figure_by_region_season(results_path):
    for season in ['DJF','JJA']:
        for region in ['SE-BR', 'LA-PLATA', 'ARG']:
            # Read the energetics data for patterns
            patterns_clusters_paths = f"{results_path}/{region}_{season}/IcItMD/"
            patterns_energetics, clusters_center, results = read_patterns(patterns_clusters_paths, PHASES, TERMS)

            # Initialize the Lorenz Phase Space plotter
            lps = Visualizer(LPS_type='mixed', zoom=True,
                    x_limits=[-40, 2],
                    y_limits=[-2, 8],
                    color_limits=[-10, 10])

            # Plot each pattern onto the Lorenz Phase Space diagram
            for df in patterns_energetics:
                plot_system(lps, df)
            
            # Save the final plot
            plot_filename = f'lps_all_clusters_{region}_{season}.png'
            plot_path = os.path.join(patterns_clusters_paths, plot_filename)
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            plt.close()
            print(f"Final plot saved to {plot_path}")

def plot_each_cluster_by_region_season(results_path):
    for season in ['DJF','JJA']:
        for region in ['SE-BR', 'LA-PLATA', 'ARG']:
            # Read the energetics data for patterns
            patterns_clusters_paths = f"{results_path}/{region}_{season}/IcItMD/"
            patterns_energetics, clusters_center, results = read_patterns(patterns_clusters_paths, PHASES, TERMS)

            # Plot each pattern onto the Lorenz Phase Space diagram
            for idx, df in enumerate(patterns_energetics):
                # Initialize the Lorenz Phase Space plotter
                lps = Visualizer(LPS_type='mixed', zoom=True,
                    x_limits=[-40, 2],
                    y_limits=[-2, 8],
                    color_limits=[-10, 10])
                
                # Plot each pattern onto the Lorenz Phase Space diagram
                plot_system(lps, df)

                # Settings for the plot
                explained_variance = results.loc['Cluster Fraction'].iloc[idx]
                cluster_number = clusters_center.index[idx][-1]
                cluster_name = f"cl_{cluster_number}"
                title = f"Cluster {cluster_number} - Explained Variance: {explained_variance:.2f}"

                # Add title
                plt.title(title)
            
                # Save the final plot
                plot_filename = f'lps_{region}_{season}_{cluster_name}.png'
                plot_path = os.path.join(patterns_clusters_paths, plot_filename)
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                plt.savefig(plot_path)
                plt.close()
                print(f"Final plot saved to {plot_path}")


def main():
    results_path = "../results_kmeans/"

    plot_all_clusters(results_path)
    plot_all_clusters_one_figure_by_region_season(results_path)
    plot_each_cluster_by_region_season(results_path)


if __name__ == "__main__":
    main()