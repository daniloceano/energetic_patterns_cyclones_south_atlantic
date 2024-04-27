# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_lps_patterns_region_season.py                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/27 12:10:53 by daniloceano       #+#    #+#              #
#    Updated: 2024/04/27 12:46:01 by daniloceano      ###   ########.fr        #
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
from plot_lps_aux import plot_system

def plot_all_clusters(patterns_path, output_directory_all_patterns):

    # Initialize the Lorenz Phase Space plotter
    lps = Visualizer(LPS_type='mixed', zoom=True,
                     x_limits=[-40, 2],
                     y_limits=[-2, 8],
                     color_limits=[-10, 10])

    # Read the patterns

    for season in ['DJF','JJA']:
        for region in ['SE-BR', 'LA-PLATA', 'ARG']:
            patterns_clusters_paths = glob(f"{patterns_path}/{region}_{season}/*/*")

            # Plot each pattern onto the Lorenz Phase Space diagram
            for cluster_file in patterns_clusters_paths:
                df = pd.read_csv(cluster_file, index_col=0)
                plot_system(lps, df)
            
    # Save the final plot
    plot_filename = f'lps_all_clusters_region_season.png'
    plot_path = os.path.join(output_directory_all_patterns, plot_filename)

    plt.savefig(plot_path)
    plt.close()
    print(f"Final plot saved to {plot_path}")

def main():
    patterns_path = "../csv_patterns/"
    output_directory_all_patterns = '../figures_lps/clusters_region_season/'
    output_directory_each_pattern = '../figures_lps/clusters_each_pattern_region_season/'
    os.makedirs(output_directory_all_patterns, exist_ok=True)
    os.makedirs(output_directory_each_pattern, exist_ok=True)

    plot_all_clusters(patterns_path, output_directory_all_patterns)


if __name__ == "__main__":
    main()