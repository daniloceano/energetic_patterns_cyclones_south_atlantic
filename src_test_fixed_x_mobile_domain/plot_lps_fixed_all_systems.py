# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_lps_fixed_all_systems.py                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/05 18:56:28 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/06 09:57:59 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from lorenz_phase_space.phase_diagrams import Visualizer

def plot_all_systems_by_region_season(systems_energetics, id_list_directory, season, region, output_directory):

    # Initialize the Lorenz Phase Space plotter
    lps = Visualizer(LPS_type='mixed', zoom=False)

    ids_file = glob(f"{id_list_directory}/*{region}*{season}.csv")
    print(f"IDs file: {ids_file}")
    track_ids = pd.read_csv(ids_file[0])

    # Plot each system onto the Lorenz Phase Space diagram
    for system_id, df in tqdm(systems_energetics.items(), desc="Plotting systems"):
        system_id = int(system_id)
        if system_id in track_ids.values:
            plot_system(lps, df)
    
    # Save the final plot
    plot_filename = f'lps_all_systems_{region}_{season}.png'
    plot_path = os.path.join(output_directory, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Final plot saved to {plot_path}")

def main():
    base_path = '../csv_database_energy_by_periods'
    output_directory = '../figures_lps/test_fixed_framework/all_systems/'

    tracks_filtered = pd.read_csv('../tracks_SAt_filtered/tracks_SAt_filtered.csv')
    tracks_filtered_with_periods = pd.read_csv('../tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv')

    os.makedirs(output_directory, exist_ok=True)

    id_list_directory = '../csv_track_ids_by_region_season'

    # Read the energetics data for all systems
    systems_energetics = read_life_cycles(base_path)

    clusters_to_use = ["ARG_DJF_cl_2", "ARG_JJA_cl_1",
                       "LA-PLATA_DJF_cl_2", "LA-PLATA_JJA_cl_2",
                       "SE-BR_DJF_cl_2", "SE-BR_JJA_cl_3"]
    
    for cluster in clusters_to_use:
        plot_all_systems_by_region_season(systems_energetics, id_list_directory, cluster, output_directory)

if __name__ == "__main__":
    main()

