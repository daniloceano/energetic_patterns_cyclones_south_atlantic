# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_lps_all_systems_by_region_season.py           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/25 23:38:24 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/01 17:11:52 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Plot the LPS for all systems for a specific region and season.
"""

import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from lorenz_phase_space.phase_diagrams import Visualizer
from plot_lps_aux import read_life_cycles, plot_system, determine_global_limits


def plot_all_systems_by_region_season(systems_energetics, id_list_directory, season, region, lps_type, output_directory):

    # Initialize the Lorenz Phase Space plotter
    lps = Visualizer(LPS_type=lps_type, zoom=False)

    ids_file = glob(f"{id_list_directory}/*{region}*{season}.csv")
    print(f"IDs file: {ids_file}")
    track_ids = pd.read_csv(ids_file[0])

    # Plot each system onto the Lorenz Phase Space diagram
    for system_id, df in tqdm(systems_energetics.items(), desc="Plotting systems"):
        system_id = int(system_id)
        if system_id in track_ids.values:
            plot_system(lps, df, lps_type)
    
    # Save the final plot
    plot_filename = f'lps_{lps_type}_all_systems_{region}_{season}.png'
    plot_path = os.path.join(output_directory, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Final plot saved to {plot_path}")

def main():
    base_path = '../csv_database_energy_by_periods'
    output_directory = '../figures_lps/all_systems/'
    os.makedirs(output_directory, exist_ok=True)

    id_list_directory = '../csv_track_ids_by_region_season'

    # Read the energetics data for all systems
    systems_energetics = read_life_cycles(base_path)

    for lps_type in ['mixed', 'imports']:
        for season in ['DJF', 'MAM', 'JJA', 'SON']:
            for region in ['SE-BR', 'LA-PLATA', 'ARG']:
                plot_all_systems_by_region_season(systems_energetics, id_list_directory, season, region, lps_type, output_directory)

if __name__ == "__main__":
    main()