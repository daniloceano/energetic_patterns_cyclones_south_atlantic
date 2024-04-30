# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    kmeans_by_season_region.py                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/27 10:56:55 by daniloceano       #+#    #+#              #
#    Updated: 2024/04/29 17:34:58 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
import glob
import json
from tqdm import tqdm  
from kmeans_aux import preprocess_energy_data, calculate_sse_for_clusters, plot_elbow_method, kmeans_energy_data, assign_cyclones_to_clusters 

"""
This script is used to perform K-means clustering on all systems in the database that have
the lifecycle of 'incipient', 'intensification', 'mature', 'decay', as it is the most common
in the database.
"""

def get_energetics_region_season(energetics_path, id_list_directory, region, season):
    """
    Retrieves energetic data for all systems.

    Parameters:
        path (str): The path to the directory containing the CSV files with energetic data.

    Returns:
        list: A list of pandas DataFrames, each containing energetic data for a specific system.

    Description:
        This function retrieves energetic data for all systems by reading CSV files from the specified directory. 
        It creates a list of all CSV files in the directory and then iterates over each file, reading the data 
        into a pandas DataFrame and selecting the desired columns. The DataFrames are then appended to a list. 
        Finally, the list of DataFrames is returned.

    Note:
        The CSV files are expected to have the following columns: 'Ck', 'Ca', 'Ke', 'Ge'.

    """
    all_files = []
    files = glob.glob(os.path.join(energetics_path, "*.csv"))
    all_files.extend(files)
    # Creating a list to save all dataframes
    results_energetics_all_systems = []

    ids_file = glob.glob(f"{id_list_directory}/*{region}*{season}.csv")
    print(f"IDs file: {ids_file}")
    track_ids = pd.read_csv(ids_file[0])

    # Reading all files and saving in a list of dataframes with a progress bar
    for case in tqdm(all_files, desc="Reading files"):
        track_id = int(os.path.basename(case).split("_")[0])
        if track_id in track_ids["track_id"].values:
            columns_to_read = ['Ck', 'Ca', 'Ke', 'Ge']
            df_system = pd.read_csv(case, header=0, index_col=0)
            df_system = df_system[columns_to_read]
            df_system.index.name = track_id
            results_energetics_all_systems.append(df_system)
    
    return results_energetics_all_systems

def filter_lifecycle(results_energetics_all_systems):
    """
    Filters the given list of pandas DataFrames based on the lifecycle of the data.

    Parameters:
        results_energetics_all_systems (list): A list of pandas DataFrames, each containing energetic data for a specific system.

    Returns:
        list: A filtered list of pandas DataFrames, each containing energetic data for a system with the lifecycle of 'incipient', 'intensification', 'mature', 'decay'.

    Description:
        This function iterates over the given list of pandas DataFrames and filters them based on the lifecycle of the data. 
        It checks if the set of phases in each DataFrame's index (excluding NaN values) is equal to the LIFECYCLE set defined in the code.
        If a DataFrame's lifecycle matches the LIFECYCLE set, it is appended to the results_energetics_lifecycle list. The filtered list of DataFrames is then returned.

    """
    results_energetics_lifecycle = []
    for df in tqdm(results_energetics_all_systems, desc="Filtering lifecycles"):
        phases = set(df.index.dropna())
        if phases == LIFECYCLE:
            results_energetics_lifecycle.append(df)
    return results_energetics_lifecycle

ENERGETICSPATH = '../csv_database_energy_by_periods/'
IDSPATH = '../csv_track_ids_by_region_season/'
LIFECYCLE = {'incipient', 'intensification', 'mature', 'decay'}

def main():
    for region in ['SE-BR', 'LA-PLATA', 'ARG']:
        for season in ['DJF', 'JJA']:
            results_energetics_all_systems = get_energetics_region_season(ENERGETICSPATH, IDSPATH, region, season)
            results_energetics_lifecycle = filter_lifecycle(results_energetics_all_systems)

            # Setting up directories
            csv_path_region_season = f'../results_kmeans/{region}_{season}/'
            pattern_folder = os.path.join(csv_path_region_season, "IcItMD")
            os.makedirs(csv_path_region_season, exist_ok=True)
            os.makedirs(pattern_folder, exist_ok=True)

            # Determining the number of clusters
            clmax = 10  # max number of clusters to test
            scaled_features = preprocess_energy_data(results_energetics_lifecycle)
            sseclusters = calculate_sse_for_clusters(clmax, scaled_features)
            ncenters = plot_elbow_method(clmax, sseclusters, pattern_folder)

            # Performing clustering
            centers, cluster_fractions, kmeans_model = kmeans_energy_data(ncenters, 'auto', 300, results_energetics_lifecycle, 'lloyd', scaler_type='none', joint_scaling=True)

            # Retrieve track IDs corresponding to each cluster
            cyclone_ids = [df.index.name for df in results_energetics_lifecycle]
            cluster_ids = assign_cyclones_to_clusters(kmeans_model, cyclone_ids, ncenters)

            # Prepare data to save as JSON, organizing by cluster
            results_json = {}
            for i in range(ncenters):
                cluster_label = f"Cluster {i + 1}"
                results_json[cluster_label] = {
                    "Cluster Center": centers[i].tolist(),  # Convert numpy array to list for JSON serialization
                    "Cluster Fraction": float(cluster_fractions[i] * 100),
                    "Cyclone IDs": cluster_ids[cluster_label]
                }

            # Save results to a JSON file
            json_path = os.path.join(pattern_folder, 'kmeans_results.json')
            with open(json_path, 'w') as json_file:
                json.dump(results_json, json_file, indent=4)

if __name__ == '__main__':
    main()