# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    kmeans_high_vorticity.py                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/07/04 14:10:29 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/05 15:21:49 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import glob
import json
import pandas as pd
import numpy as np
from tqdm import tqdm  
from kmeans_aux import preprocess_energy_data, calculate_sse_for_clusters, plot_elbow_method, kmeans_energy_data, assign_cyclones_to_clusters 

"""
This script performs K-means clustering on systems with vorticity above the 0.9 quantile.
"""

ENERGETICSPATH = '../csv_database_energy_by_periods/'
RESULTSPATH = '../results_kmeans/'
TRACKSPATH = '../tracks_SAt_filtered/tracks_SAt_filtered_with_periods.csv'

LIFECYCLE_CONFIGS = {
    'IcItMD': {'incipient', 'intensification', 'mature', 'decay'},
    'ItMD': {'intensification', 'mature', 'decay'},
    'IcItMDIt2M2D2': {'incipient', 'intensification', 'mature', 'decay', 'intensification 2', 'mature 2', 'decay 2'},
    'ItMDIt2M2D2': {'intensification', 'mature', 'decay', 'intensification 2', 'mature 2', 'decay 2'},
    'IcDItMD2': {'incipient', 'decay', 'intensification', 'mature', 'decay 2'},
    'DItMD2': {'decay', 'intensification', 'mature', 'decay 2'}
}

label_mapping = {
    'incipient': 'Ic',
    'incipient 2': 'Ic2',
    'intensification': 'It',
    'intensification 2': 'It2',
    'mature': 'M',
    'mature 2': 'M2',
    'decay': 'D',
    'decay 2': 'D2',
}

def get_energetics_all_systems(path):
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
        The CSV files are expected to have the following columns: 'Ck', 'Ca', 'Ke', 'Ge', 'BKe', 'BAe'.

    """
    all_files = []
    files = glob.glob(os.path.join(ENERGETICSPATH, "*.csv"))
    all_files.extend(files)
    # Creating a list to save all dataframes
    results_energetics_all_systems = []

    # Reading all files and saving in a list of dataframes with a progress bar
    for case in tqdm(all_files, desc="Reading files"):
        system_id = int((os.path.basename(case)).split('_')[0])
        columns_to_read = ['Ck', 'Ca', 'Ke', 'Ge', 'BKe', 'BAe']
        df_system = pd.read_csv(case, header=0, index_col=0)
        df_system = df_system[columns_to_read]
        df_system.index.name = system_id
        results_energetics_all_systems.append(df_system)
    
    return results_energetics_all_systems

def filter_lifecycle(results_energetics_all_systems, lifecycle_set):
    """
    Filters the given list of pandas DataFrames based on the lifecycle of the data.

    Parameters:
        results_energetics_all_systems (list): A list of pandas DataFrames, each containing energetic data for a specific system.
        lifecycle_set (set): A set of lifecycle phases to filter by.

    Returns:
        list: A filtered list of pandas DataFrames, each containing energetic data for a system with the specified lifecycle phases.

    Description:
        This function iterates over the given list of pandas DataFrames and filters them based on the lifecycle of the data. 
        It checks if the set of phases in each DataFrame's index (excluding NaN values) is equal to the lifecycle set specified.
        If a DataFrame's lifecycle matches the lifecycle set, it is appended to the results_energetics_lifecycle list. The filtered list of DataFrames is then returned.

    """
    results_energetics_lifecycle = []
    for df in tqdm(results_energetics_all_systems, desc="Filtering lifecycles"):
        phases = set(df.index.dropna())
        if phases == lifecycle_set:
            results_energetics_lifecycle.append(df)
    return results_energetics_lifecycle

def create_label(lifecycle_set, label_mapping):
    """
    Creates a label for the lifecycle configuration based on the given label mapping.

    Parameters:
        lifecycle_set (set): A set of lifecycle phases.
        label_mapping (dict): A dictionary mapping lifecycle phases to their labels.

    Returns:
        str: A string representing the label for the lifecycle configuration.

    Description:
        This function creates a label for the lifecycle configuration by mapping each phase in the lifecycle set
        to its corresponding label in the label mapping. The labels are concatenated in the order specified by the lifecycle set.

    """
    return ''.join([label_mapping[phase] for phase in lifecycle_set])

def get_high_vorticity_systems(track_path, quantile=0.9):
    """
    Retrieves the system IDs for systems with vorticity above the specified quantile.

    Parameters:
        track_path (str): The path to the CSV file containing vorticity data.
        quantile (float): The quantile threshold for vorticity.

    Returns:
        set: A set of system IDs with vorticity above the specified quantile.

    Description:
        This function reads the vorticity data from the specified CSV file and calculates the quantile threshold.
        It then identifies the system IDs for systems with vorticity above this threshold and returns them as a set.

    """
    df_tracks = pd.read_csv(track_path)
    vor_threshold = df_tracks['vor42'].quantile(quantile)
    high_vorticity_ids = df_tracks[df_tracks['vor42'] > vor_threshold]['track_id'].unique()
    return set(high_vorticity_ids)

def main():
    # Setting up directories
    results_path = os.path.join(RESULTSPATH, "high_vorticity_systems")
    os.makedirs(RESULTSPATH, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    # Loading data
    results_energetics_all_systems = get_energetics_all_systems(ENERGETICSPATH)

    # Get high vorticity systems
    high_vorticity_systems = get_high_vorticity_systems(TRACKSPATH)

    # Filter systems based on high vorticity
    results_energetics_high_vorticity = [df for df in results_energetics_all_systems if df.index.name in high_vorticity_systems]

    # Iterate through each lifecycle configuration
    for config_label, lifecycle_set in LIFECYCLE_CONFIGS.items():
        results_energetics_lifecycle = filter_lifecycle(results_energetics_high_vorticity, lifecycle_set)

        if not results_energetics_lifecycle:
            print(f"No data for lifecycle configuration {config_label}")
            continue

        # Create directory for results for this lifecycle configuration
        pattern_folder = os.path.join(results_path, config_label)
        os.makedirs(pattern_folder, exist_ok=True)

        # Determine the number of clusters
        clmax = 10  # max number of clusters to test
        features = preprocess_energy_data(results_energetics_lifecycle)
        sseclusters = calculate_sse_for_clusters(clmax, features)
        ncenters = plot_elbow_method(clmax, sseclusters, pattern_folder)

        # Perform clustering
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
        json_path = os.path.join(pattern_folder, f'kmeans_results_{config_label}.json')
        with open(json_path, 'w') as json_file:
            json.dump(results_json, json_file, indent=4)

        # Save explanation to a README file
        readme_text = f"""
            This JSON file contains the K-means clustering results for lifecycle configuration {config_label} in SE-BR, LA-PLATA and ARG regions.
            It includes the cluster centers for each cluster, along with the cluster fraction and the IDs of the cyclones in each cluster.
            Each cluster center array consists of 16 values.
            These values represent the average scaled measurements of the energy terms (Ck, Ca, Ke, Ge, BAe, BKe) across the lifecycle phases.
            The first 4 values correspond to Ck for each phase, followed by 4 values for Ca, Ke, Ge, BAe, and BKe respectively.
            """
        readme_path = os.path.join(pattern_folder, f'README_{config_label}.txt')
        with open(readme_path, 'w') as readme_file:
            readme_file.write(readme_text)

        print(f"Results saved to {json_path}")
        print(f"README saved to {readme_path}")

if __name__ == '__main__':
    main()
