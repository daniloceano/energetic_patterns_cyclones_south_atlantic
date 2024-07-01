# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    kmeans_all_systems.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/27 10:57:02 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/01 18:17:16 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import glob
import json
import pandas as pd
from tqdm import tqdm  
from kmeans_aux import preprocess_energy_data, calculate_sse_for_clusters, plot_elbow_method, kmeans_energy_data, assign_cyclones_to_clusters 

"""
This script is used to perform K-means clustering on all systems in the database that have
the lifecycle of 'incipient', 'intensification', 'mature', 'decay', as it is the most common
in the database.
"""

ENERGETICSPATH = '../csv_database_energy_by_periods/'
RESULTSPATH = '../results_kmeans/'
LIFECYCLE = {'incipient', 'intensification', 'mature', 'decay'}

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
        The CSV files are expected to have the following columns: 'Ck', 'Ca', 'Ke', 'Ge'.

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


def main():
    # Setting up directories
    pattern_folder = os.path.join(RESULTSPATH, "all_systems")
    os.makedirs(RESULTSPATH, exist_ok=True)
    os.makedirs(pattern_folder, exist_ok=True)

    # Loading data
    results_energetics_all_systems = get_energetics_all_systems(ENERGETICSPATH)
    results_energetics_lifecycle = filter_lifecycle(results_energetics_all_systems)

    # Determining the number of clusters
    clmax = 10  # max number of clusters to test
    features = preprocess_energy_data(results_energetics_lifecycle)
    sseclusters = calculate_sse_for_clusters(clmax, features)
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

    # Save explanation to a README file
    readme_text = """
            This JSON file contains the K-means clustering results for the all systems in SE-BR, LA-PLATA and ARG regions.
            It includes the cluster centers for each cluster, along with the cluster fraction and the IDs of the cyclones in each cluster.
            Each cluster center array consists of 16 values.
            These values represent the average scaled measurements of the energy terms (Ck, Ca, Ke, Ge, BAe, BKe) across the lifecycle phases (incipient, intensification, mature, decay).
            The first 4 values correspond to Ck for each phase, followed by 4 values for Ca, Ke, Ge, BAe, and BKe respectively.    """
    readme_path = os.path.join(pattern_folder, 'README.txt')
    with open(readme_path, 'w') as readme_file:
        readme_file.write(readme_text)

    print(f"Results saved to {json_path}")
    print(f"README saved to {readme_path}")

if __name__ == '__main__':
    main()