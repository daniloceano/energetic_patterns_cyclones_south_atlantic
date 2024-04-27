# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    kmeans_all_systems.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/27 10:57:02 by daniloceano       #+#    #+#              #
#    Updated: 2024/04/27 10:57:14 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
import glob
from tqdm import tqdm  
from sklearn.cluster import KMeans
from kmeans_aux import prepare_to_kmeans, slice_mk, sel_clusters_to_df


"""
This script is used to perform K-means clustering on all systems in the database that have
the lifecycle of 'incipient', 'intensification', 'mature', 'decay', as it is the most common
in the database.
"""

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
        columns_to_read = ['Ck', 'Ca', 'Ke', 'Ge']
        df_system = pd.read_csv(case, header=0, index_col=0)
        df_system = df_system[columns_to_read]
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
CSVSAVEPATH = '../csv_patterns/all_systems/'
LIFECYCLE = {'incipient', 'intensification', 'mature', 'decay'}

def main():
    results_energetics_all_systems = get_energetics_all_systems(ENERGETICSPATH)
    results_energetics_lifecycle = filter_lifecycle(results_energetics_all_systems)
    dsk_means = prepare_to_kmeans(results_energetics_lifecycle)
    mk = KMeans(n_clusters=4,n_init=10).fit(dsk_means)
    centers_Ck, centers_Ca, centers_Ke, centers_Ge = slice_mk(mk, LIFECYCLE)
    df_cl1, df_cl2, df_cl3, df_cl4 = sel_clusters_to_df(centers_Ck, centers_Ca, centers_Ke, centers_Ge, results_energetics_lifecycle)

    os.makedirs(CSVSAVEPATH, exist_ok=True)
    pattern_folder = os.path.join(CSVSAVEPATH, "IcItMD")
    os.makedirs(pattern_folder, exist_ok=True)

    for df, name in zip([df_cl1, df_cl2, df_cl3, df_cl4], ['df_cl1', 'df_cl2', 'df_cl3', 'df_cl4']):
        df.to_csv(os.path.join(pattern_folder, f'{name}.csv'))
        print(f"Saved {name} to {os.path.join(pattern_folder, f'{name}.csv')}")

if __name__ == '__main__':
    main()