import os
import pandas as pd
import json
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from lorenz_phase_space.phase_diagrams import Visualizer

def read_life_cycles(base_path):
    """
    Reads all CSV files in the specified directory and collects DataFrame for each system.
    """
    systems_energetics = {}

    for filename in tqdm(os.listdir(base_path), desc="Reading CSV files"):
        if filename.endswith('.csv'):
            file_path = os.path.join(base_path, filename)
            system_id = filename.split('_')[0]

            # Read the CSV file
            try:
                df = pd.read_csv(file_path, index_col=0)
                systems_energetics[system_id] = df
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return systems_energetics

def remove_outliers(df):
    """
    Remove outliers from a DataFrame using the IQR method.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

def plot_system(lps, df):
    """
    Plots the Lorenz Phase Space diagram for a single system
    """
    # Generate the phase diagram
    lps.plot_data(
        x_axis=df['Ck'],
        y_axis=df['Ca'],
        marker_color=df['Ge'],
        marker_size=df['Ke']
    )

def plot_all_systems_by_region_season(averages_df, output_directory):
    # Initialize the Lorenz Phase Space plotter
    lps = Visualizer(LPS_type='mixed', zoom=True,
                    x_limits=[averages_df['Ck'].min() -1, averages_df['Ck'].max() +1],
                    y_limits=[averages_df['Ca'].min() -1, averages_df['Ca'].max() +1],
                    color_limits=[averages_df['Ge'].min() -1, averages_df['Ge'].max() +1])

    plot_system(lps, averages_df)
    
    # Save the final plot
    plot_filename = f'lps_fixed_means.png'
    plot_path = os.path.join(output_directory, plot_filename)
    lps.fig.savefig(plot_path)

    plt.close()
    print(f"Saved plot to {plot_path}")

def main():
    base_path = '../csv_fixed_framework_database_energy_by_periods'
    output_directory = '../figures_lps/test_fixed_framework/'
    clusters_directory = '../results_kmeans'

    os.makedirs(output_directory, exist_ok=True)

    # Read the energetics data for all systems
    systems_energetics = read_life_cycles(base_path)
    
    id_list_directory = os.path.join(clusters_directory, 'all_systems', 'IcItMD')

    # Get ids to plot
    json_file = f'{id_list_directory}/kmeans_results_IcItMD.json'
    json_data = pd.read_json(json_file)
    cluster_number = 3
    ids = json_data[f'Cluster {cluster_number}']['Cyclone IDs']

    # Get data for the cluster
    systems_energetics_cluster = {k: v for k, v in systems_energetics.items() if int(k) in ids}

    ids = list(systems_energetics_cluster.keys())
    periods = list(systems_energetics_cluster[str(ids[0])].index)

    # Create an empty DataFrame to store the average values
    results_df = pd.DataFrame(columns=['track_id', 'period'] + list(systems_energetics_cluster[str(ids[0])].columns))

    # Collect the results for each period for each system
    for period in periods:
        for track_id, df in systems_energetics_cluster.items():
            # Calculate the average for each term across different periods
            period_values = df.loc[period]
            results_df = pd.concat([results_df, pd.DataFrame({'track_id': [track_id], 'period': [period], **period_values})], ignore_index=True)

    # Remove outliers
    results_df = remove_outliers(results_df)

    # Compute the averages
    results_df.set_index('track_id', inplace=True)
    averages_df = results_df.groupby('period').mean()

    plot_all_systems_by_region_season(averages_df, output_directory)

if __name__ == "__main__":
    main()
