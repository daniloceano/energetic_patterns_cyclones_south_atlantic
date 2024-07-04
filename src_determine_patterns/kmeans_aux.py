# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    kmeans_aux.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/27 10:39:53 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/04 14:02:49 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import explained_variance_score

"""
Auxiliary functions for K-means clustering.
"""

def prepare_to_kmeans(results_energetics):
    """
    Generate the means of the energy data for K-means clustering.

    Parameters:
    - results_energetics (list): A list of pandas DataFrames containing the energy data for each system.

    Returns:
    - dsk_means (ndarray): An array of shape (n_samples, n_features) containing the means of the energy data for each feature.
    """
    combined_df = pd.concat(results_energetics, axis=1)

    Ck1 = combined_df['Ck'].values.T
    Ca1 = combined_df['Ca'].values.T
    Ke1	= combined_df['Ke'].values.T
    Ge1 = combined_df['Ge'].values.T
    BKe1 = combined_df['BKe'].values.T
    BAe1 = combined_df['BAe'].values.T

    dsk_means = np.concatenate((Ck1,Ca1,Ke1,Ge1,BKe1,BAe1),axis=1)
    
    return dsk_means

def slice_mk(mk, LIFECYCLE):
    """
    Slice the cluster centers of a K-means model into separate arrays for each feature.

    Parameters:
    - mk (KMeans): The K-means model object.
    - LIFECYCLE (list): A list of feature names.

    Returns:
    - centers_Ck (ndarray): An array of shape (n_clusters,) containing the cluster centers for the 'Ck' feature.
    - centers_Ca (ndarray): An array of shape (n_clusters,) containing the cluster centers for the 'Ca' feature.
    - centers_Ke (ndarray): An array of shape (n_clusters,) containing the cluster centers for the 'Ke' feature.
    - centers_Ge (ndarray): An array of shape (n_clusters,) containing the cluster centers for the 'Ge' feature.
    - centers_BKe (ndarray): An array of shape (n_clusters,) containing the cluster centers for the 'BKe' feature.
    - centers_BAe (ndarray): An array of shape (n_clusters,) containing the cluster centers for the 'BAe' feature.
    """
    slcenter = len(LIFECYCLE)
    centers_Ck = mk.cluster_centers_[:,0:slcenter]
    centers_Ca = mk.cluster_centers_[:,slcenter:slcenter*2]
    centers_Ke = mk.cluster_centers_[:,slcenter*2:slcenter*3]
    centers_Ge = mk.cluster_centers_[:,slcenter*3:slcenter*4]
    centers_BKe = mk.cluster_centers_[:,slcenter*4:slcenter*5]
    centers_BAe = mk.cluster_centers_[:,slcenter*5:slcenter*6]
    return centers_Ck, centers_Ca, centers_Ke, centers_Ge, centers_BKe, centers_BAe

def preprocess_energy_data(dataframes):
    """
    Prepare data for K-means clustering by organizing and scaling energy terms for each phase across all systems.

    Parameters:
        dataframes (list of DataFrame): Each DataFrame corresponds to one system's data across different phases.

    Returns:
        DataFrame: Processed data ready for clustering, with one row per phase across all systems and scaled features.
    """
    
    combined_df = pd.concat(dataframes, axis=1)

    ck = combined_df['Ck'].values.T
    ca = combined_df['Ca'].values.T
    ke = combined_df['Ke'].values.T
    ge = combined_df['Ge'].values.T
    bae = combined_df['BAe'].values.T
    bke = combined_df['BKe'].values.T

    features = np.concatenate((ck, ca, ke, ge, bae, bke), axis=1)

    return features


def calculate_sse_for_clusters(max_clusters, data_features):
    """
    Calculate the sum of squared distances (SSE) for different numbers of clusters using the K-means algorithm.

    Parameters:
        max_clusters (int): The maximum number of clusters to test (from 1 to max_clusters-1).
        data_features (ndarray): The feature matrix where each row represents a data point and each column a feature.

    Returns:
        list: A list of the sum of squared distances for each tested number of clusters, illustrating how tightly grouped the clusters are within each partition.

    Description:
        This function iterates from 1 up to (max_clusters-1) to fit the K-means model using different cluster sizes.
        For each model, it calculates the sum of squared errors (SSE), which is a measure of the distance of each point from its centroid, summed across all clusters.
        A lower SSE value indicates a model with better fit (though it may also imply overfitting if the number of clusters is too high).
    """
    sse = []
    kmeans_kwargs = {
        "init": "random",   # Initialize the centroids randomly.
        "n_init": 10,       # Run the algorithm 10 times with different centroid seeds.
        "max_iter": 300,    # Maximum number of iterations of the algorithm for a single run.
        "random_state": 42  # Set the random seed for reproducibility.
    }
    
    for k in range(1, max_clusters):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data_features)
        sse.append(kmeans.inertia_)  # Collect the sum of squared distances to centroids for each k.
    
    return sse

def plot_elbow_method(num_clusters, sse, save_dir):
    """
    Plots the sum of squared distances (SSE) for each number of clusters and identifies the "elbow point" using the KneeLocator.

    Parameters:
        num_clusters (int): The maximum number of clusters tested (exclusive).
        sse (list): The sum of squared distances for each number of clusters.
        save_dir (str): The directory where the plot image will be saved.

    Returns:
        int: The identified elbow point indicating the optimal number of clusters.

    Description:
        This function plots the SSE against the number of clusters to visualize the elbow method.
        It uses the KneeLocator to automatically find the elbow point, which suggests the optimal number of clusters for K-means clustering.
        The plot is then saved to the specified directory.
    """
    # Initialize the KneeLocator to find the elbow point
    knee_locator = KneeLocator(range(1, num_clusters), sse, curve="convex", direction="decreasing")

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, num_clusters), sse, '-o', markersize=8, linewidth=2, label='SSE', color='#A1456D')
    ax.plot(knee_locator.elbow, sse[knee_locator.elbow - 1], marker='o', markersize=14,
            label=f'Ideal number of clusters = {knee_locator.elbow}', color='red')

    # Configure labels and title
    ax.set_xlabel('Number of Clusters (k)', fontsize=14)
    ax.set_ylabel('SSE', fontsize=14)
    ax.set_title('Elbow Method Plot for Optimal k', fontsize=16)

    # Add grid and customize its appearance
    ax.grid(color='black', linestyle='-', linewidth=0.25, alpha=0.5)

    # Customize tick parameters
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)

    # Add legend
    ax.legend(fontsize=12, loc='upper right')

    # Save the figure
    filename = os.path.join(save_dir, 'elbow_method_plot.png')
    plt.savefig(filename, dpi=300)
    plt.close()

    # Return the identified elbow point
    return knee_locator.elbow

def kmeans_energy_data(ncenters, ninit, maxiter, results_energetics, algorithm, scaler_type='none', joint_scaling=True):
    """
    Executes the K-Means algorithm to cluster atmospheric energy data, returning cluster centers and related statistics.

    Parameters:
    - ncenters (int): Desired number of clusters in K-Means.
    - ninit (int): Number of initializations for the K-Means algorithm.
    - maxiter (int): Maximum number of iterations for a single run.
    - dataframes (list): List of pandas DataFrames each containing columns 'Ck', 'Ca', 'Ke', 'Ge'.
    - algorithm (str): Name of the algorithm to be used by K-Means.
    - scaler_type (str): Type of normalization to apply ('standard' for StandardScaler, 'minmax' for MinMaxScaler, or 'none' for no scaling).
    - joint_scaling (bool): If True, apply normalization to all data together; if False, normalize separately.

    Returns:
    - centers: Cluster centers for each component.
    - cluster_fractions: Fraction of each cluster in the total dataset.
    - kmeans_model: KMeans object after training.
    """
    # Concatenate all dataframes into a single dataframe
    features = preprocess_energy_data(results_energetics)

    # Choose the scaler based on the scaler_type parameter
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    # Apply scaling if selected
    if scaler:
        scaled_features = scaler.fit_transform(features)
    else:
        scaled_features = features

    # Perform K-Means clustering
    kmeans_model = KMeans(n_clusters=ncenters, init='random', n_init=ninit, max_iter=maxiter, algorithm=algorithm)
    kmeans_model.fit(scaled_features)
    centers = kmeans_model.cluster_centers_

    # Inverse transform the centers if scaling was applied
    if scaler:
        centers = scaler.inverse_transform(centers)

    # Calculate the fraction of each cluster
    cluster_fractions = np.bincount(kmeans_model.labels_) / len(kmeans_model.labels_)

    # Sort clusters by fraction in descending order
    sorted_indices = np.argsort(-cluster_fractions)
    centers = centers[sorted_indices]
    cluster_fractions = cluster_fractions[sorted_indices]
    kmeans_model.labels_ = np.array([sorted_indices[label] for label in kmeans_model.labels_])

    return centers, cluster_fractions, kmeans_model

def sel_clusters_to_df(centers, results_energetics_lifecycle):
    """
    Generate a DataFrame for each cluster by selecting the cluster centers for each feature.
    """
    dataframes = []
    for i in range(centers.shape[0]):  # Assuming centers.shape[0] is the number of clusters
        df = pd.DataFrame(centers[i, :].reshape(1, -1), columns=['Ck', 'Ca', 'Ke', 'Ge'])
        df.index = [f"Cluster {i + 1}"]
        dataframes.append(df)
    return dataframes

import numpy as np

def assign_cyclones_to_clusters(kmeans_model, cyclone_ids, n_clusters):
    """
    Assigns cyclone track IDs to clusters based on K-Means labels.

    Parameters:
    - kmeans_model (KMeans): The trained KMeans model object.
    - cyclone_ids (list): List of unique cyclone track IDs.
    - n_clusters (int): Number of clusters used in KMeans.

    Returns:
    - dict: Dictionary mapping 'Cluster i' to a list of cyclone track IDs in that cluster.
    """
    cluster_labels = [f"Cluster {i + 1}" for i in range(n_clusters)]
    cluster_ids = {label: [] for label in cluster_labels}  # Initialize empty lists for each cluster label

    # Assign each cyclone track ID to its respective cluster
    for idx, track_id in enumerate(cyclone_ids):
        cluster_index = kmeans_model.labels_[idx]  # Get cluster index for this data point
        cluster_label = f"Cluster {cluster_index + 1}"  # Create cluster label
        cluster_ids[cluster_label].append(track_id)

    return cluster_ids