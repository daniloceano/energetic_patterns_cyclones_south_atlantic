import os
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm  # Import tqdm for the progress bar
from sklearn.cluster import KMeans

ENERGETICSPATH = '../csv_database_energy_by_periods/'
CSVSAVEPATH = '../csv_patterns/all_systems/'
LIFECYCLE = {'incipient', 'intensification', 'mature', 'decay'}


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

    dsk_means = np.concatenate((Ck1,Ca1,Ke1,Ge1),axis=1)
    
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
    """
    slcenter = len(LIFECYCLE)
    centers_Ck = mk.cluster_centers_[:,0:slcenter]
    centers_Ca = mk.cluster_centers_[:,slcenter:slcenter*2]
    centers_Ke = mk.cluster_centers_[:,slcenter*2:slcenter*3]
    centers_Ge = mk.cluster_centers_[:,slcenter*3:slcenter*4]
    return centers_Ck, centers_Ca, centers_Ke, centers_Ge

def sel_clusters_to_df(centers_Ck, centers_Ca, centers_Ke, centers_Ge, results_energetics_lifecycle):
    """
    Generate a DataFrame for each cluster by selecting the cluster centers for each feature and assigning them to the corresponding DataFrame.

    Parameters:
    - centers_Ck (ndarray): An array of shape (n_clusters,) containing the cluster centers for the 'Ck' feature.
    - centers_Ca (ndarray): An array of shape (n_clusters,) containing the cluster centers for the 'Ca' feature.
    - centers_Ke (ndarray): An array of shape (n_clusters,) containing the cluster centers for the 'Ke' feature.
    - centers_Ge (ndarray): An array of shape (n_clusters,) containing the cluster centers for the 'Ge' feature.
    - results_energetics_lifecycle (list): A list of DataFrames containing the results of the energetics for a particular lifecycle.

    Returns:
    - df_cl1 (DataFrame): A DataFrame containing the cluster centers for the first cluster.
    - df_cl2 (DataFrame): A DataFrame containing the cluster centers for the second cluster.
    - df_cl3 (DataFrame): A DataFrame containing the cluster centers for the third cluster.
    - df_cl4 (DataFrame): A DataFrame containing the cluster centers for the fourth cluster.
    """

    cl1Ck = centers_Ck[0,:]
    cl2Ck = centers_Ck[1,:]
    cl3Ck = centers_Ck[2,:]
    cl4Ck = centers_Ck[3,:]

    cl1Ca = centers_Ca[0,:]
    cl2Ca = centers_Ca[1,:]
    cl3Ca = centers_Ca[2,:]
    cl4Ca = centers_Ca[3,:]

    cl1Ke = centers_Ke[0,:]
    cl2Ke = centers_Ke[1,:]
    cl3Ke = centers_Ke[2,:]
    cl4Ke = centers_Ke[3,:]

    cl1Ge = centers_Ge[0,:]
    cl2Ge = centers_Ge[1,:]
    cl3Ge = centers_Ge[2,:]
    cl4Ge = centers_Ge[3,:]

    df_cl1 = results_energetics_lifecycle[0].copy()
    df_cl1[:] = np.nan
    df_cl2 = df_cl1.copy()
    df_cl3 = df_cl1.copy()
    df_cl4 = df_cl1.copy()
        #att branch
    df_cl1['Ck'] = cl1Ck
    df_cl1['Ca'] = cl1Ca
    df_cl1['Ke'] = cl1Ke
    df_cl1['Ge'] = cl1Ge

    df_cl2['Ck'] = cl2Ck
    df_cl2['Ca'] = cl2Ca
    df_cl2['Ke'] = cl2Ke
    df_cl2['Ge'] = cl2Ge

    df_cl3['Ck'] = cl3Ck
    df_cl3['Ca'] = cl3Ca
    df_cl3['Ke'] = cl3Ke
    df_cl3['Ge'] = cl3Ge

    df_cl4['Ck'] = cl4Ck
    df_cl4['Ca'] = cl4Ca
    df_cl4['Ke'] = cl4Ke
    df_cl4['Ge'] = cl4Ge

    return df_cl1, df_cl2, df_cl3, df_cl4

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

results_energetics_lifecycle = []
for df in tqdm(results_energetics_all_systems, desc="Filtering lifecycles"):
    phases = set(df.index.dropna())
    if phases == LIFECYCLE:
        results_energetics_lifecycle.append(df)

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