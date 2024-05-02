# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_systems_to_be_analysed.py                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/02 10:31:47 by daniloceano       #+#    #+#              #
#    Updated: 2024/05/02 10:34:16 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script will export all the systems that will be analysed in the next step.
These systems corresponde to the systems related to the clusters that have presented the highest values for the barotropic instability term.
"""

import os
import pandas as pd
import numpy as np
from glob import glob

def select_systems(results_directory):
    """
    Filter systems for the choosen clusters.
    """
    clusters_to_use = ["ARG_DJF_cl_2", "ARG_JJA_cl_1",
                       "LA-PLATA_DJF_cl_2", "LA-PLATA_JJA_cl_2",
                       "SE-BR_DJF_cl_2", "SE-BR_JJA_cl_3"]
    
    selected_systems = []

    for cluster in clusters_to_use:
        # Get information about the cluster
        region, season = cluster.split('_')[:2]
        cluster_number = cluster.split('_')[3]

        # Open cluster json file
        system_dir = os.path.join(results_directory, f"{region}_{season}", "IcItMD")
        json_file = glob(f"{system_dir}/*.json")[0]
        df_system = pd.read_json(json_file)

        # Get system IDs
        cluster_ids = df_system[f'Cluster {cluster_number}']['Cyclone IDs']

        for system_id in cluster_ids:
            selected_systems.append(int(system_id))

    return selected_systems

# Select systems
RESULTS_DIR = "../results_kmeans/"
selected_systems = select_systems(RESULTS_DIR)

# Export selected systems
with open("systems_to_be_analysed.txt", "w") as f:
    for system in selected_systems:
        f.write(f"{system}\n")