# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_systems_to_be_analysed.py                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/05/02 10:31:47 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/14 13:24:35 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script will export all the systems that will be analysed in the next step.
These systems correspond to the systems related to the clusters that have presented the highest values for the imports of eddy APE term.
"""

import os
import pandas as pd
import numpy as np
from glob import glob

def select_systems(results_directory):
    """
    Filter systems for the choosen clusters.
    """
    
    selected_systems = []

    # Get information about the cluster
    cluster_number = 3

    # Open cluster json file
    system_dir = os.path.join(results_directory, "all_systems", "IcItMD")
    json_file = glob(f"{system_dir}/*.json")[0]
    df_system = pd.read_json(json_file)

    # Get system IDs
    cluster_ids = df_system[f'Cluster {cluster_number}']['Cyclone IDs']

    for system_id in cluster_ids:
        selected_systems.append(int(system_id))

    print(f"{len(selected_systems)} systems selected")

    return selected_systems

# Select systems
RESULTS_DIR = "../results_kmeans/"
selected_systems = select_systems(RESULTS_DIR)

# Export selected systems
with open("systems_to_be_analysed.txt", "w") as f:
    for system in selected_systems:
        f.write(f"{system}\n")