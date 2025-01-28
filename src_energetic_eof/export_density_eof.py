# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_density_eof.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/09 12:48:17 by Danilo            #+#    #+#              #
#    Updated: 2025/01/28 10:19:52 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import pandas as pd
import xarray as xr
import numpy as np

from glob import glob
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

def compute_density(tracks, num_time):
    k = 64
    longrd = np.linspace(-180, 180, 2 * k)
    latgrd = np.linspace(-87.863, 87.863, k)
    tx, ty = np.meshgrid(longrd, latgrd)
    mesh = np.vstack((ty.ravel(), tx.ravel())).T
    mesh *= np.pi / 180.  # Convert to radians

    pos = tracks[['lat vor', 'lon vor']]
    h = np.vstack([pos['lat vor'].values, pos['lon vor'].values]).T
    h *= np.pi / 180.  # Convert to radians

    kde = KernelDensity(bandwidth=0.05, metric='haversine', kernel='gaussian', algorithm='ball_tree').fit(h)
    v = np.exp(kde.score_samples(mesh)).reshape((k, 2 * k))

    R = 6369345.0 * 1e-3  # Earth radius in km
    factor = (1 / (R ** 2)) * 1.e6
    density = v * pos.shape[0] * factor / num_time

    return density, longrd, latgrd

def export_density_by_eof(tracks, num_time, output_directory):
    unique_eofs = tracks['dominant_eof'].unique()

    for eof in unique_eofs:
        eof_tracks = tracks[tracks['dominant_eof'] == eof]
        print(f"Computing density for EOF {eof}...")

        density, lon, lat = compute_density(eof_tracks, num_time)
        data = xr.DataArray(density, coords={'lon': lon, 'lat': lat}, dims=['lat', 'lon'], name=f"EOF_{eof}")

        region_str = "SAt_"
        fname = f'{output_directory}/{region_str}track_density_eof_{eof}.nc'
        data.to_netcdf(fname)
        print(f'Wrote {fname}')

def main():
    output_directory = f'../csv_eofs_energetics_with_track/Total/track_density/'
    os.makedirs(output_directory, exist_ok=True)

    # Get tracks
    tracks = pd.read_csv('../tracks_SAt_filtered/tracks_SAt_filtered_with_energetics.csv')

    # Load EOFs with dominant_eof information
    pcs_path = "../csv_eofs_energetics_with_track/Total/pcs_with_dominant_eof.csv"
    pcs_df = pd.read_csv(pcs_path)

    # Merge tracks with EOF data
    tracks = tracks.merge(pcs_df[['track_id', 'dominant_eof']], on='track_id', how='left')

    # Filter for unique years and months
    tracks['date'] = pd.to_datetime(tracks['date'])
    unique_years_months = tracks['date'].dt.to_period('M').unique()
    num_time = len(unique_years_months)
    print(f"Total number of time months: {num_time}")

    # Export density maps for each EOF
    export_density_by_eof(tracks, num_time, output_directory)

if __name__ == '__main__':
    main()
