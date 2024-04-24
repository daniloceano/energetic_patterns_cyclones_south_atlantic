# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_barotropic_instability.py                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 19:56:13 by daniloceano       #+#    #+#              #
#    Updated: 2024/04/24 10:26:34 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script will plot maps with the derivative of PV in y, as to display if barotropic instability is occuring.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import cmocean.cm as cmo
from metpy.units import units
import metpy.calc

def get_bounds(lat, lon, length, width):
    # Assuming length and width are in degrees for simplicity; adjust based on your coordinate system if necessary
    lat_min = lat - length / 2
    lat_max = lat + length / 2
    lon_min = lon - width / 2
    lon_max = lon + width / 2
    return lat_min, lat_max, lon_min, lon_max

def map_decorators(ax):
    """
    Adds coastlines and gridlines to the map.

    Parameters:
    - ax: The matplotlib axes object for the map.
    """
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle='dashed', alpha=0.7, linewidth=0.5, color='#383838')
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    gl.top_labels = None
    gl.right_labels = None

# Load the dataset
ds = xr.open_dataset('../../nc_data/Akara-subset_ERA5.nc')

# Load periods
periods = pd.read_csv('../../akara_analysis/LEC_Akara-subset_ERA5_track/periods.csv', parse_dates=['start', 'end'], index_col=0)
periods = periods.sort_values('start')

# Load track file
track = pd.read_csv('../../akara_analysis/LEC_Akara-subset_ERA5_track/Akara-subset_ERA5_track_trackfile', sep=';', index_col=0)
track.index = pd.to_datetime(track.index)

# Assign units
temperature = ds['t'] * units.kelvin
pressure = ds.level * units.hPa
u = ds['u'] * units('m/s')
v = ds['v'] * units('m/s')

# Calculate potential temperature and potential vorticity
potential_temperature = metpy.calc.potential_temperature(pressure, temperature)
pv = metpy.calc.potential_vorticity_baroclinic(potential_temperature, pressure, u, v)

# Calculate the y-derivative of PV
pv_y_derivative = pv.diff('latitude')

# Set up the map projection and features
crs_longlat = ccrs.PlateCarree()
land_feature = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor=cfeature.COLORS['land'])

# Configure the plot for PV
cmap_pv = cmo.balance
pv_300_min = pv.sel(level=300).min().values / 5
pv_300_max = pv.sel(level=300).max().values
norm_pv = colors.TwoSlopeNorm(vmin=pv_300_min, vcenter=0, vmax=pv_300_max)

# Configure the plot for PV derivative
cmap_pv_derivative = cmo.curl
pv_derivative_300_min = pv_y_derivative.sel(level=300).min().values / 5
pv_derivative_300_max = pv_y_derivative.sel(level=300).max().values
norm_derivative = colors.TwoSlopeNorm(vmin=pv_derivative_300_min, vcenter=0, vmax=pv_derivative_300_max)

# Setting up the figure with different axes types
fig = plt.figure(figsize=(10, 20))

# Number of phases
num_phases = len(periods)

# Creating subplots with mixed types of axes
axs = []
for i in range(num_phases):
    # Adding geographic plots for PV and PV derivative
    ax_geo1 = fig.add_subplot(num_phases, 3, i*3 + 1, projection=crs_longlat)
    ax_geo2 = fig.add_subplot(num_phases, 3, i*3 + 2, projection=crs_longlat)
    axs.append([ax_geo1, ax_geo2])
    
    # Adding a regular plot for longitudinal mean
    ax_reg = fig.add_subplot(num_phases, 3, i*3 + 3)
    axs[i].append(ax_reg)

for i, (phase, row) in enumerate(periods.iterrows()):
    ax_pv = axs[i][0]
    ax_pv_derivative = axs[i][1]
    ax_pv_derivative_long_mean = axs[i][2]

    # Make composites for each phase
    track_phase = track.loc[row['start']: row['end']]

    # Initialize lists to store data slices
    pv_slices = []
    pv_y_derivative_slices = []
    pv_y_derivative_long_mean_slices = []
    
    for time, row in track_phase.iterrows():
        central_lat, central_lon = row['Lat'], row['Lon']
        width, length = 5, 5

        # Find the min and max latitudes and longitudes
        lat_min, lat_max = central_lat - length / 2, central_lat + length / 2
        lon_min, lon_max = central_lon - width / 2, central_lon + width / 2
        
        # Slice the PV data at the central point
        pv_300 = pv.sel(level=300).sortby(['latitude', 'longitude'])
        pv_slice = pv_300.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max), time=time)
        pv_slices.append(pv_slice.values)

        # Slice the PV derivative data at the central point
        pv_y_derivative_300 = pv_y_derivative.sel(level=300).sortby(['latitude', 'longitude'])
        pv_y_derivative_slice = pv_y_derivative_300.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max), time=time)
        pv_y_derivative_slices.append(pv_y_derivative_slice.values)

        # Make a longitudinal average for PV derivative
        pv_y_derivative_slice_long = pv_y_derivative_slice.mean(dim='longitude')
        pv_y_derivative_long_mean_slices.append(pv_y_derivative_slice_long.values)
    
    # Compute the spatial mean of the composites
    pv_phase_mean = np.mean(pv_slices, axis=0)
    pv_y_derivative_mean = np.mean(pv_y_derivative_slices, axis=0)
    pv_y_derivative_long_mean = np.mean(pv_y_derivative_long_mean_slices, axis=0)
    composite_x, composite_y = np.meshgrid(np.linspace(0, pv_phase_mean.shape[0]-1, pv_phase_mean.shape[0]), np.linspace(0, pv_phase_mean.shape[1]-1, pv_phase_mean.shape[1]))

    # Main PV plot
    cf = ax_pv.contourf(composite_x, composite_y, pv_phase_mean, cmap=cmap_pv, transform=crs_longlat)
    ax_pv.set_title(phase)
    plt.colorbar(cf, ax=ax_pv, pad=0.1, orientation='horizontal', shrink=0.5)

    # Derivative of PV plot
    cf = ax_pv_derivative.contourf(composite_x, composite_y, pv_y_derivative_mean, cmap=cmap_pv_derivative, transform=crs_longlat)
    ax_pv_derivative.set_title(r'$\frac{\partial PV}{\partial y}$')
    plt.colorbar(cf, ax=ax_pv_derivative, pad=0.1, orientation='horizontal', shrink=0.5)

    # Longitudinal average of PV derivative plot
    ax_pv_derivative_long_mean.plot(pv_y_derivative_long_mean, np.arange(len(pv_y_derivative_long_mean)), color='#003049', linewidth=3)
    ax_pv_derivative_long_mean.axvline(0, color='#c1121f', linestyle='--', linewidth=0.5)
    ax_pv_derivative_long_mean.set_title(r'$\frac{\partial PV}{\partial y}$ Longitudinal Average')

# Adjust layout
plt.tight_layout()
plt.savefig('test.png')

