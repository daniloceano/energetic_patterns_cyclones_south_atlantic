# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_barotropic_instability.py                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 19:56:13 by daniloceano       #+#    #+#              #
#    Updated: 2024/04/24 23:56:29 by daniloceano      ###   ########.fr        #
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
import matplotlib.gridspec as gridspec

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


# Calculate potential temperature and potential vorticity
ds = xr.open_dataset('pv_composite.nc')
pv_300_composite = ds.__xarray_dataarray_variable__

# Calculate the y-derivative of PV
pv_y_derivative = pv_300_composite.diff('y')

# Set up the map projection and features
crs_longlat = ccrs.PlateCarree()
land_feature = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor=cfeature.COLORS['land'])

# Configure the plot for PV
cmap_pv = cmo.balance
pv_300_min = pv_300_composite.min().values / 5
pv_300_max = pv_300_composite.max().values

if pv_300_min < 0 and pv_300_max > 0:
    norm_pv = colors.TwoSlopeNorm(vmin=pv_300_min, vcenter=0, vmax=pv_300_max)
else:
    abs_min = np.amin([np.abs(pv_300_min), np.abs(pv_300_max)])
    norm_pv = colors.Normalize(vmin=abs_min, vmax=-abs_min)

# Configure the plot for PV derivative
cmap_pv_derivative = cmo.curl
pv_derivative_300_min = pv_y_derivative.min().values / 5
pv_derivative_300_max = pv_y_derivative.max().values
norm_derivative = colors.TwoSlopeNorm(vmin=pv_derivative_300_min, vcenter=0, vmax=pv_derivative_300_max)

# Set up the plot
fig = plt.figure(figsize=(15, 5))  # Wider figure to accommodate three plots
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.5])  # Adjust width ratios to fit your needs
ax_pv = fig.add_subplot(gs[0], projection=crs_longlat)
ax_pv_derivative = fig.add_subplot(gs[1], projection=crs_longlat)
ax_pv_derivative_long_mean = fig.add_subplot(gs[2])  # Regular axis, no projection

# Main PV plot
composite_x, composite_y = pv_300_composite.x, pv_300_composite.y
cf = ax_pv.contourf(composite_x, composite_y, pv_300_composite, cmap=cmap_pv, transform=crs_longlat)
ax_pv.set_title("PV")
plt.colorbar(cf, ax=ax_pv, pad=0.1, orientation='horizontal', shrink=0.5)

# Derivative of PV plot
cf = ax_pv_derivative.contourf(composite_x, composite_y[1:], pv_y_derivative, cmap=cmap_pv_derivative, transform=crs_longlat)
ax_pv_derivative.set_title(r'$\frac{\partial PV}{\partial y}$')
plt.colorbar(cf, ax=ax_pv_derivative, pad=0.1, orientation='horizontal', shrink=0.5)

# Longitudinal average of PV derivative plot
pv_y_derivative_long_mean = pv_y_derivative.mean('x')
ax_pv_derivative_long_mean.plot(pv_y_derivative_long_mean, np.arange(len(pv_y_derivative_long_mean)), color='#003049', linewidth=3)
ax_pv_derivative_long_mean.axvline(0, color='#c1121f', linestyle='--', linewidth=0.5)
ax_pv_derivative_long_mean.set_title(r'$\frac{\partial PV}{\partial y}$ Longitudinal Average')
# Remove x and y ticks and labels
ax_pv_derivative_long_mean.set_xticks([])
ax_pv_derivative_long_mean.set_yticks([])

# Adjust layout
plt.tight_layout()
plt.savefig('test.png')

