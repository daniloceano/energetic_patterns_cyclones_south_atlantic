# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_composites.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/23 19:56:13 by daniloceano       #+#    #+#              #
#    Updated: 2024/07/04 11:53:08 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script will plot maps with the advection of temperature, Geopotential height contours, and wind vectors.
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from metpy.units import units

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
GRID_LABEL_SIZE = 10
FIGURES_DIR = '../figures_bae_fluxes/composites'
CRS = ccrs.PlateCarree()

NC_PATH = '../results_nc_files/composites_bae/'

def plot_map(ax, data, u, v, hgt, **kwargs):
    """Plot data with Geopotential height contours and wind vectors."""
    transform = ccrs.PlateCarree()
    cmap, levels, title, units = kwargs.get('cmap'), kwargs.get('levels'), kwargs.get('title'), kwargs.get('units')

    # Create the contour plot for the data
    levels_min, levels_max = np.min(levels), np.max(levels)
    if levels_min < 0 and levels_max > 0:
        norm = colors.TwoSlopeNorm(vmin=np.min(levels), vcenter=0, vmax=np.max(levels))
    else:
        norm = colors.Normalize(vmin=np.min(levels), vmax=np.max(levels))
    cf = ax.contourf(data.x, data.y, data, cmap=cmap, norm=norm, transform=transform, levels=levels, extend='both')
    
    try:
        colorbar = plt.colorbar(cf, ax=ax, pad=0.1, orientation='horizontal', shrink=0.5, label=units)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))  # Adjust these limits based on your specific needs
        colorbar.ax.xaxis.set_major_formatter(formatter)

        current_ticks = colorbar.get_ticks()
        new_ticks = current_ticks[::2]  # Take every second tick
        colorbar.set_ticks(new_ticks)   # Set the modified ticks
        colorbar.update_ticks()

    except ValueError:
        pass

    # Add hgt as a contour
    ax.contour(data.x, data.y, hgt, colors='k', linestyles='-', linewidths=2, transform=transform)

    wsp = np.sqrt(u**2 + v**2)
    wsp_mean, wsp_max = int(np.mean(wsp)), round(int(np.max(wsp)), -1)

    # Plot wind vectors
    skip_n = 8
    scale_factor = wsp_mean * 60 if wsp_max <= 30 else wsp_mean * 30
    label = wsp_max if wsp_max <= 30 else wsp_max + 10
    width = 0.005
    skip = (slice(None, None, skip_n), slice(None, None, skip_n))
    qu = ax.quiver(data.x[skip[0]], data.y[skip[0]], u[skip], v[skip], transform=transform, zorder=1,
              width=width, headwidth=2, headlength=2, headaxislength=2,  scale=scale_factor)
    
    # Quiver key
    ax.quiverkey(qu, X=0.5, Y=-0.1, U=label, label=f'{label} m/s', labelpos='E', coordinates='axes')

    # Set up grid lines
    ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.5, color='k')

    # Draw the 15x15 square centered on the domain
    lon_center, lat_center = 0, 0
    square_half_size = 7.5 / 0.25
    square_lon = [lon_center - square_half_size, lon_center + square_half_size, lon_center + square_half_size, lon_center - square_half_size, lon_center - square_half_size]
    square_lat = [lat_center - square_half_size, lat_center - square_half_size, lat_center + square_half_size, lat_center + square_half_size, lat_center - square_half_size]
    ax.plot(square_lon, square_lat, transform=transform, color='r', linewidth=2)

    # Customize the ticks on x and y axes
    ax.xaxis.set_major_locator(ticker.AutoLocator())  # Automatically determine the location of ticks
    ax.yaxis.set_major_locator(ticker.AutoLocator())

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))

    ax.set_xticks(np.arange(-50, 50, 5))
    ax.set_yticks(np.arange(-50, 50, 5))

    ax.tick_params(axis='both', which='major', labelsize=GRID_LABEL_SIZE)

    ax.set_title(title, fontsize=TITLE_SIZE)  # You can adjust the fontsize as necessary

def plot_variable(data, u, v, hgt, output_dir, **map_attrs):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=CRS)
    plot_map(ax, data, u, v, hgt, **map_attrs)
    plt.tight_layout()
    filename = map_attrs['filename']
    file_path = os.path.join(output_dir, filename)
    plt.savefig(file_path)
    print(f'Saved {filename}')

def plot_composites(netcdf_dir, phase):

    # Open the netCDF file
    filepath = os.path.join(netcdf_dir, f'bae_composite_{phase}_mean.nc')
    ds = xr.open_dataset(filepath)

    # Set up output directory
    output_dir = os.path.join(FIGURES_DIR, phase)
    os.makedirs(output_dir, exist_ok=True)

    # Open variables
    temp_advection = ds['temp_advection']
    temp_advection = (temp_advection * units('K/s')).metpy.convert_units('K/day')
    u = ds['u']
    v = ds['v']
    hgt = ds['hgt']
    T = ds['t']

    for level in ds.level:

        temp_advection_level = temp_advection.sel(level=level)
        u_level = u.sel(level=level)
        v_level = v.sel(level=level)
        hgt_level = hgt.sel(level=level)
        T_level = T.sel(level=level)

        # Compute u * T^2 and v * T^2
        u_T2 = u_level * (T_level ** 2)
        v_T2 = v_level * (T_level ** 2)

        level_str = str(int(level))

        # Calculate the maximum absolute value for symmetric levels
        min_val = temp_advection_level.min(skipna=True).metpy.dequantify().item()
        max_val = temp_advection_level.max(skipna=True).metpy.dequantify().item()
        abs_max = max(abs(min_val), abs(max_val))

        min_val_u_T2 = u_T2.min(skipna=True).metpy.dequantify().item()
        max_val_u_T2 = u_T2.max(skipna=True).metpy.dequantify().item()
        abs_max_u_T2 = max(abs(min_val_u_T2), abs(max_val_u_T2))

        min_val_v_T2 = v_T2.min(skipna=True).metpy.dequantify().item()
        max_val_v_T2 = v_T2.max(skipna=True).metpy.dequantify().item()
        abs_max_v_T2 = max(abs(min_val_v_T2), abs(max_val_v_T2))

        # Create symmetric contour levels
        contour_levels = np.linspace(-abs_max, abs_max, 12)
        contour_levels_u_T2 = np.linspace(-abs_max_u_T2, abs_max_u_T2, 12)
        contour_levels_v_T2 = np.linspace(-abs_max_v_T2, abs_max_v_T2, 12)

        # Plot Temperature Advection
        map_attrs_advection = {
            'cmap': 'RdBu_r',
            'title': r'Temperature Advection @ ' + f'{level_str} hPa',
            'levels': contour_levels,
            'units': 'K/day',
            'filename': f'composite_temp_advection_{level_str}hpa.png'
        }
        plot_variable(temp_advection_level, u_level, v_level, hgt_level, output_dir, **map_attrs_advection)

        # Plot u * T^2
        map_attrs_u_T2 = {
            'cmap': 'RdBu_r',
            'title': f'$uT^2$ @ {level_str} hPa',
            'levels': contour_levels_u_T2,
            'units': 'm^2/s^2',
            'filename': f'composite_u_T2_{level_str}hpa.png'
        }
        plot_variable(u_T2, u_level, v_level, hgt_level, output_dir, **map_attrs_u_T2)

        # Plot v * T^2
        map_attrs_v_T2 = {
            'cmap': 'RdBu_r',
            'title': f'$vT^2$ @ {level_str} hPa',
            'levels': contour_levels_v_T2,
            'units': 'm^2/s^2',
            'filename': f'composite_v_T2_{level_str}hpa.png'
        }
        plot_variable(v_T2, u_level, v_level, hgt_level, output_dir, **map_attrs_v_T2)

def main():
    # create output directory if it doesn't exist
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for phase in ['incipient', 'mature']:
        plot_composites(NC_PATH, phase)


if __name__ == '__main__':
    main()
