import pandas as pd
from glob import glob
import os
from lorenz_phase_space.phase_diagrams import Visualizer
import matplotlib.pyplot as plt

ENERGY_PATH = '../csv_database_energy_by_periods'
FIGURES_DIR = '../figures_bae_fluxes/lps'

os.makedirs(FIGURES_DIR, exist_ok=True)

selected_systems = pd.read_csv('systems_to_be_analysed.txt', header=None)[0].tolist()
energy_results_periods = glob(f'{ENERGY_PATH}/*')

# Select results for selected systems
selected_energy_results = [
    period for period in energy_results_periods
    if int(os.path.basename(period).split('_')[0]) in selected_systems
]

# Initialize lists to accumulate data
ck_all, ca_all, ge_all, ke_all = [], [], [], []
bke_all, bae_all = [], []

# Plot LPS for selected systems
for result in selected_energy_results:
    data = pd.read_csv(result)
    track_id = int(os.path.basename(result).split('_')[0])

    # Extract relevant columns
    ck_all.extend(data['Ck'])
    ca_all.extend(data['Ca'])
    ge_all.extend(data['Ge'])
    ke_all.extend(data['Ke'])
    bke_all.extend(data['BKe'])
    bae_all.extend(data['BAe'])

# Convert lists to pandas Series for plotting
ck_all = pd.Series(ck_all)
ca_all = pd.Series(ca_all)
ge_all = pd.Series(ge_all)
ke_all = pd.Series(ke_all)
bke_all = pd.Series(bke_all)
bae_all = pd.Series(bae_all)

# Plot all systems on a single LPS for mixed type
lps_mixed_all = Visualizer(
    LPS_type='mixed', zoom=True,
    y_limits=(ca_all.min() * 1.1, ca_all.max() * 1.1),
    x_limits=(ck_all.min() * 1.1, ck_all.max() * 1.1),
    color_limits=(ge_all.min() * 1.1, ge_all.max() * 1.1),
    marker_limits=(ke_all.min() * 1.1, ke_all.max() * 1.1)
)

lps_mixed_all.plot_data(x_axis=ck_all, y_axis=ca_all, marker_color=ge_all, marker_size=ke_all)
outdir_mixed_all = os.path.join(FIGURES_DIR, 'lps_mixed_all_systems.png')
plt.savefig(outdir_mixed_all, dpi=300)
plt.clf()
print(f"Saved {outdir_mixed_all}")

# Plot all systems on a single LPS for import type
lps_import_all = Visualizer(
    LPS_type='imports', zoom=True,
    y_limits=(bae_all.min() * 1.1, bae_all.max() * 1.1),
    x_limits=(bke_all.min() * 1.1, bke_all.max() * 1.1),
    color_limits=(ge_all.min() * 1.1, ge_all.max() * 1.1),
    marker_limits=(ke_all.min() * 1.1, ke_all.max() * 1.1)
)

lps_import_all.plot_data(x_axis=bke_all, y_axis=bae_all, marker_color=ge_all, marker_size=ke_all)
outdir_import_all = os.path.join(FIGURES_DIR, 'lps_import_all_systems.png')
plt.savefig(outdir_import_all, dpi=300)
plt.clf()
print(f"Saved {outdir_import_all}")

# Plot individual systems (optional)
for result in selected_energy_results:
    data = pd.read_csv(result)
    track_id = int(os.path.basename(result).split('_')[0])

    # Extract relevant columns
    ck = data['Ck']
    ca = data['Ca']
    ge = data['Ge']
    ke = data['Ke']
    bke = data['BKe']
    bae = data['BAe']

    # Initialize the Lorenz Phase Space plotter for mixed type without zoom
    lps_mixed = Visualizer(
        LPS_type='mixed', zoom=True,
        y_limits=(ca.min() * 1.1, ca.max() * 1.1),
        x_limits=(ck.min() * 1.1, ck.max() * 1.1),
        color_limits=(ge.min() * 1.1, ge.max() * 1.1),
        marker_limits=(ke.min() * 1.1, ke.max() * 1.1)
    )

    # Plot data for mixed type
    lps_mixed.plot_data(x_axis=ck, y_axis=ca, marker_color=ge, marker_size=ke)
    outdir = os.path.join(FIGURES_DIR, f'lps_mixed_{track_id}.png')
    plt.savefig(outdir, dpi=300)
    plt.clf()
    print(f"Saved {outdir}")

    # Initialize the Lorenz Phase Space plotter for import type without zoom
    lps_import = Visualizer(
        LPS_type='imports', zoom=True,
        y_limits=(bae.min() * 1.1, bae.max() * 1.1),
        x_limits=(bke.min() * 1.1, bke.max() * 1.1),
        color_limits=(ge.min() * 1.1, ge.max() * 1.1),
        marker_limits=(ke.min() * 1.1, ke.max() * 1.1)
    )

    # Plot data for import type
    lps_import.plot_data(x_axis=bke, y_axis=bae, marker_color=ge, marker_size=ke)
    outdir = os.path.join(FIGURES_DIR, f'lps_import_{track_id}.png')
    plt.savefig(outdir, dpi=300)
    plt.clf()
    print(f"Saved {outdir}")
