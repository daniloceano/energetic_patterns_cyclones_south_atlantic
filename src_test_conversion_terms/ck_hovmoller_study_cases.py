import os
from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import cmocean as cmo

# Set paths
RESULTS_DIR = "../../LEC_Results_conversion_terms_test"
FIGURES_DIR = "../figures_test_conversion_terms/study_cases_ck"

# Create directory for saving figures if it doesn't exist
os.makedirs(FIGURES_DIR, exist_ok=True)

# Get list of result directories
results_dirs = glob(f"{RESULTS_DIR}/*")
results_dirs = [os.path.basename(result_dir) for result_dir in results_dirs]

# Loop through each result directory
for result_dir in results_dirs:
    # Create a directory for each track in study cases directory
    track_id = result_dir.split('_')[0]
    track_dir = os.path.join(FIGURES_DIR, track_id)
    os.makedirs(track_dir, exist_ok=True)
    
    # Initialize an empty DataFrame to hold all the integrated data
    all_data = pd.DataFrame()
    
    for i in range(1, 6):
        df = pd.read_csv(f"{RESULTS_DIR}/{result_dir}/Ck_{i}_level.csv", index_col=0)
        df.columns = [float(col) for col in df.columns]
        df['ck_term'] = f'Ck_{i}'
        df['result_dir'] = result_dir
        df['time'] = df.index
        
        # Perform trapezoidal integration for each time step
        df['integrated_value'] = np.trapz(df.drop(['ck_term', 'result_dir', 'time'], axis=1).values, x=df.columns[:-3], axis=1)
        
        all_data = pd.concat([all_data, df])
        
        # Plotting Hovmoller diagram for each ck term
        plt.figure(figsize=(15, 10))
        hovmoller_data = df.drop(['ck_term', 'result_dir', 'integrated_value', 'time'], axis=1).T
        vertical_levels = hovmoller_data.index
        hovmoller_data.index = hovmoller_data.index / 100  # Adjust this if needed to convert to appropriate units
        
        # Reverse the order of vertical levels and corresponding data
        vertical_levels = vertical_levels[::-1]
        hovmoller_data = hovmoller_data.iloc[::-1]

        # Color normalization
        imin = hovmoller_data.min(numeric_only=True).min()
        imax = hovmoller_data.max(numeric_only=True).max()
        absmax = np.amax([np.abs(imin), imax])
        cmap = "coolwarm"
        norm = colors.TwoSlopeNorm(vcenter=0, vmin=-absmax, vmax=absmax)

        levels = np.linspace(-absmax, absmax, 11)

        plt.contourf(hovmoller_data, cmap=cmap, norm=norm, levels=levels, extend='both')
        plt.colorbar(label='Values')
        plt.title(f'Hovmoller Diagram for {track_id} - Ck_{i}')
        plt.xlabel('Time')
        plt.ylabel('Vertical Level')
        plt.xticks(ticks=np.arange(len(df.index)), labels=df.index, rotation=90)
        plt.yticks(ticks=np.arange(len(vertical_levels)), labels=vertical_levels)
        plt.tight_layout()
        plt.savefig(os.path.join(track_dir, f'{track_id}_Ck_{i}_hovmoller.png'))
        plt.close()

        print(f"Hovmoller diagram for {track_id} - Ck_{i} saved successfully.")
        
    all_data.reset_index(drop=True, inplace=True)