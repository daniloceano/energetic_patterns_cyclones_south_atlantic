import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

# Load the data
eofs_path = '../csv_eofs_energetics/Total/'
output_directory = '../figures_statistics_energetics/eofs/'
eofs = pd.read_csv(os.path.join(eofs_path, 'eofs.csv'), header=None)
pcs = pd.read_csv(os.path.join(eofs_path, 'pcs.csv'), header=None)
variance_fraction = pd.read_csv(os.path.join(eofs_path, 'variance_fraction.csv'), header=None)

variance_list = [var * 100 for var in variance_fraction[0]]

# Plot the variance explained by each EOF
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(variance_fraction) + 1), variance_list)
plt.xlabel('EOF')
plt.ylabel('Variance Explained (%)')
plt.title('Variance Explained by Each EOF')
plt.show()

# Plot the first few EOFs
plt.figure(figsize=(10, 6))
for i in range(min(3, eofs.shape[1])):  # Plot first 3 EOFs or less if fewer exist
    plt.plot(eofs.loc[i], label=f'EOF {i+1}')
plt.xlabel('LEC Term')
plt.ylabel('Amplitude')
plt.title('First Few EOFs')
plt.legend()
plt.show()

# Plot the first few PCs
plt.figure(figsize=(10, 6))
for i in range(min(3, pcs.shape[1])):  # Plot first 3 PCs or less if fewer exist
    plt.plot(pcs.loc[:, i], label=f'PC {i+1}')
plt.xlabel('Cyclone ID')
plt.ylabel('Amplitude')
plt.title('First Few PCs')
plt.legend()
plt.show()
