import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data
eofs_path = '../csv_eofs_energetics/Total/'
output_directory = '../figures_statistics_energetics/eofs/'
eofs = pd.read_csv(os.path.join(eofs_path, 'eofs.csv'), header=None)
pcs = pd.read_csv(os.path.join(eofs_path, 'pcs.csv'), header=None)
variance_fraction = pd.read_csv(os.path.join(eofs_path, 'variance_fraction.csv'), header=None)

variance_list = [var * 100 for var in variance_fraction[0]]

# Set Seaborn style
sns.set(style="whitegrid")

# Plot the variance explained by each EOF
plt.figure(figsize=(10, 6))
sns.barplot(x=list(range(1, len(variance_fraction) + 1)), y=variance_list, palette="pastel")
plt.xlabel('EOF')
plt.ylabel('Variance Explained (%)')
plt.title('Variance Explained by Each EOF')
plt.savefig(os.path.join(output_directory, 'variance_explained.png'))

# Prepare data for plotting EOFs
eofs_transposed = eofs.T
eofs_transposed.columns = [f'EOF {i+1}' for i in range(eofs_transposed.shape[1])]
eofs_transposed['LEC Term'] = range(eofs_transposed.shape[0])
eofs_melted = eofs_transposed.melt(id_vars=['LEC Term'], var_name='EOF', value_name='Amplitude')

# Plot the first few EOFs
plt.figure(figsize=(10, 6))
sns.lineplot(data=eofs_melted[eofs_melted['EOF'].isin(['EOF 1', 'EOF 2', 'EOF 3'])], x='LEC Term', y='Amplitude', hue='EOF', linewidth=2.5)
plt.xlabel('LEC Term')
plt.ylabel('Amplitude')
plt.title('First Few EOFs')
plt.legend()
plt.savefig(os.path.join(output_directory, 'first_few_eofs.png'))

# Prepare data for plotting PCs
pcs_transposed = pcs.T
pcs_transposed.columns = [f'PC {i+1}' for i in range(pcs_transposed.shape[1])]
pcs_transposed['Cyclone ID'] = range(pcs_transposed.shape[0])
pcs_melted = pcs_transposed.melt(id_vars=['Cyclone ID'], var_name='PC', value_name='Amplitude')

# Plot the first few PCs
plt.figure(figsize=(10, 6))
sns.lineplot(data=pcs_melted[pcs_melted['PC'].isin(['PC 1', 'PC 2', 'PC 3'])], x='Cyclone ID', y='Amplitude', hue='PC', linewidth=2.5)
plt.xlabel('Cyclone ID')
plt.ylabel('Amplitude')
plt.title('First Few PCs')
plt.legend()
plt.savefig(os.path.join(output_directory, 'first_few_pcs.png'))
