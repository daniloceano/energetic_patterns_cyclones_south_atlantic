import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths
base_path = '../csv_eofs_energetics/'
output_directory = '../figures_statistics_energetics/eofs/'

COLOR_PHASES = {
    'incipient': '#65a1e6',
    'intensification': '#f7b538',
    'mature': '#d62828',
    'decay': '#9aa981',
    'intensification 2': '#ca6702',
    'mature 2': '#9b2226',
    'decay 2': '#386641',
    }

# Load the data for the "Total" phase
total_path = os.path.join(base_path, 'Total')
eofs = pd.read_csv(os.path.join(total_path, 'eofs.csv'), header=None)
pcs = pd.read_csv(os.path.join(total_path, 'pcs.csv'), header=None)
variance_fraction_total = pd.read_csv(os.path.join(total_path, 'variance_fraction.csv'), header=None)

variance_list_total = [var * 100 for var in variance_fraction_total[0]]

# Set Seaborn style
sns.set(style="whitegrid")

# Plot the variance explained by each EOF for the "Total" phase
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=list(range(1, len(variance_fraction_total) + 1)), y=variance_list_total,
                 hue=list(range(1, len(variance_fraction_total) + 1)), palette="pastel", dodge=False,
                 legend=False)
plt.xlabel('EOF', fontsize=16)
plt.ylabel('Variance Explained (%)', fontsize=16)

# Annotate the bar plot with percentages
for index, value in enumerate(variance_list_total):
    ax.text(index, value + 0.3, f'{value:.1f}%', ha='center')

plt.savefig(os.path.join(output_directory, 'variance_explained_total.png'))

# Get the list of directories excluding "Total"
phases = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name)) and name != 'Total']

# Dictionary to store variances
variance_dict = {}

# Load variance fractions for each phase
for phase in phases:
    phase_path = os.path.join(base_path, phase)
    variance_fraction = pd.read_csv(os.path.join(phase_path, 'variance_fraction.csv'), header=None)
    variance_list = [var * 100 for var in variance_fraction[0]]
    variance_dict[phase] = variance_list

# Plot the variance explained by each EOF for all phases
plt.figure(figsize=(14, 8))

# Flatten the variance_dict to plot
variance_data = []
for phase, variances in variance_dict.items():
    for i, variance in enumerate(variances):
        variance_data.append([phase, i+1, variance])

variance_df = pd.DataFrame(variance_data, columns=['Phase', 'EOF', 'Variance Explained (%)'])

# Create the bar plot
ax = sns.barplot(data=variance_df, x='EOF', y='Variance Explained (%)', hue='Phase',
                 palette=COLOR_PHASES.values(), dodge=True,
                     hue_order=['incipient', 'intensification', 'mature', 'decay',
                            'intensification 2', 'mature 2', 'decay 2'])


plt.xlabel('EOF', fontsize=16)
plt.ylabel('Variance Explained (%)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title='Phase', loc='upper right', fontsize=14)

# Save the plot
plt.savefig(os.path.join(output_directory, 'variance_explained_all_phases.png'))