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
ax = sns.barplot(x=list(range(1, len(variance_fraction) + 1)), y=variance_list,
                 hue=list(range(1, len(variance_fraction) + 1)), palette="pastel",
                 legend=False)
plt.xlabel('EOF')
plt.ylabel('Variance Explained (%)')

# Annotate the bar plot with percentages
for index, value in enumerate(variance_list):
    ax.text(index, value + 0.3, f'{value:.1f}%', ha='center')

plt.savefig(os.path.join(output_directory, 'variance_explained.png'))
