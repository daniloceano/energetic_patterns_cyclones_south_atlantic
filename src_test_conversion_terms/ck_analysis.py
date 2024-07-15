import os
from glob import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set paths
RESULTS_DIR = "../../LEC_Results_conversion_terms_test"
FIGURES_DIR = "../figures_test_conversion_terms/ck_terms"

# Create directory for saving figures if it doesn't exist
os.makedirs(FIGURES_DIR, exist_ok=True)

# Get list of result directories
results_dirs = glob(f"{RESULTS_DIR}/*")
results_dirs = [os.path.basename(result_dir) for result_dir in results_dirs]

# Initialize an empty DataFrame to hold all the integrated data
all_data = pd.DataFrame()

# Loop through each result directory and concatenate the data
for result_dir in results_dirs:
    for i in range(1, 6):
        df = pd.read_csv(f"{RESULTS_DIR}/{result_dir}/Ck_{i}_level.csv", index_col=0)
        df.columns = [float(col) for col in df.columns]
        df['ck_term'] = f'Ck_{i}'
        df['result_dir'] = result_dir
        
        # Perform trapezoidal integration for each time step
        df['integrated_value'] = np.trapz(df.drop(['ck_term', 'result_dir'], axis=1).values, x=df.columns[:-2], axis=1)
        
        all_data = pd.concat([all_data, df])

# Keep only the integrated values for the box plot
integrated_data = all_data[['integrated_value', 'ck_term']]

# Plotting the box plot for integrated values
plt.figure(figsize=(15, 10))
sns.boxplot(x='ck_term', y='integrated_value', data=integrated_data)
plt.xticks(rotation=90)
plt.title('Box Plot of Integrated Values for Each CK Term')
plt.xlabel('CK Terms')
plt.ylabel('Integrated Values')
plt.savefig(os.path.join(FIGURES_DIR, 'integrated_values_boxplot.png'))
plt.close()

# Extracting headers (terms)
terms = [col for col in df.columns if col not in ['ck_term', 'result_dir', 'integrated_value']]

# Plotting vertical distribution for each term (ck_1, ck_2, etc.)
fig, axes = plt.subplots(len(all_data['ck_term'].unique()), 1, figsize=(15, len(all_data['ck_term'].unique()) * 3), sharex=True)

for i, term in enumerate(all_data['ck_term'].unique()):
    term_data = all_data[all_data['ck_term'] == term].reset_index()
    melted_data = term_data.melt(id_vars=['time', 'ck_term', 'result_dir'], var_name='vertical_level', value_name='value')
    # Remove integrated values
    melted_data = melted_data[melted_data['vertical_level'] != 'integrated_value']
    
    sns.boxplot(x='vertical_level', y='value', data=melted_data, ax=axes[i])
    axes[i].set_title(f'Vertical Distribution for {term}')
    axes[i].set_xlabel('Vertical Level')
    axes[i].set_ylabel('Values')
    axes[i].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'vertical_distribution_boxplot.png'))
plt.close()
