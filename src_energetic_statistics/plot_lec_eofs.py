# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_lec_eofs.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/03/05 09:22:16 by daniloceano       #+#    #+#              #
#    Updated: 2024/06/20 18:29:31 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from glob import glob

# Global constants for plotting
TERM_DETAILS = {
    'energy': {
        'terms': ['Az', 'Ae', 'Kz', 'Ke'],
        'label': 'Energy',
        'unit': 'J·m⁻²'
    },
    'conversion': {
        'terms': ['Cz', 'Ca', 'Ck', 'Ce'],
        'label': 'Conversion',
        'unit': 'W·m⁻²'
    },
    'boundary': {
        'terms': ['BAz', 'BAe', 'BKz', 'BKe'],
        'label': 'Transport across boundaries',
        'unit': 'W·m⁻²'
    },
    'budget_diff': {
        'terms': ['∂Az/∂t (finite diff.)', '∂Ae/∂t (finite diff.)', '∂Kz/∂t (finite diff.)', '∂Ke/∂t (finite diff.)'],
        'label': 'Energy budgets (estimated using finite diffs.)',
        'unit': 'W·m⁻²'
    },
    'residuals': {
        'terms': ['RGz', 'RKz', 'RGe', 'RKe'],
        'label': 'Residuals',
        'unit': 'W·m⁻²'
    },
    'generation_dissipation': {
        'terms': ['Gz', 'Ge', 'Dz', 'De'],
        'label': 'Generation/Dissipation',
        'unit': 'W·m⁻²'
    },
    'comparing_generation': {
        'terms': ['RGz', 'RGe', 'Gz', 'Ge'],
        'label': 'Comparing Generation',
        'unit': 'W·m⁻²'
    },
    'comparing_dissipation': {
        'terms': ['RKz', 'Dz', 'RKe', 'De'],
        'label': 'Comparing Dissipation',
        'unit': 'W·m⁻²'
    }
}

COLORS = ['#3B95BF', '#87BF4B', '#BFAB37', '#BF3D3B', '#873e23', '#A13BF0']
MARKERS = ['s', 'o', '^', 'v', '<', '>']
MARKER_COLORS = ['#59c0f0', '#b0fa61', '#f0d643', '#f75452', '#f07243', '#bc6ff7']
LINESTYLE = '-'
LINEWIDTH = 3
TEXT_COLOR = '#383838'
MARKER_EDGE_COLOR = 'grey'
LEGEND_FONT_SIZE = 10
AXIS_LABEL_FONT_SIZE = 12
TITLE_FONT_SIZE = 18

def plot_boxes(ax, data, normalized_data, positions, size, plot_example=False):
    # Define edge width range
    min_edge_width = 0
    max_edge_width = 5

    # Create energy boxes and text labels with updated terms
    for term, pos in positions.items():
        term_value = data[term]

        # Get normalized value for the term to determine edge width
        normalized_value = normalized_data[term]
        # Scale edge width based on normalized value
        edge_width = min_edge_width + (max_edge_width - min_edge_width) * normalized_value / 10

        # Determine value text color based on term value
        value_text_color = '#386641'  # Dark green for positive values
        if term_value < 0:
            value_text_color = '#ae2012'  # Dark red for negative values

        square = patches.Rectangle((pos[0] - size / 2, pos[1] - size / 2), size, size, fill=True, color='skyblue', ec='black', linewidth=edge_width)
        ax.add_patch(square)

        # Term text in bold black
        if plot_example:
            ax.text(pos[0], pos[1], f'{term}', ha='center', va='center', fontsize=16, color='k', fontweight='bold')

        # Value text in the specified color
        else:
            ax.text(pos[0], pos[1], f'{term_value:.2f}', ha='center', va='center', fontsize=16, color=value_text_color, fontweight='bold')
     

def plot_arrow(ax, start, end, term_value, color='#5C5850'):
    """Draws an arrow on the given axes from start to end point."""

    # Determine arrow size based on term value
    for n in range(0, 10):
        if np.abs(term_value) < 1:
            size = 3 + np.abs(term_value)
        elif np.abs(term_value) < 5:
            size = 3 + np.abs(term_value)
        elif np.abs(term_value) < 10:
            size = 3 + np.abs(term_value)
        else:
            size = 15 + np.abs(term_value) * 0.1

    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(facecolor=color, edgecolor=color, width=size, headwidth=size*3, headlength=size*3))

def plot_term_text_and_value(ax, start, end, term, term_value, offset=(0, 0), plot_example=False):
    # Determine text color based on term value
    text_color = '#386641'
    if term_value < 0:
        text_color = '#ae2012'

    mid_point = ((start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1])

    if term in ['Ca', 'BAz', 'BAe']:
        offset_x = -0.05
    elif term in ['Ck', 'BKz', 'BKe']:
        offset_x = 0.05
    else:
        offset_x = 0

    if term == 'Ce':
        offset_y = -0.05
    elif term == 'Cz':
        offset_y = 0.05
    else:
        offset_y = 0

    x_pos = mid_point[0] + offset_x
    y_pos = mid_point[1] + offset_y

    # Plot term text in bold black
    if plot_example:
        ax.text(x_pos, y_pos, term, ha='center', va='center', fontsize=16, color='k', fontweight='bold')

    # Plot value text in the specified color
    else:
        ax.text(x_pos, y_pos, f'{term_value:.2f}', ha='center', va='center',
                color=text_color, fontsize=16, fontweight='bold')

def plot_term_value(ax, position, value, offset=(0, 0)):
    ax.text(position[0] + offset[0], position[1] + offset[1], f'{value:.2f}', ha='center', va='center', fontsize=16)

def plot_term_arrows_and_text(ax, size, term, data, positions, plot_example=False):
    
    term_value = data[term]

    arrow_color = '#5C5850'  # Default color

    if term == 'Cz':
        start = (positions['∂Az/∂t'][0] + size/2, positions['∂Az/∂t'][1]) 
        end = (positions['∂Kz/∂t'][0] - size/2, positions['∂Kz/∂t'][1])
        plot_term_text_and_value(ax, start, end, term, term_value, offset=(0, 0.1), plot_example=plot_example)

    elif term == 'Ca':
        start = (positions['∂Az/∂t'][0], positions['∂Az/∂t'][1] - size/2)
        end = (positions['∂Ae/∂t'][0], positions['∂Ae/∂t'][1] + size/2)
        plot_term_text_and_value(ax, start, end, term, term_value, offset=(-0.1, 0), plot_example=plot_example)

    elif term == 'Ck':
        start = (positions['∂Kz/∂t'][0], positions['∂Ke/∂t'][1] + size/2)
        end = (positions['∂Ke/∂t'][0], positions['∂Kz/∂t'][1] - size/2)
        plot_term_text_and_value(ax, start, end, term, term_value, offset=(0.1, 0), plot_example=plot_example)

    elif term == 'Ce':
        start = (positions['∂Ae/∂t'][0] + size/2, positions['∂Ke/∂t'][1])
        end = (positions['∂Ke/∂t'][0] - size/2, positions['∂Ae/∂t'][1])
        plot_term_text_and_value(ax, start, end, term, term_value, offset=(0, -0.1), plot_example=plot_example)

    # Plot text for generation terms
    elif term == 'Gz':
        start = (positions['∂Az/∂t'][0], 1)
        end = (positions['∂Az/∂t'][0], positions['∂Az/∂t'][1] + size/2)
        plot_term_text_and_value(ax, start, end, term, term_value, offset=(0, 0.2), plot_example=plot_example)

    elif term == 'Ge':
        start = (positions['∂Ae/∂t'][0], -1)
        end = (positions['∂Ae/∂t'][0], positions['∂Ae/∂t'][1] - size/2)
        plot_term_text_and_value(ax, start, end, term, term_value, offset=(0, -0.2), plot_example=plot_example)

    # Plot text for residuals
    elif term == 'RGz':
        return None, None
    #     start = (positions['∂Az/∂t'][0], 1)
    #     end = (positions['∂Az/∂t'][0], positions['∂Az/∂t'][1] + size/2)
    #     plot_term_text_and_value(ax, start, end, term, term_value, offset=(0, 0.2), plot_example=plot_example)

    elif term == 'RGe':
        return None, None
    #     start = (positions['∂Ae/∂t'][0], -1)
    #     end = (positions['∂Ae/∂t'][0], positions['∂Ae/∂t'][1] - size/2)
    #     plot_term_text_and_value(ax, start, end, term, term_value, offset=(0, -0.2), plot_example=plot_example)

    elif term == 'RKz':
        start = (positions['∂Kz/∂t'][0], 1)
        end = (positions['∂Kz/∂t'][0], positions['∂Kz/∂t'][1] + size/2)
        plot_term_text_and_value(ax, start, end, term, term_value, offset=(0, 0.2), plot_example=plot_example)

    elif term == 'RKe':
        start = (positions['∂Ke/∂t'][0], -1)
        end = (positions['∂Ke/∂t'][0], positions['∂Ke/∂t'][1] - size/2)
        plot_term_text_and_value(ax, start, end, term, term_value, offset=(0, -0.2), plot_example=plot_example)

    # Plot text for boundaries
    elif term in ['BAz', 'BAe']:
            refered_term = '∂Az/∂t' if term == 'BAz' else '∂Ae/∂t'
            start = (-1, positions[refered_term][1])
            end = (positions[refered_term][0] - size/2, positions[refered_term][1])
            plot_term_text_and_value(ax, start, end, term, term_value, offset=(-0.23, 0), plot_example=plot_example) 

    elif term in ['BKz', 'BKe']:
        refered_term = '∂Kz/∂t' if term == 'BKz' else '∂Ke/∂t'
        start = (1, positions[refered_term][1])
        end = (positions[refered_term][0] + size/2, positions[refered_term][1])
        plot_term_text_and_value(ax, start, end, term, term_value, offset=(0.23, 0), plot_example=plot_example) 

    if term_value < 0:
        start_normalized, end_normalized = end, start  # Swap start and end for negative values
    else:
        start_normalized, end_normalized = start, end

    # Plot arrow
    plot_arrow(ax, start_normalized, end_normalized, data[term], color=arrow_color)

    return start, end

def _call_plot(data, normalized_data, plot_example=False):
    # Prepare data
    conversions = TERM_DETAILS['conversion']['terms']
    residuals = TERM_DETAILS['residuals']['terms'][1::2]
    boundaries = TERM_DETAILS['boundary']['terms']
    geneneration = TERM_DETAILS['generation_dissipation']['terms'][:2]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis('off')

    # Define positions and size of energy boxes
    positions = {
        '∂Az/∂t': (-0.5, 0.5),
        '∂Ae/∂t': (-0.5, -0.5),
        '∂Kz/∂t': (0.5, 0.5),
        '∂Ke/∂t': (0.5, -0.5)
    }
    size = 0.4
    
    plot_boxes(ax, data, normalized_data, positions, size, plot_example)

    # # Add title
    if not plot_example:
    #     if type(data.name) == pd.Timestamp:
    #         data.name = data.name.strftime('%Y-%m-%d')
        ax.text(0, 0, data.name, fontsize=16, ha='center', va='center', fontweight='bold', color='black')

    for term in conversions + geneneration + boundaries + residuals:
        start, end = plot_term_arrows_and_text(ax, size, term, data, positions, plot_example=plot_example)

    plt.tight_layout()

def normalize_idata_not_energy(data):
    df_not_energy = np.abs(data.drop(columns=['Az', 'Ae', 'Kz', 'Ke']))
    normalized_data_not_energy = ((df_not_energy - df_not_energy.min()) / (df_not_energy.max() - df_not_energy.min()))
    normalized_data_not_energy = df_not_energy.clip(lower=1.5, upper=15)
    return normalized_data_not_energy

def main():
    results_path = '../csv_database_energy_by_periods'
    eofs_path = '../csv_eofs_energetics'
    output_directory = '../figures_statistics_energetics/eofs/'
    os.makedirs(output_directory, exist_ok=True)

    phases = ['incipient', 'intensification', 'mature', 'decay', 'intensification 2', 'mature 2', 'decay 2', 'Total']

    for phase in phases:

        # Read dummy results in order to get the columns
        results = glob(results_path+'/*.csv')
        dummy_result = pd.read_csv(results[0], index_col=0)
        columns = dummy_result.columns

        # Load the DataFrame with EOF values
        phase_directory = os.path.join(eofs_path, phase)

        for mode in ['', '_with_mean']:
            eof_file_phase = os.path.join(phase_directory, f'eofs{mode}.csv')
            df = pd.read_csv(eof_file_phase, header=None)

            # Set the columns of df to be equal to the columns variable
            df.columns = columns

            # Fix column names
            df.columns = [col if '∂' not in col else '∂' + col.split('∂')[1].split('/')[0] + '/∂t' for col in df.columns]

            # Normalize data
            df_not_energy = np.abs(df.drop(columns=['Az', 'Ae', 'Kz', 'Ke']))
            normalized_data_not_energy = ((df_not_energy - df_not_energy.min().mean()) / (df_not_energy.max().max() - df_not_energy.min().min()))
            normalized_data_not_energy = df_not_energy.clip(lower=1.5, upper=15)

            # Load explained variance
            explained_variance = pd.read_csv(os.path.join(phase_directory, 'variance_fraction.csv'), header=None)

            # plot example figure
            if mode != '_with_mean':
                _call_plot(df.iloc[0], normalized_data_not_energy.iloc[0], plot_example=True)
                plt.savefig(os.path.join(output_directory, 'example.png'))

            # plot each deaily mean
            for eof in range(len(df)):
                # Create directory for each EOF
                eof_output_directory = os.path.join(output_directory, f'eof_{eof+1}')
                os.makedirs(eof_output_directory, exist_ok=True)

                # Get data for the current EOF
                idata = df.iloc[eof]
                normalized_idata_not_energy = normalize_idata_not_energy(idata)

                # Title for each EOF
                explained_variance_eof_percentage = round(float(explained_variance.iloc[eof].values[0] * 100), 2)
                idata.name = f'\nEOF {eof+1}\n{phase.capitalize()}\nExp. Var.: {explained_variance_eof_percentage}%'

                # plot
                _call_plot(idata, normalized_idata_not_energy)
                plt.savefig(os.path.join(eof_output_directory, f'{phase}_eof{eof+1}{mode}.png'))
                print(f"EOF {eof+1} for {phase} complete and saved to file.")

if __name__ == "__main__":
    
    main()