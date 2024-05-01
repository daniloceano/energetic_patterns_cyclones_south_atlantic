#!/bin/bash

# Navigate to the LEC_Results directory
cd ../../LEC_Results/

# Initialize counters
count_with_track_results=0
count_without_track_results_but_with_csv=0
count_without_any_csv=0

# List all directories
for dir in */ ; do
    # Check if the directory contains a *_track_results.csv file
    if ls "${dir}"*"_track_results.csv" 1> /dev/null 2>&1; then
        # Increment the counter if the *_track_results.csv file is found
        ((count_with_track_results++))
    else
        # Check if the directory contains any other .csv file
        if ls "${dir}"*.csv 1> /dev/null 2>&1; then
            # Increment the counter if other .csv files are found
            ((count_without_track_results_but_with_csv++))
        else
            # Increment the counter if no .csv files are found
            ((count_without_any_csv++))
        fi
    fi
done

# Print the results
echo "Directories with *_track_results.csv: $count_with_track_results"
echo "Directories without *_track_results.csv but with other .csv files: $count_without_track_results_but_with_csv"
echo "Directories without any .csv files: $count_without_any_csv"

