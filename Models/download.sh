#!/bin/bash

# Create the output folder Data/ if it doesn't exist
if [ -d "Data" ]; then
  echo "Directory 'Data' already exists. Doing nothing."
else
  mkdir "Data"
  echo "Directory 'Data' created."
fi

# Define the years and splits
years=("2019" "2020")
modes=("test" "train" "val")

# Get the data for year 2019 and 2020
for year in "${years[@]}"
do

  # Download patches data
  for i in {0..19}
  do
    # Construct the URL with the current value of i
    wget --header="Cookie: oc_sessionPassphrase=G58vqE1m1B4pBYVOAnvgNHrLX6tHcaWWPGtOIaL16X0lxxinrTg3off1h7WaVNukrS39Jr1BkhsdJreHtvLeVevT9CrXjr%2BzbE4Nli9dI4BHB1AjqAiLQl82nCZZ0QmC; ocDownloadStarted=gxairgqzc" \
         --header="Connection: keep-alive" \ 
         --header="Referer: https://libdrive.ethz.ch/" \
         "https://libdrive.ethz.ch/index.php/s/VPio6i5UlXTgir0/download?path=%2F&files=data_subset-${year}-v4_${i}-20.h5&downloadStartSecret=gxairgqzc" -O Data/data_subset-${year}-v4_${i}-20.h5
  done

  # Download tabular data
  for mode in "${modes[@]}"
  do
    wget --header="Cookie: oc_sessionPassphrase=G58vqE1m1B4pBYVOAnvgNHrLX6tHcaWWPGtOIaL16X0lxxinrTg3off1h7WaVNukrS39Jr1BkhsdJreHtvLeVevT9CrXjr%2BzbE4Nli9dI4BHB1AjqAiLQl82nCZZ0QmC; ocDownloadStarted=gxairgqzc" \
         --header="Connection: keep-alive" \ 
         --header="Referer: https://libdrive.ethz.ch/" \
         "https://libdrive.ethz.ch/index.php/s/VPio6i5UlXTgir0/download?path=%2F&files=${mode}_features_${year}-v4.csv&downloadStartSecret=gxairgqzc" -O Data/${mode}_features_${year}-v4.csv
  done

done

# Download the statistics
wget --header="Cookie: oc_sessionPassphrase=G58vqE1m1B4pBYVOAnvgNHrLX6tHcaWWPGtOIaL16X0lxxinrTg3off1h7WaVNukrS39Jr1BkhsdJreHtvLeVevT9CrXjr%2BzbE4Nli9dI4BHB1AjqAiLQl82nCZZ0QmC; ocDownloadStarted=gxairgqzc" \
     --header="Connection: keep-alive" \ 
     --header="Referer: https://libdrive.ethz.ch/" \
     "https://libdrive.ethz.ch/index.php/s/VPio6i5UlXTgir0/download?path=%2F&files=statistics_subset_2019-2020-v4.pkl&downloadStartSecret=gxairgqzc" -O Data/statistics_subset_2019-2020-v4.pkl

# Download the train/test/val split
wget --header="Cookie: oc_sessionPassphrase=G58vqE1m1B4pBYVOAnvgNHrLX6tHcaWWPGtOIaL16X0lxxinrTg3off1h7WaVNukrS39Jr1BkhsdJreHtvLeVevT9CrXjr%2BzbE4Nli9dI4BHB1AjqAiLQl82nCZZ0QmC; ocDownloadStarted=gxairgqzc" \
     --header="Connection: keep-alive" \ 
     --header="Referer: https://libdrive.ethz.ch/" \
     "https://libdrive.ethz.ch/index.php/s/VPio6i5UlXTgir0/download?path=%2F&files=biomes_splits_to_name.pkl&downloadStartSecret=gxairgqzc" -O Data/biomes_splits_to_name.pkl
