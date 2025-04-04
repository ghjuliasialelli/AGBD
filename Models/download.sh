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
    wget "https://libdrive.ethz.ch/index.php/s/VPio6i5UlXTgir0/download?path=%2Fdata_subset-${year}-v4_${i}-20.h5" -O Data/data_subset-${year}-v4_${i}-20.h5
  done

done

# Download tabular data
for mode in "${modes[@]}"
do
  wget "https://libdrive.ethz.ch/index.php/s/Y9A9b156b8H0KYf/download?path=%2F${mode}_features-v6.csv" -O Data/${mode}_features-v6.csv
done

# Download the statistics
wget "https://libdrive.ethz.ch/index.php/s/Y9A9b156b8H0KYf/download?path=%2Fstatistics_subset_2019-2020-v4_new.pkl" -O Data/statistics_subset_2019-2020-v4_new.pkl

# Download the train/test/val split
wget "https://libdrive.ethz.ch/index.php/s/VPio6i5UlXTgir0/download?path=%2Fbiomes_splits_to_name.pkl" -O Data/biomes_splits_to_name.pkl
